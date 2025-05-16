from einops import rearrange
import numbers
import math
import torch
import torch.nn.functional as F
from torch import nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


def d4_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def d3_to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def d5_to_3d(x):
    return rearrange(x, 'b c s h w -> b (s h w) c')


def d3_to_5d(x, s, h, w):
    x = rearrange(x, 'b (s h w) c -> b c s h w', s=s, h=h, w=w)
    return x


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight
        if self.mu_sigma:
            return x, mu, sigma
        else:
            return x


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)
        self.out_dir = out_dir

    def forward(self, x):
        h, w = x.shape[-2:]
        x = d4_to_3d(x)

        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_4d(x, h, w), d3_to_4d(mu, h, w), d3_to_4d(sigma, h, w)
        else:
            x = self.body(x)
            return d3_to_4d(x, h, w)


def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    if isinstance(padder_size, int):
        padder_size_h = padder_size
        padder_size_w = padder_size
    else:
        padder_size_h, padder_size_w = padder_size
    mod_pad_h = (padder_size_h - h % padder_size_h) % padder_size_h
    mod_pad_w = (padder_size_w - w % padder_size_w) % padder_size_w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x


class WAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, padding_mode='zeros'):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        self.GMLP = nn.Sequential(
            nn.Linear(window_size**2, window_size**2, bias=True),
            nn.GELU(),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                        h=self.window_size, w=self.window_size)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out * self.GMLP(v)
        out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                        w1=Wx//self.window_size, h=self.window_size, w=self.window_size)

        return out[:, :, :H, :W]

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out


class DFFreq(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8):
        super(DFFreq, self).__init__()

        self.dim = dim
        self.window_size = window_size

        self.dwt_transform = DWTForward(J=1, wave='haar')
        self.idwt_transform = DWTInverse(wave='haar')

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.dwt_attn = WAttention(dim, num_heads, bias, window_size=window_size)
        self.fft_attn = WAttention(dim, num_heads, bias, window_size=8)
        self.fft_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # LoSFB
        x_dwt_l, x_dwt_h = self.dwt_transform(x)
        x_dwt = torch.cat([x_dwt_l.unsqueeze(2), x_dwt_h[0]], dim=2)
        bs, _, _, hh, ww = x_dwt.shape
        dwt_img = rearrange(x_dwt, 'bs c c1 (h1 h) (w1 w) -> bs c c1 (h1 w1 h w)', h1=hh//2, w1=ww//2, h=2, w=2)
        x_attn = self.dwt_attn(self.norm1(dwt_img))
        x_dwt = dwt_img + x_attn
        x_dwt = rearrange(x_dwt, 'bs c c1 (h1 w1 h w)-> bs c c1 (h1 h) (w1 w)', h1=hh//2, w1=ww//2, h=2, w=2)
        x_idwt = self.idwt_transform((x_dwt[:, :, 0], [x_dwt[:, :, 1:]]))

        # GloFB
        _, _, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_fft)

        pha = torch.angle(x_fft)
        pha = self.fft_attn(pha)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_fft = torch.complex(real, imag)
        x_ifft = torch.fft.irfft2(x_fft, s=(H, W), norm='backward') * self.fft_scale

        x = x_idwt + x_ifft * 0.2

        return x
