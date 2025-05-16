import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clip import clip
import math
import copy
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class LoRALayer(nn.Module):
    """
    LoRA层的实现：添加低秩矩阵以高效微调大型模型
    """

    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        # 缩放系数，决定了LoRA部分的权重
        self.scale = alpha / rank

        # 创建LoRA的低秩矩阵A和B
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)

        # 初始化：下投影随机初始化，上投影初始化为0
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # 计算LoRA的贡献并返回
        return self.lora_up(self.lora_down(x)) * self.scale


# ---------------------- 2. CLIP的LoRA包装器 ----------------------

class CLIPWithLoRA(nn.Module):
    """
    为CLIP的图像编码器添加LoRA适配层的完整实现
    """

    def __init__(self, base_clip_model, lora_rank=8, lora_alpha=16):
        super().__init__()
        # 保存原始的CLIP模型
        self.clip = base_clip_model

        # 冻结原始CLIP的所有参数
        for param in self.clip.parameters():
            param.requires_grad = False

        # 分析CLIP的图像编码器结构
        # 对于ViT-L/14，每个Transformer块有三个投影矩阵
        # 我们将为每个投影矩阵添加LoRA层
        self.image_encoder = self.clip.visual

        # 检查并确定这是一个Vision Transformer结构
        assert hasattr(self.image_encoder, 'transformer'), "不支持的CLIP视觉编码器类型"

        # 创建LoRA层字典
        self.lora_layers = nn.ModuleDict()

        # 获取隐藏层维度（对于ViT-L/14，通常是1024）
        if hasattr(self.image_encoder, 'width'):
            hidden_dim = self.image_encoder.width
        else:
            hidden_dim = self.image_encoder.transformer.width

        # 为每个Transformer块的注意力层添加LoRA
        for block_idx, block in enumerate(self.image_encoder.transformer.resblocks):
            # 为QKV矩阵添加LoRA
            # 注意：根据OpenAI的CLIP实现，通常QKV是在一个单一的投影中
            attn = block.attn

            # 添加LoRA到Query投影
            self.lora_layers[f"block_{block_idx}_q"] = LoRALayer(
                hidden_dim, hidden_dim,
                rank=lora_rank, alpha=lora_alpha
            )

            # 添加LoRA到Key投影
            self.lora_layers[f"block_{block_idx}_k"] = LoRALayer(
                hidden_dim, hidden_dim,
                rank=lora_rank, alpha=lora_alpha
            )

            # 添加LoRA到Value投影
            self.lora_layers[f"block_{block_idx}_v"] = LoRALayer(
                hidden_dim, hidden_dim,
                rank=lora_rank, alpha=lora_alpha
            )

            # 修改注意力的前向传播方法
            self._patch_attention_forward(block.attn, block_idx)

    def _patch_attention_forward(self, attn_module, block_idx):
        """
        通过钩子技术修改注意力层的前向传播，插入LoRA计算
        """
        # 保存原始的前向传播方法
        original_forward = attn_module.forward

        # 保存self引用以在闭包中使用
        _self = self

        # 定义新的前向传播方法
        def lora_forward(self, query, key, value, need_weights=False, **kwargs):
            # 获取模型尺寸
            B, N, C = query.shape

            # 原始的QKV投影
            original_qkv = self.in_proj_weight

            # 将输入分成三部分用于q、k、v计算
            qkv = F.linear(query, original_qkv, self.in_proj_bias)
            qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 应用LoRA修改到q、k、v
            q = q + _self.lora_layers[f"block_{block_idx}_q"](query)
            k = k + _self.lora_layers[f"block_{block_idx}_k"](query)
            v = v + _self.lora_layers[f"block_{block_idx}_v"](query)

            # 继续原始注意力计算
            self.scale = q.size(-1) ** -0.5
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            # attn = self.attn_dropout(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.out_proj(x)
            # x = self.out_dropout(x)

            if need_weights:
                return x, attn
            return x

        # 替换原始的前向传播方法
        attn_module.forward = lora_forward.__get__(attn_module, type(attn_module))

    def forward(self, image):
        """
        模型的前向传播，使用LoRA修改的CLIP进行图像编码
        """
        # 直接使用修改后的CLIP模型
        image_features = self.clip.encode_image(image)

        # 返回归一化的图像特征
        return image_features
