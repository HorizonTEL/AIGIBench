import torch.nn as nn
import math
import torch
from collections import OrderedDict
from torch.utils import model_zoo
from torch.nn import functional as F
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class TinyLaDeDa(nn.Module):
    def __init__(self, block, layer, stride, kernel, preprocess_type, num_classes=1, pool=True):
        self.inplanes = 8
        super(TinyLaDeDa, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 8, blocks=layer, stride=stride, kernel3=kernel, prefix='layer1')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(8 * block.expansion, num_classes)
        self.pool = pool
        self.block = block
        self.preprocess_type = preprocess_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, kernel_size=kernel))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    def preprocess(self, x, preprocess_type):
        if preprocess_type == "raw":
            return x
        if preprocess_type == "NPR":
            return x - self.interpolate(x, 0.5)
        grad_kernel = None
        # Define kernels for gradients in x, y, and diagonal directions
        if preprocess_type == "x_grad":
            grad_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if preprocess_type == "y_grad":
            grad_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if preprocess_type == "left_diag":
            grad_kernel = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32,
                                        device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if preprocess_type == "right_diag":
            grad_kernel = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32,
                                        device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)



        grad_representation = F.conv2d(x, grad_kernel, groups=3, padding="same")
        return grad_representation



    def forward(self, x):
        x = self.preprocess(x, self.preprocess_type)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 2 conv layers
        x = self.layer1(x)
        if self.pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)
            x = x.permute(0, 3, 1, 2)
        return x


def tiny_ladeda(preprocess_type, **kwargs):
    model = TinyLaDeDa(Bottleneck, layer=1, stride=2, kernel=1, preprocess_type=preprocess_type, **kwargs)
    return model


