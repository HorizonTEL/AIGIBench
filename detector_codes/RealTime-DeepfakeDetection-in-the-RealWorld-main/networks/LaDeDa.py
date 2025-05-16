import torch.nn as nn
import math
import torch
from collections import OrderedDict
from torch.nn import functional as F
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False)  # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class LaDeDa(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], preprocess_type="NPR", num_classes=1, pool=True):
        self.inplanes = 64
        super(LaDeDa, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    def preprocess(self, x, grad_type):
        if grad_type == "raw":
            return x
        if grad_type == "NPR":
            return x - self.interpolate(x, 0.5)
        grad_kernel = None
        # Define kernels for gradients in x, y, and diagonal directions
        if grad_type == "x_grad":
            grad_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if grad_type == "y_grad":
            grad_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if grad_type == "left_diag":
            grad_kernel = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32,
                                        device=x.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        if grad_type == "right_diag":
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # in case you want to return image-level deepfake score
        if self.pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        # in case you want to return patch-wise deepfake scores
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)
            x = x.permute(0, 3, 1, 2)
        return x


def LaDeDa33(strides=[2, 2, 2, 1], **kwargs):
    model = LaDeDa(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 1], **kwargs)
    return model


def LaDeDa17(strides=[2, 2, 2, 1], **kwargs):
    model = LaDeDa(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 0], **kwargs)
    return model


def LaDeDa9(strides=[2, 2, 2, 1], **kwargs):
    model = LaDeDa(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 0, 0], **kwargs)
    return model

def LaDeDa5(strides=[2, 2, 2, 1], **kwargs):
    model = LaDeDa(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 0, 0, 0], **kwargs)
    return model

