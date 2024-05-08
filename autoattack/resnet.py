import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride = 1, downsample=None):

        super(BasicBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.nonlinear1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.nonlinear2 = nn.ReLU(inplace = True)

        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size = 1, stride = self.stride, bias = False),
                nn.BatchNorm2d(self.out_planes)
                )

    def forward(self, x):

        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.nonlinear2(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, num_block_list = [2, 2, 2, 2], block=BasicBlock, num_classes = 10, **kwargs):

        super(ResNet, self).__init__()

        self.num_block_list = num_block_list
        self.in_planes = 64
        self.num_classes = num_classes
        print('ResNet: num_block_list = %s, num_class = %d' % (self.num_block_list, num_classes))

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block = block, out_planes = 64, num_blocks = num_block_list[0], stride = 1)
        self.layer2 = self._make_layer(block = block, out_planes = 128, num_blocks = num_block_list[1], stride = 2)
        self.layer3 = self._make_layer(block = block, out_planes = 256, num_blocks = num_block_list[2], stride = 2)
        self.layer4 = self._make_layer(block = block, out_planes = 512, num_blocks = num_block_list[3], stride = 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

    def _make_layer(self, block, out_planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))
        self.in_planes = out_planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18(num_classes=10):
    return ResNet([2, 2, 2, 2], block=BasicBlock, num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet([3, 4, 6, 3], block=BasicBlock, num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet([3, 4, 6, 3], block=BottleNeck, num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet([3, 4, 23, 3], block=BottleNeck, num_classes=num_classes)


