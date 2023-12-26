import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from src.cnn.residual_blocks import BasicBlock, BottleNeck

class ResNet(nn.Module):

    def _conv_x(self,
                channels: int,
                blocks: int,
                strides,
                index: int):
        list_strides = [strides] + [1] * (blocks - 1)
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = f'block_{index}_{i}'
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.message == 'basic' else channels * 4
        return conv_x

    def __init__(self,
                 block: Union[BasicBlock, BottleNeck],
                 groups: List[int],
                 num_classes=1000):
        super(ResNet, self).__init__()
        self.channels = 64
        self.block = block
        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 512 if self.block.message == "basic" else 512 * 4
        self.fc = nn.Linear(patches, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out
    

def resnet_18(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)
