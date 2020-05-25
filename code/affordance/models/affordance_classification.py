'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

from .convlstm import ConvLSTM
from .squeeze_and_excitation import ChannelSELayer

__all__ = ['AffordanceClassificationNet', 'ACNet']

class Bottleneck(nn.Module):
    '''
    A residual module
    '''
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # print(inplanes)
        # print(planes)
        # print()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print(residual.shape)
        # print(out.shape)
        # print()

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x) # also a residual block, will be bypassed to deeper layer
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1) # called by recursive
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class AffordanceClassificationNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(AffordanceClassificationNet, self).__init__()

        self.inplanes = 64 # feature dim after "input -> conv" at start ?
        self.num_feats = 128
        self.num_stacks = num_stacks

        # # input channel number. 3 (RGB) + 1 (mask)
        # self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                            #    bias=True)
        
        #  input channel number. 3 (RGB) + 1 (depth) + 1 (mask)
        self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # layer1 to layer3 are placed in front of stacked hourglass model
        self.layer1 = self._make_residual(block, self.inplanes, 1) # block = bottleneck
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.sigmoid = nn.Sigmoid()

        ### semantic
        # self.test_layer_1 = nn.Conv2d(256, 64, kernel_size=1, bias=True)
        # self.fc_test = nn.Linear(64 * 32 * 32, 5)

        self.convLSTM = ConvLSTM(input_dim=256,
                 hidden_dim=[64, 32, 64],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
        self.residual = self._make_residual_v2(block, 64, 32, 1)
        self.SE_layer = ChannelSELayer(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.fc_1 = nn.Linear(32 * 16 * 16, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 1)


    def _make_residual_v2(self, block, in_planes, planes, blocks, stride=1):
        '''
        If blocks = 1 : equal to generate a single residual module
        If blokcs > 1 : a residual module appends with more resiual modules
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(in_planes, planes, stride, downsample))
        # in_planes = planes * block.expansion
        # for i in range(1, blocks):
        #     layers.append(block(in_planes, planes))

        return nn.Sequential(*layers)

    def _make_residual(self, block, planes, blocks, stride=1):
        '''
        If blocks = 1 : equal to generate a single residual module
        If blokcs > 1 : a residual module appends with more resiual modules
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        '''
        1x1 conv
        '''
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x, input_last_state = None):
        # out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.maxpool(x)
        x = self.layer2(x) # layer before stacked hourgasss model
        x = self.layer3(x) # layer before stacked hourgasss model # [B, 256, 64, 64]

        x, last_state = self.convLSTM(x, input_last_state) 


        '''
        2020.5.15 test to move out attention module
        '''
        original_x = x # [B, 64, 64, 64]
        x = self.residual(x) 
        x = self.SE_layer(x) # [B, 64, 64, 64] # SE block and scale back to x
        x = x + original_x # [B, 64, 64, 64]

        # classfication head
        x = self.conv_2(x)
        x = self.conv_3(x) # [B, 32, 16, 16]
        x = self.relu(x)
        x = x.view(-1, 32 * 16 * 16)
        # print(x.shape)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x) 
        x = self.sigmoid(x)
        out = x
        return out, last_state


def ACNet(**kwargs):
    model = AffordanceClassificationNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
