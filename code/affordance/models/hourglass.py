'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']

class Bottleneck(nn.Module):
    '''
    A residual module
    '''
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

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


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64 # feature dim after "input -> conv" at start ?
        self.num_feats = 128
        self.num_stacks = num_stacks

        # # input channel number. 3 (RGB) + 1 (depth)
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)

        
        #  input channel number. 3 (RGB) + 1 (depth) + 1 (last_pred_mask)
        # self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3,
                            #    bias=True)

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

        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) # reverse feature_num of fc 
        self.score_ = nn.ModuleList(score_)  # reverse feature_num of score 

        ### prev mask
        self.last_mask_layer = nn.Conv2d(ch + 1, ch, kernel_size=1, bias=True)
        # self.dropout_layer = nn.Dropout(p=0.75) # if have bn, these is no need of dropout

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

    '''
    new_tsm_feature [batch_size, 2, 256, 64, 64]
    1 : first stack (50 % + 50 %)
    2 : second stack (50 % + 50 %)
    '''
    def forward(self, x, tsm_flag = None, new_tsm_feature = None):      
        if tsm_flag:
            tsm_0_feature = new_tsm_feature[:, 0]
            tsm_1_feature = new_tsm_feature[:, 1]


        out = []
        out_test = [] # semantic
        out_tsm_feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x) # layer before stacked hourgasss model
        x = self.layer3(x) # layer before stacked hourgasss model # [B, 256, 64, 64]
        
        for i in range(self.num_stacks):
            if not tsm_flag :
                out_tsm_feature.append(x)
            else :
                if i == 0 :
                    x = x * 0.5 + tsm_0_feature * 0.5
                elif i == 1 :
                    x = x * 0.5 + tsm_1_feature * 0.5
                else :
                    pass # x = x
            y = self.hg[i](x) # x will be residual data in last line # [B, 256, 64, 64]
            y = self.res[i](y)
            y = self.fc[i](y) # 256 x 64 x 64

            score = self.score[i](y) # blue block in hourglass paper
            ## 2020.3.1 for IoU loss
            score = self.sigmoid(score)
            out.append(score) # for computing intermediate loss
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y) # middle feature project back
                score_ = self.score_[i](score) # logits feature project back
                x = x + fc_ + score_ # identity + fc_  + score_ 

        return out, out_tsm_feature
        
        # semantic
        # return out, out_test




def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
