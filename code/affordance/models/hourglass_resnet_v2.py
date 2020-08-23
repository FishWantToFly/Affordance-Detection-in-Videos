'''
2020.8.5
Remove region prediciton.

'''
import torch.nn as nn
import torch.nn.functional as F
import torch

from .convlstm import ConvLSTM
from .squeeze_and_excitation import ChannelSELayer

from torchvision import models, transforms

__all__ = ['HourglassNet_Resnet_v2', 'hg_resnet_v2']

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet50(pretrained=True)
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output) # torch.Size([1, 256, 56, 56])
        '''
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        '''
        return output

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

class HourglassNet_Resnet_v2(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet_Resnet_v2, self).__init__()

        self.inplanes = 64 # feature dim after "input -> conv" at start ?
        self.num_feats = 128
        self.num_stacks = num_stacks

        # input channel number. 3 (RGB) + 1 (depth)
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # layer1 to layer3 are placed in front of stacked hourglass model
        self.layer1 = self._make_residual(block, self.inplanes, 1) # block = bottleneck
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
        # ResNet-50
        self.feature_model = net().cuda().eval()

        # Step 1 
        # stack for step = 1 right now
        self.num_stacks_step_1 = 1

        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.num_stacks_step_1):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < self.num_stacks_step_1-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) # reverse feature_num of fc 
        self.score_ = nn.ModuleList(score_)  # reverse feature_num of score 

        # Step 2
        self.convLSTM = ConvLSTM(input_dim=256,
                 hidden_dim=[64, 64, 64],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 lstm_state='stateful')
        self.residual = self._make_residual_v2(block, 64, 32, 1)
        self.SE_layer = ChannelSELayer(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.fc_1 = nn.Linear(32 * 16 * 16, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 1)

        # TSM module
        self.conv_tsm = nn.Conv2d(256, 256, kernel_size=1, bias=True)

        # 2020.8.7 
        self.conv_att_to_label = nn.Conv2d(257, 256, kernel_size=1, bias=True) # position 1
        # self.conv_att_to_label = nn.Conv2d(65, 64, kernel_size=1, bias=True) # position 2


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

    # for residual module
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
        return nn.Sequential(*layers)

    def forward(self, x, input_state = None, tsm_input = None):
        out_heatmap = []
        out_mask = []
        
        x = self.feature_model(x)
        x = F.interpolate(x, size = (64, 64))

        #################
        '''
        TSM module here
        online setting
        '''
        
        c = x.shape[1]
        # with tsm
        if tsm_input == None :
            tsm_input = torch.zeros((x.shape[0], c // 8, 64, 64)).cuda()

        x1, x2 = x[:, : c // 8], x[:, c // 8:]
        x = x + self.conv_tsm(torch.cat((tsm_input, x2), dim=1))
        tsm_output = x1
        
        # no tsm
        # tsm_output = torch.zeros((x.shape[0], c // 8, 64, 64)).cuda()
        #################


        x_before_step_1 = x

        # step 1 : affordance atention heatmap
        for i in range(self.num_stacks_step_1):
            y = self.hg[i](x) # x will be residual data in last line # [B, 256, 64, 64]
            y = self.res[i](y)
            y = self.fc[i](y) # 256 x 64 x 64

            score = self.score[i](y) # blue block in hourglass paper
            score = self.sigmoid(score)
            out_heatmap.append(score) # for computing intermediate loss

            if i < self.num_stacks_step_1-1:
                fc_ = self.fc_[i](y) # middle feature project back
                score_ = self.score_[i](score) # logits feature project back
                x = x + fc_ + score_ # identity + fc_  + score_
        
        affordance_attention_heatmap = out_heatmap[-1] # [B, 1, 64, 64]

        # step 2 : affordance existence predictor
        x = x_before_step_1 # [B, 256, 64, 64]

        ### TESTING NOW
        # '''
        x = torch.cat((x, affordance_attention_heatmap), 1) # [B, 257, 64, 64]
        x = self.conv_att_to_label(x)
        # '''

        x, output_state = self.convLSTM(x, input_state = input_state)

        # '''
        # Attention module + residual module
        original_x = x # [B, 64, 64, 64]
        x = self.residual(x) 
        x = self.SE_layer(x) # [B, 64, 64, 64] # SE block and scale back to x
        x = x + original_x # [B, 64, 64, 64]
        # '''

        # 2.1 Predict affordance existence label
        x = self.conv_2(x)
        x = self.conv_3(x) # [B, 32, 16, 16]
        x = self.relu(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x) 
        x = self.sigmoid(x)
        out_label = x

        return out_heatmap, out_label, output_state, tsm_output



def hg_resnet_v2(**kwargs):
    model = HourglassNet_Resnet_v2(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
