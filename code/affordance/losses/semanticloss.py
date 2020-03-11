from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class SemanticLoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(SemanticLoss, self).__init__()
        self.weight_sad = torch.tensor([15/189, 189/189, 10/189, 27/189, 14/189])
        self.criterion = nn.CrossEntropyLoss(weight = self.weight_sad)

    def forward(self, output, target, target_weight = None):
        # right now target_weight is useless
        # batch_size = output.size(0)
        # num_joints = output.size(1)
        # print(output.shape)
        # print(target.shape)

        loss = self.criterion(output, target)

        return loss
