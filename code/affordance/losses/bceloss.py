from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, output, target, target_weight = None):
        loss = self.criterion(output, target)
        return loss
