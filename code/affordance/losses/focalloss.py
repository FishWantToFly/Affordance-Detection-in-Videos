from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, inputs, targets, target_weight = None):
        F_loss = -(targets >= 0.5).float() *self.alpha*((1.-inputs)**self.gamma)*torch.log(inputs+1e-8)\
                        -(1.-(targets >= 0.5).float())*(1.-self.alpha)*(inputs** self.gamma)*torch.log(1.-inputs+1e-8)
                    
        return torch.mean(F_loss)
