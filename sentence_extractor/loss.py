import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

class SIG_LOSS(nn.Module):
    def __init__(self, device):
        super(SIG_LOSS, self).__init__()

        self.m_device = device
        self.m_criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        loss = self.m_criterion(preds, targets)

        return loss

class XE_LOSS(nn.Module):
    def __init__(self, item_num, device):
        super(XE_LOSS, self).__init__()
        self.m_item_num = item_num
        self.m_device = device
    
    def forward(self, preds, targets):
        # print("==="*10)
        # print(targets.size())
        targets = F.one_hot(targets, self.m_item_num)

        # print("target", targets.size())

        # print(targets.size())
        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0

        preds = F.log_softmax(preds, 1)
        xe_loss = torch.sum(preds*targets, dim=-1)

        xe_loss = -torch.mean(xe_loss)

        # exit()

        return xe_loss

class BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(BPR_LOSS, self).__init__()
        self.m_device = device

    def forward(self, preds):
        log_prob = F.logsigmoid(preds).mean()
        # print("log_prob", log_prob)
        
        loss = -log_prob

        return loss