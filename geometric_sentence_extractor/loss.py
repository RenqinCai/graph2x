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
        self.m_criterion = nn.BCEWithLogitsLoss(reduction="mean")

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

    def forward(self, G):

        soft_label_num = 4
        loss_list = []
        for i, g in enumerate(G):
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
            labels = g.ndata["label"][snode_id]
            logits = g.ndata["p"][snode_id]

            log_prob = []
            for soft_label_idx in range(1, soft_label_num):
                
                pos_mask = (labels == soft_label_idx)
                neg_mask = (labels < soft_label_idx)

                pos_logits = logits[pos_mask]
                neg_logits = logits[neg_mask]


                if pos_logits.size()[0] == 0:
                    continue

                if neg_logits.size()[0] == 0:
                    continue
                
                delta_logits = pos_logits.unsqueeze(1)-neg_logits

                log_prob.append(F.logsigmoid(delta_logits).mean().unsqueeze(-1))

            log_prob = torch.cat(log_prob, dim=-1)
            # log_prob = torch.tensor(log_prob).to(self.m_device)

            loss_list.append(log_prob.mean().unsqueeze(-1))

        # loss = torch.tensor(loss_list).to(self.m_device)
        loss = -torch.cat(loss_list, dim=-1)
        loss = loss.mean()

        return loss

           
