import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GATStackLayer import MultiHeadLayer, PositionwiseFeedForward, GATLayer


class ALLGAT(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out):
        super().__init__()
        self.layer = MultiHeadLayer(in_dim, int(out_dim / head_num), head_num, attn_drop_out, layer=GATLayer)
        self.layer_1 = MultiHeadLayer(out_dim, int(out_dim / head_num), head_num, attn_drop_out, layer=GATLayer)
        # self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, x):
        h = x

        h = self.layer(g, h)

        h = F.elu(h)
        # h = x + h

        h = self.layer_1(g, h)
        # h = F.elu(self.layer_1(g, h))
        # h = x + h

        # h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h
