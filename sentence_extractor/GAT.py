import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from GATStackLayer import MultiHeadLayer, WSGATLayer, SWGATLayer, PositionwiseFeedForward, GATLayer


class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / head_num), head_num, attn_drop_out, layer=WSGATLayer)
        elif layer_type == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / head_num), head_num, attn_drop_out, layer=SWGATLayer)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s):
        if self.layer_type == "W2S":
            origin, neighbor = s, w
        elif self.layer_type == "S2W":
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

class ALLGAT(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out):
        super().__init__()
        
        self.layer = MultiHeadLayer(in_dim, int(out_dim / head_num), head_num, attn_drop_out, layer=GATLayer)
       
        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, x):
        h = x
        h = F.elu(self.layer(g, h))
        h = x + h
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h