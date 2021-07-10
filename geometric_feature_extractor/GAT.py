import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class GATNET(torch.nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout_rate):
        super(GATNET, self).__init__()

        self.gat_layer_1 = GATConv(in_dim, out_dim, heads=head_num, dropout=dropout_rate)
        # default, concat all attention head
        self.gat_layer_2 = GATConv(head_num*out_dim, out_dim, heads=1, concat=False, dropout=dropout_rate)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat_layer_1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat_layer_2(x, edge_index)

        return x

