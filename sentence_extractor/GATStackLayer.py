import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import dgl

class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, attn_drop_out, layer, merge='cat'):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(head_num):
            self.heads.append(layer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            result = torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            result = torch.mean(torch.stack(head_outs))
        return result

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        weight_range = 1e-3
        nn.init.xavier_normal_(self.w_1.weight, gain=weight_range)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output

class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        # self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2* out_dim, 1, bias=False)

    def edge_attention(self, edges):

        ### use tfidf as edge features
        # dfeat = self.feat_fc(edges.data["weight"])                  # [edge_num, out_dim]

        ### aggregate node features, edge features to node representations
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]

        # print('wa', wa.size())
        
        ### combine tf-idf
        tfidf_edge_weight = edges.data["weight"]
        # print("edges weight tfidf", tfidf_edge_weight.size())
        tfidf_edge_weight = tfidf_edge_weight.view(-1, 1)

        wa = tfidf_edge_weight*wa

        # print("wa", wa.size())

        return {'e': wa}

    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        e = nodes.mailbox['e']
        # print("e", e.size())
        alpha = F.softmax(e, dim=1)

        # print("nodes size", nodes.size())
        # print("alpha", alpha.size())

        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}


    def forward(self, g, h):

        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        g.nodes[wnode_id].data['z'] = z

        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]

class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        # self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # start_time = time.time()

        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... edge attent duration 0", duration)

        ### combine tf-idf
        # wa = F.softmax(edges.data["weight"]*wa, dim=-1)

        tfidf_edge_weight = edges.data["weight"]
        tfidf_edge_weight = tfidf_edge_weight.view(-1, 1)
        wa = tfidf_edge_weight*wa

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... edge attent duration 1", duration)

        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # start_time = time.time()
        

        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # end_time = time.time()
        # duration = end_time - start_time
        # print("+++ reduce duration 2", duration)

        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        print("=== gat ==")
        start_time = time.time()

        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)

        end_time = time.time()
        duration = end_time - start_time
        print("... duration 0", duration)

        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        end_time = time.time()
        duration = end_time - start_time
        print("... duration 1", duration)

        g.pull(wnode_id, self.message_func, self.reduce_func)
        end_time = time.time()
        duration = end_time - start_time
        print("... duration 2", duration)

        g.ndata.pop('z')
        h = g.ndata.pop('sh')

        end_time = time.time()
        duration = end_time - start_time
        print("... duration 3", duration)
        return h[wnode_id]

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        # self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # start_time = time.time()

        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... edge attent duration 0", duration)

        ### combine tf-idf
        # wa = F.softmax(edges.data["weight"]*wa, dim=-1)

        tfidf_edge_weight = edges.data["weight"]
        tfidf_edge_weight = tfidf_edge_weight.view(-1, 1)
        wa = tfidf_edge_weight*wa

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... edge attent duration 1", duration)

        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # start_time = time.time()
        

        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # end_time = time.time()
        # duration = end_time - start_time
        # print("+++ reduce duration 2", duration)

        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        # print("=== gat ==")
        # start_time = time.time()

        z = self.fc(h)

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... duration 0", duration)

        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        # end_time = time.time()
        # duration = end_time - start_time
        # print("... duration 1", duration)

        g.update_all(self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')

        # end_time = time.time()
        # duration = end_time - start_time
        # print("... duration 3", duration)
        return h