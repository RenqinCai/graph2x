import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

from GAT import WSWGAT

class GraphX(nn.Module):
    def __init__(self, args, vocab_obj, device):
        super().__init__()

        self.m_device = device

        user_num = vocab_obj.user_num
        item_num = vocab_obj.item_num
        feature_num = vocab_obj.feature_num

        self.m_user_embed = nn.Embedding(user_num, args.user_embed_size)
        self.m_item_embed = nn.Embedding(item_num, args.item_embed_size)

        self.m_feature_embed = vocab_obj.m_fid2fembed
        self.m_feature_embed_size = args.feature_embed_size

        self.m_sent_embed = vocab_obj.m_sid2sembed
        self.m_sent_embed_size = args.sent_embed_size

        self.sent_state_proj = nn.Linear(args.sent_embed_size, args.hidden_size)
        self.user_state_proj = nn.Linear(args.user_embed_size, args.hidden_size, bias=False)
        self.item_state_proj = nn.Linear(args.item_embed_size, args.hidden_size, bias=False)

        self.m_n_iter = args.n_iter

        self.feature2sent = WSWGAT(in_dim=args.hidden_size, out_dim=args.hidden_size, head_num=args.head_num, attn_drop_out=args.attn_dropout_rate, ffn_inner_hidden_size=args.ffn_inner_hidden_size, ffn_drop_out=args.ffn_dropout_rate, layer_type="W2S")

        self.sent2feature = WSWGAT(in_dim=args.hidden_size, out_dim=args.hidden_size, head_num=args.head_num, attn_drop_out=args.attn_dropout_rate, ffn_inner_hidden_size=args.ffn_inner_hidden_size, ffn_drop_out=args.ffn_dropout_rate, layer_type="S2W")

        ### node classification
        # self.output_hidden_size = args.output_hidden_size
        # self.wh = nn.Linear(self.output_hidden_size * 2, 2)
        self.wh = nn.Linear(args.hidden_size, 2)

        self = self.to(self.m_device)

    def set_fnembed(self, graph):
        fnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        fsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"]==0)

        fid = graph.nodes[fnode_id].data["id"]
        f_embed = self.m_feature_embed(fid)

        graph.nodes[fnode_id].data["embed"] = f_embed
        etf = graph.edges[fsedge_id].data["tffrac"]
        
        ### normalize the tfidf to get the weight
        graph.edges[fsedge_id].data["weight"] = etf

        return f_embed

    def set_snembed(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
        
        sid = graph.nodes[snode_id].data["id"]
        s_embed = self.m_sent_embed(sid)

        return s_embed
        
    def set_unembed(self, graph):
        unode_id_list = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_embed_list = []
        snid2unid = {}

        for unode_id in unode_id_list:
            fnodes = [nid for nid in graph.predecessors(unode_id) if graph.nodes[nid].data["dtype"] == 0]
            u_embed_f = graph.nodes[fnodes].data["init_state"].mean(dim=0)
            assert not torch.any(torch.isnan(u_embed)), "user embed element"

            uid = graph.nodes[unode_id].data["id"]
            u_embed_b = self.m_user_embed(uid)

            u_embed = u_embed_b+u_embed_f
            # u_embed = torch.cat([u_embed_f, u_embed_b])

            node_embed_list.append(u_embed)

            # for s in snodes:
            #     snid2unid[int(s)] = unode_id

        node_embed = torch.stack(node_embed_list)

        return node_embed

    def set_inembed(self, graph):
        inode_id_list = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 3)
        node_embed_list = []
        snid2iid = {}

        for inode_id in inode_id_list:
            fnodes = [nid for nid in graph.predecessors(inode_id) if graph.nodes[nid].data["dtype"] == 0]
            i_embed_f = graph.nodes[fnodes].data["init_state"].mean(dim=0)
            assert not torch.any(torch.isnan(i_embed)), "item embed element"

            iid = graph.nodes[inode_id].data["id"]
            i_embed_b = self.m_item_embed(iid)

            i_embed = i_embed_b+i_embed_f

            node_embed_list.append(i_embed)

        node_embed = torch.stack(node_embed_list)

        return node_embed

    def forward(self, graph):
        """
        graph: [batch_size]*DGLGraph
        node: feature: unit=0, dtype=0, id=(int) featureid in vocab
            sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            user: unit=1, dtype=2 
            item: unit=1, dtype=3
        edge:
            feature2sent, sent2feature: tffrac=int, type=0
            feature2user, user2feature: tffrac=int, type=0
            feature2item, item2feature: tffrac=int, type=0
        return: 
            [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        unode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        inode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 3)

        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        ### feature, sentence, 
        feature_embed = self.set_fnembed(graph)
        sent_init_state = self.sent_state_proj(self.set_snembed(graph))

        ### user, item node init
        graph.nodes[snode_id].data["init_state"] = sent_init_state

        user_embed = self.set_unembed(graph)
        user_init_state = self.user_state_proj(user_embed)
        graph.nodes[unode_id].data["init_state"] = user_init_state

        item_embed = self.set_inembed(graph)
        item_init_state = self.item_state_proj(item_embed)
        graph.nodes[inode_id].data["init_state"] = item_init_state

        feature_init_state = feature_embed
        sent_init_state = graph.nodes[supernode_id].data["init_state"]

        feature_state = feature_init_state
        sent_state = self.feature2sent(graph, feature_init_state, sent_init_state)

        for i in range(self._n_iter):

            ### sent -> feature
            feature_state = self.sent2feature(graph, feature_state, sent_state)

            ### feature -> sent
            sent_state = self.feature2sent(graph, feature_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        s_state_list = []
        for snid in snode_id:
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        result = self.wh(s_state)

        return result