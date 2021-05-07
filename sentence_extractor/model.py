import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

from GAT import ALLGAT
import time

class GraphX(nn.Module):
    def __init__(self, args, vocab_obj, device):
        super().__init__()

        self.m_device = device

        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num
        self.m_feature_num = vocab_obj.feature_num
        self.m_total_sent_num = vocab_obj.sent_num
        self.m_train_sent_num = vocab_obj.train_sent_num

        print("user num", self.m_user_num)
        print("item num", self.m_item_num)
        print("feature num", self.m_feature_num)
        print("total sent num", self.m_total_sent_num)
        print("train sent num", self.m_train_sent_num)

        self.m_user_embed = nn.Embedding(self.m_user_num, args.user_embed_size)
        self.m_item_embed = nn.Embedding(self.m_item_num, args.item_embed_size)

        self.m_feature_embed = nn.Embedding(self.m_feature_num, args.feature_embed_size)

        # self.m_feature_embed = vocab_obj.m_fid2fembed
        self.m_feature_embed_size = args.feature_embed_size
        self.f_load_feature_embedding(vocab_obj.m_fid2fembed)

        self.m_sent_embed = nn.Embedding(self.m_train_sent_num, args.sent_embed_size)
        # self.m_sent_embed = vocab_obj.m_sid2sembed
        self.m_sent_embed_size = args.sent_embed_size
        self.f_load_sent_embedding(vocab_obj.m_sid2sembed)

        self.sent_state_proj = nn.Linear(args.sent_embed_size, args.hidden_size, bias=False)
        self.feature_state_proj = nn.Linear(args.feature_embed_size, args.hidden_size, bias=False)
        self.user_state_proj = nn.Linear(args.user_embed_size, args.hidden_size, bias=False)
        self.item_state_proj = nn.Linear(args.item_embed_size, args.hidden_size, bias=False)

        self.m_gat = ALLGAT(in_dim=args.hidden_size, out_dim=args.hidden_size, head_num=args.head_num, attn_drop_out=args.attn_dropout_rate, ffn_inner_hidden_size=args.ffn_inner_hidden_size, ffn_drop_out=args.ffn_dropout_rate)
        ### node classification
        # self.output_hidden_size = args.output_hidden_size
        # self.wh = nn.Linear(self.output_hidden_size * 2, 2)
        # self.wh = nn.Linear(args.hidden_size, 2)
        self.wh = nn.Linear(args.hidden_size, 1)

        self._n_iter = 2

        self.f_initialize()

        self = self.to(self.m_device)

    def f_initialize(self):
        nn.init.uniform_(self.m_user_embed.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.m_item_embed.weight, a=-1e-3, b=1e-3)

        nn.init.uniform_(self.sent_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.feature_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.user_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.item_state_proj.weight, a=-1e-3, b=1e-3)

        nn.init.uniform_(self.wh.weight, a=-1e-3, b=1e-3)


    def f_load_feature_embedding(self, pre_feature_embed):

        pre_feature_embed_weight = []
        print("pre_feature_embed", len(pre_feature_embed))

        for f_idx in range(self.m_feature_num):
            feature_embed_i = pre_feature_embed[f_idx]
            pre_feature_embed_weight.append(feature_embed_i)

        self.m_feature_embed.weight.data.copy_(torch.Tensor(pre_feature_embed_weight))
        self.m_feature_embed.weight.requires_grad = False

    def f_load_sent_embedding(self, pre_sent_embed):

        print("pre_sent_embed", len(pre_sent_embed))
        print("sent num", self.m_train_sent_num)

        pre_sent_embed_weight = []

        for s_idx in range(self.m_train_sent_num):
            sent_embed_i = pre_sent_embed[s_idx]
            pre_sent_embed_weight.append(sent_embed_i)

        print("pre_sent_embed_weight", len(pre_sent_embed_weight))
        print("sent num", self.m_train_sent_num)

        self.m_sent_embed.weight.data.copy_(torch.Tensor(pre_sent_embed_weight))
        self.m_sent_embed.weight.requires_grad = False

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

        # start_time = time.time()

        fnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        fsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"]==0)

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

        inode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 3)

        unode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)

        # end_time = time.time()
        # print("duration 0", end_time-start_time)

        fid = graph.nodes[fnode_id].data["id"]
        f_embed = self.m_feature_embed(fid)

        sid = graph.nodes[snode_id].data["id"]
        s_embed = self.m_sent_embed(sid)

        iids = graph.nodes[inode_id].data["id"]
        i_embed = self.m_item_embed(iids)

        uids = graph.nodes[unode_id].data["id"]
        u_embed = self.m_user_embed(uids)

        etf = graph.edges[fsedge_id].data["tffrac"]

        ### normalize the tfidf to get the weight
        # ? why this is normalize
        graph.edges[fsedge_id].data["weight"] = etf

        # supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # end_time = time.time()
        # print("duration 1", end_time-start_time)

        # ### feature, sentence,
        # feature_embed = self.set_fnembed(graph)
        # sent_init_state = self.sent_state_proj(self.set_snembed(graph))
        # user_embed = self.set_unembed(graph)

        f_init_state = self.feature_state_proj(f_embed)
        graph.nodes[fnode_id].data["init_state"] = f_init_state

        # item_embed = self.set_inembed(graph)
        item_init_state = self.item_state_proj(i_embed)
        graph.nodes[inode_id].data["init_state"] = item_init_state

        user_init_state = self.user_state_proj(u_embed)
        # print("user size", user_init_state.size())
        graph.nodes[unode_id].data["init_state"] = user_init_state

        ### user, item node init
        sent_init_state = self.sent_state_proj(s_embed)
        graph.nodes[snode_id].data["init_state"] = sent_init_state

        # end_time = time.time()
        # print("duration 2", end_time-start_time)

        x = graph.ndata["init_state"]
        h = self.m_gat(graph, x)

        graph.ndata["hidden_state"] = h
        # sent_state = self.feature2sent(graph, feature_state, sent_state)

        # end_time = time.time()
        # print("duration 3", end_time-start_time)

        # for i in range(self._n_iter):
        #     start_time = time.time()
        #     ### sent -> feature
        #     feature_state = self.sent2feature(graph, feature_state, sent_state)
        #     end_time = time.time()
        #     print("duration 4", end_time-start_time)

        #     ### feature -> sent
        #     sent_state = self.feature2sent(graph, feature_state, sent_state)

        # graph.nodes[supernode_id].data["hidden_state"] = sent_state

        s_state_list = []
        for snid in snode_id:
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        logits = self.wh(s_state)

        # print("logits", logits.size())

        return logits

