import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

from GAT import GATNET
import time
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


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

        self.m_gat = GATNET(in_dim=args.hidden_size, out_dim=args.hidden_size, head_num=args.head_num, dropout_rate=args.attn_dropout_rate)
        ### node classification
        # self.output_hidden_size = args.output_hidden_size
        # self.wh = nn.Linear(self.output_hidden_size * 2, 2)
        # self.wh = nn.Linear(args.hidden_size, 2)
        self.wh = nn.Linear(args.hidden_size, 1)

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

    def forward(self, graph_batch):
        ## init node embeddings
        # print("graph_batch", graph_batch.num_graphs)
        batch_size = graph_batch.num_graphs
        ##### feature node
        # for g in graph_batch:
        # for g_idx, g in enumerate(graph_batch):
        # for i in range(batch_size):
        #     g = graph_batch[i]
        #     # print("g", g)
        #     fid = g.f_rawid
        #     f_embed = self.m_feature_embed(fid)
        #     f_node_embed = self.feature_state_proj(f_embed)

        #     ##### sentence node
        #     sid = g.s_rawid
        #     s_embed = self.m_sent_embed(sid)
        #     s_node_embed = self.sent_state_proj(s_embed)

        #     ##### item node
        #     itemid = g.i_rawid
        #     item_embed = self.m_item_embed(itemid)
        #     item_node_embed = self.item_state_proj(item_embed)

        #     ##### user node
        #     userid = g.u_rawid
        #     user_embed = self.m_user_embed(userid)
        #     user_node_embed = self.user_state_proj(user_embed)

        #     # print("f size", f_embed.size(), f_node_embed.size())
        #     # print("s size", s_embed.size(), s_node_embed.size())
        #     # print("item size", item_embed.size(), item_node_embed.size())
        #     # print("user size", user_embed.size(), user_node_embed.size())

        #     x = torch.cat([f_node_embed, s_node_embed, user_node_embed, item_node_embed], dim=0)

        #     g["x"] = x
        #     print("g x", x.size(), g.x.size())
        
        fid = graph_batch.f_rawid
        f_embed = self.m_feature_embed(fid)
        f_node_embed = self.feature_state_proj(f_embed)

        ##### sentence node
        sid = graph_batch.s_rawid
        s_embed = self.m_sent_embed(sid)
        s_node_embed = self.sent_state_proj(s_embed)

        ##### item node
        itemid = graph_batch.i_rawid
        item_embed = self.m_item_embed(itemid)
        item_node_embed = self.item_state_proj(item_embed)

        ##### user node
        userid = graph_batch.u_rawid
        user_embed = self.m_user_embed(userid)
        user_node_embed = self.user_state_proj(user_embed)

        # print("f size", f_embed.size(), f_node_embed.size())
        # print("s size", s_embed.size(), s_node_embed.size())
        # print("item size", item_embed.size(), item_node_embed.size())
        # print("user size", user_embed.size(), user_node_embed.size())

        x = torch.cat([f_node_embed, s_node_embed, user_node_embed, item_node_embed], dim=0)
        graph_batch["x"] = x

        ## go through GAT
        #### hidden: node_num*hidden_size
        print("x size", graph_batch.x.size())

        hidden = self.m_gat(graph_batch.x, graph_batch.edge_index)

        hidden_batch, _ = to_dense_batch(hidden, graph_batch.batch)
        print("hidden_batch size", hidden_batch.size())

        hidden_s_batch = []

        ## fetch sentence hidden vectors
        for batch_idx in range(batch_size):
        # for g_idx, g in enumerate(graph_batch):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            s_nid = g.s_nid

            hidden_s_g_i = hidden_g_i[s_nid]

            hidden_s_batch.append(hidden_s_g_i)

        hidden_s_batch = torch.cat(hidden_s_batch, dim=0)
        logits = self.wh(hidden_s_batch)

        ### make predictions

        return logits

    def eval_forward(self, graph_batch):
        ## init node embeddings

        ##### feature node
        # for g_idx, g in enumerate(graph_batch):
        #     fid = g.f_rawid
        #     f_embed = self.m_feature_embed(fid)

        #     ##### sentence node
        #     sid = g.s_rawid
        #     s_embed = self.m_sent_embed(sid)

        #     ##### item node
        #     itemid = g.i_rawid
        #     item_embed = self.m_item_embed(itemid)

        #     ##### user node
        #     userid = g.u_rawid
        #     user_embed = self.m_user_embed(userid)

        #     x = torch.cat([f_embed, s_embed, item_embed, user_embed], dim=0)

        #     g["x"] = x
        
        batch_size = graph_batch.num_graphs

        fid = graph_batch.f_rawid
        f_embed = self.m_feature_embed(fid)
        f_node_embed = self.feature_state_proj(f_embed)

        ##### sentence node
        sid = graph_batch.s_rawid
        s_embed = self.m_sent_embed(sid)
        s_node_embed = self.sent_state_proj(s_embed)

        ##### item node
        itemid = graph_batch.i_rawid
        item_embed = self.m_item_embed(itemid)
        item_node_embed = self.item_state_proj(item_embed)

        ##### user node
        userid = graph_batch.u_rawid
        user_embed = self.m_user_embed(userid)
        user_node_embed = self.user_state_proj(user_embed)

        # print("f size", f_embed.size(), f_node_embed.size())
        # print("s size", s_embed.size(), s_node_embed.size())
        # print("item size", item_embed.size(), item_node_embed.size())
        # print("user size", user_embed.size(), user_node_embed.size())

        x = torch.cat([f_node_embed, s_node_embed, user_node_embed, item_node_embed], dim=0)
        graph_batch["x"] = x

        ## go through GAT
        #### hidden: node_num*hidden_size
        hidden = self.m_gat(graph_batch.x, graph_batch.edge_index)

        hidden_batch, mask_batch = hidden.to_dense_batch()

        hidden_s_batch = []
        sid_batch = []
        mask_s_batch = []
        target_s_batch = []

        max_s_num_batch = 0
        for batch_idx in range(batch_size):
        # for g_idx, g in enumerate(graph_batch):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            s_nid = g.s_nid
            s_num = s_nid.size(0)

            max_s_num_batch = max(max_s_num_batch, s_num)

        ## fetch sentence hidden vectors
        # for g_idx, g in enumerate(graph_batch):
        #     hidden_g_i = hidden_batch[g_idx]

        for batch_idx in range(batch_size):
        # for g_idx, g in enumerate(graph_batch):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            s_nid = g.s_nid
            s_num = s_nid.size(0)
            pad_s_num = max_s_num_batch-s_num

            hidden_s_g_i = hidden_g_i[s_nid]
            pad_s_g_i = torch.zeros(pad_s_num, hidden_s_g_i.size(1)).to(self.m_device) 
            hidden_pad_s_g_i = torch.cat([hidden_s_g_i, pad_s_g_i], dim=1)

            hidden_s_batch.append(hidden_pad_s_g_i.unsqueeze(0))

            sid = g.s_rawid
            pad_sid_i = torch.zeros(pad_s_num).to(self.m_device)
            sid_pad_i = torch.cat([sid, pad_sid_i], dim=-1)

            sid_batch.append(sid_pad_i.unsqueeze(0))

            mask_s = torch.zeros(max_s_num_batch).to(self.m_device)
            mask_s[:s_num] = 1
            mask_s_batch.append(mask_s.unsqueeze(0))

            target_sid = g.gt_label
            target_s_batch.append(target_sid)

        #### hidden_s_batch: batch_size*max_sen_num*hidden_size
        hidden_s_batch = torch.cat(hidden_s_batch, dim=0)
        
        ### sid_batch: batch_size*max_s_num_batch
        sid_batch = torch.cat(sid_batch, dim=0)

        ### mask_s_batch: batch_size*max_s_num_batch
        mask_s_batch = torch.cat(mask_s_batch, dim=0)

        ### make predictions
        #### logits: batch_size*max_sen_num*1
        logits = self.wh(hidden_s_batch)
        logits = logits.squeeze(-1)
        logits = F.sigmoid(logits, dim=-1)*mask_s_batch

        return logits, sid_batch, mask_s_batch, target_s_batch