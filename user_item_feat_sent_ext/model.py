import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
import random
import time


class FeatSentExt(nn.Module):
    def __init__(self, args, vocab_obj, device, sent_num, feature_emb, sentence_emb):
        super().__init__()
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num
        self.m_feature_num = vocab_obj.vocab_size   # including <unk> and <pad>
        self.m_train_sent_num = sent_num
        self.m_device = device

        print("user num: {}".format(self.m_user_num))
        print("item num: {}".format(self.m_item_num))
        print("feature num (include <unk>, <pad>): {}".format(self.m_feature_num))

        self.m_user_embed_size = args.user_embed_size           # default: 256
        self.m_item_embed_size = args.item_embed_size           # default: 256
        self.m_feature_embed_size = args.feature_embed_size     # default: 256
        self.m_sent_embed_size = args.sent_embed_size           # default: 768

        self.m_feature_finetune_flag = args.feat_finetune       # default: True
        self.m_sentence_finetune_flag = args.sent_finetune      # default: False

        self.m_user_embed = nn.Embedding(self.m_user_num, args.user_embed_size)
        self.m_item_embed = nn.Embedding(self.m_item_num, args.item_embed_size)
        # Load Pre-trained Feature Embedding
        self.m_feature_embed = nn.Embedding(self.m_feature_num, args.feature_embed_size)
        self.f_load_feature_embedding(feature_emb)
        print("Pre-trained Feature Embedding Loaded.")
        # Load Pre-trained Sentence Embedding
        self.m_sent_embed = nn.Embedding(self.m_train_sent_num, args.sent_embed_size)
        self.f_load_sent_embedding(sentence_emb)
        print("Pre-trained Sentence Embedding Loaded.")

        self.m_concat_embed_size = self.m_user_embed_size + self.m_item_embed_size \
            + self.m_feature_embed_size + self.m_sent_embed_size
        self.fc = nn.Linear(self.m_concat_embed_size, 1)

        self.init_weights()
        self = self.to(self.m_device)

    def init_weights(self):
        nn.init.uniform_(self.m_user_embed.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.m_item_embed.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)

    def f_load_feature_embedding(self, pre_feature_embed):

        pre_feature_embed_weight = []
        print("pre_feature_embed", len(pre_feature_embed))

        for f_idx in range(self.m_feature_num):
            feature_embed_i = pre_feature_embed[f_idx]
            pre_feature_embed_weight.append(feature_embed_i)

        self.m_feature_embed.weight.data.copy_(torch.Tensor(pre_feature_embed_weight))

        if not self.m_feature_finetune_flag:
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
        if not self.m_sentence_finetune_flag:
            self.m_sent_embed.weight.requires_grad = False

    def forward(self, user, item, sentence, feature, feature_len):
        # user/item/sentence: (batch_size)
        # feature: (batch_size, feature_len), batch_first
        # feature_len: (batch_size)

        batch_size = user.shape[0]
        max_feat_length = feature.shape[1]

        user_embed = self.m_user_embed(user)        # shape: (batch_size, user_emb_size)
        item_embed = self.m_item_embed(item)        # shape: (batch_size, item_emb_size)
        sent_embed = self.m_sent_embed(sentence)    # shape: (batch_size, sent_emb_size)
        # Need to Get the Avg Embedding of Features
        feat_embed = self.m_feature_embed(feature)  # shape: (batch_size, feature_len, feature_emb_size)
        # Construct a mask matrix with shape of (batch_size, feature_len)
        mask = torch.zeros(batch_size, max_feat_length)
        for i in range(batch_size):
            feat_len_i = feature_len[i].item()
            mask[i][:feat_len_i] = 1
        mask = mask.unsqueeze(dim=-1)
        mask = mask.to(self.m_device)
        feat_embed = feat_embed * mask              # shape: (batch_size, feature_len, feature_emb_size)
        feat_embed = torch.sum(feat_embed, dim=1)   # shape: (batch_size, feature_emb_size)
        feat_embed = feat_embed / feature_len.unsqueeze(dim=-1)

        # Concat the 4 embeddings
        # shape: (batch_size, user_emb_size+item_emb_size+sent_emb_size+feature_emb_size)
        concat_embed = torch.cat((user_embed, item_embed, sent_embed, feat_embed), dim=-1)

        # Feed into a Linear Layer
        # shape: (batch_size, 1)
        output = self.fc(concat_embed)

        return output

    def eval_forward(self, user, item, sentence, feature, feature_len):
        # user/item/sentence: (batch_size)
        # feature: (batch_size, feature_len), batch_first
        # feature_len: (batch_size)

        batch_size = user.shape[0]
        max_feat_length = feature.shape[1]

        user_embed = self.m_user_embed(user)        # shape: (batch_size, user_emb_size)
        item_embed = self.m_item_embed(item)        # shape: (batch_size, item_emb_size)
        sent_embed = self.m_sent_embed(sentence)    # shape: (batch_size, sent_emb_size)
        # Need to Get the Avg Embedding of Features
        feat_embed = self.m_feature_embed(feature)  # shape: (batch_size, feature_len, feature_emb_size)
        # Construct a mask matrix with shape of (batch_size, feature_len)
        mask = torch.zeros(batch_size, max_feat_length)
        for i in range(batch_size):
            feat_len_i = feature_len[i].item()
            mask[i][:feat_len_i] = 1
        mask = mask.unsqueeze(dim=-1)
        mask = mask.to(self.m_device)
        feat_embed = feat_embed * mask              # shape: (batch_size, feature_len, feature_emb_size)
        feat_embed = torch.sum(feat_embed, dim=1)   # shape: (batch_size, feature_emb_size)
        feat_embed = feat_embed / feature_len.unsqueeze(dim=-1)

        # Concat the 4 embeddings
        # shape: (batch_size, user_emb_size+item_emb_size+sent_emb_size+feature_emb_size)
        concat_embed = torch.cat((user_embed, item_embed, sent_embed, feat_embed), dim=-1)

        # Feed into a Linear Layer and then feed into a sigmoid layer
        # shape: (batch_size, 1)
        output = self.fc(concat_embed)
        m = nn.Sigmoid()
        output = m(output)

        return output
