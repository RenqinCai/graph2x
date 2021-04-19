import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
import dgl
from dgl.data.utils import save_graphs, load_graphs


class Vocab():
    def __init__(self):

        self.m_user2uid = None
        self.m_item2iid = None

        self.m_user_num = 0
        self.m_item_num = 0

        self.m_feature2fid = None
        self.m_feature_num = 0

        self.m_sent2sid = None
        self.m_sent_num = 0

        self.m_fid2fembed = None
        self.m_sid2sembed = None
        
    def f_set_user2uid_vocab(self, user2uid):
        self.m_user2uid = user2uid
        self.m_user_num = len(self.m_user2uid)

    def f_set_item2iid_vocab(self, item2iid):
        self.m_item2iid = item2iid
        self.m_item_num = len(self.m_item2iid)

    def f_set_feature2fid_vocab(self, feature2fid):
        self.m_feature2fid = feature2fid
        self.m_feature_num = len(self.m_feature2fid)

    def f_set_sent2sid_vocab(self, sent2sid):
        self.m_sent2sid = sent2sid
        self.m_sent_num = len(self.m_sent2sid)

    def f_set_id_vocab(self, user2uid, item2iid, feature2fid, sent2sid):
        self.m_user2uid = user2uid
        self.m_item2iid = item2iid
        self.m_feature2fid = feature2fid
        self.m_sent2sid = sent2sid
        
        self.m_user_num = len(self.m_user2uid)
        self.m_item_num = len(self.m_item2iid)
        self.m_feature_num = len(self.m_feature2fid)
        self.m_sent_num = len(self.m_sent2sid)
        
    def f_set_embed_vocab(self, fid2fembed, sid2sembed):
        if sid2sembed != None:
            self.m_fid2fembed = fid2fembed
        if sid2sembed != None:
            self.m_sid2sembed = sid2sembed

    @property
    def user_num(self):
        return self.m_user_num
    
    @property
    def item_num(self):
        return self.m_item_num

    @property
    def feature_num(self):
        return self.m_feature_num

    @property
    def sent_num(self):
        return self.m_sent_num


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class RATEBEER(Dataset):
    def __init__(self):
        super().__init__()

        self.m_uid2fid2tfidf_dict = {}
        self.m_iid2fid2tfidf_dict = {}
        self.m_sid2fid2tfidf_dict = {}

        self.m_uid_list = []
        self.m_iid_list = []
        self.m_cdd_sid_list_list = []
        self.m_label_sid_list_list = []

    def load_sent_feature(self, vocab, sent_feature_file):
        ### sent_feature {sentid: {featureid: feature tf-idf}}
        sid2fid2tfidf_dict = {}

        sentid2fid2tfidf = readJson(sent_feature_file)
        sent_num = len(sentid2fid2tfidf)

        sent2sid_dict = vocab.m_sent2sid

        feature2fid_dict = {}

        if sent_num != vocab.sent_num:
            print("sent num error", sent_num, vocab.sent_num)

        for i in range(sent_num):
            data_i = sentid2fid2tfidf[i]

            sentid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[sentid_i]

            sid_i = sent2sid_dict[sentid_i]
            if sid_i not in sid2fid2tfidf_dict:
                sid2fid2tfidf_dict[sid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    feature2fid_dict[feautreid_ij] = len(feature2fid_dict)
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                sid2fid2tfidf_dict[sid_i][fid_ij] = tfidf_ij

        feature2fid_dict["PAD"] = len(feature2fid_dict)
        vocab.m_pad_fid = feature2fid_dict["PAD"]
        vocab.f_set_feature2fid_vocab(feature2fid_dict)

        self.m_sid2fid2tfidf_dict = sid2fid2tfidf_dict

    def load_user_feature(self, vocab, user_feature_file):
        ### user_feature {userid: {featureid: feature tf-idf}}
        uid2fid2tfidf_dict = {}

        userid2fid2tfidf = readJson(user_feature_file)
        user_num = len(userid2fid2tfidf)

        user2uid_dict = vocab.m_user2uid
        feature2fid_dict = vocab.m_feature2fid

        if user_num != vocab.user_num:
            print("user num error", user_num, vocab.user_num)

        for i in range(user_num):
            data_i = userid2fid2tfidf[i]

            userid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[userid_i]

            uid_i = user2uid_dict[userid_i]
            if uid_i not in uid2fid2tfidf_dict:
                uid2fid2tfidf_dict[uid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    print("error missing feature", userid_i, feautreid_ij)
                    continue
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                uid2fid2tfidf_dict[uid_i][fid_ij] = tfidf_ij

        self.m_uid2fid2tfidf_dict = uid2fid2tfidf_dict
        
    def load_item_feature(self, vocab, item_feature_file):
        ### item_feature {itemid: {featureid: feature tf-idf}}
        iid2fid2tfidf_dict = {}

        itemid2fid2tfidf = readJson(item_feature_file)
        item_num = len(itemid2fid2tfidf)

        if item_num != vocab.item_num:
            print("item num error", item_num, vocab.item_num)

        item2iid_dict = vocab.m_item2iid
        feature2fid_dict = vocab.m_feature2fid

        for i in range(item_num):
            data_i = item2iid_dict[i]

            itemid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[itemid_i]

            iid_i = item2iid_dict[itemid_i]
            if iid_i not in iid2fid2tfidf_dict:
                iid2fid2tfidf_dict[iid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    print("error missing feature", itemid_i, feautreid_ij)
                    continue
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                iid2fid2tfidf_dict[iid_i][fid_ij] = tfidf_ij

        self.m_iid2fid2tfidf_dict = iid2fid2tfidf_dict

    def load_useritem_cdd_label_sent(self, useritem_candidate_label_sent_file):
        #### read pair data 
        user2uid_dict = {}
        item2iid_dict = {}
        sent2sid_dict = {}

        uid_list = []
        iid_list = []
        cdd_sid_list_list = []
        label_sid_list_list = []

        #### useritem_sent {userid: {itemid: [cdd_sentid] [label_sentid]}}
        useritem_cdd_label_sent = readJson(useritem_candidate_label_sent_file)
        useritem_cdd_label_sent_num = len(useritem_cdd_label_sent)
        print("useritem_cdd_label_sent_num", useritem_cdd_label_sent_num)

        for i in range(useritem_cdd_label_sent_num):
            data_i = useritem_cdd_label_sent[i]

            userid_i = list(data_i.keys())[0]
            itemid_list_i = list(data_i[userid_i].keys())

            for itemid_ij in itemid_list_i:
                cdd_sentid_list_ij = data_i[userid_i][itemid_ij][0]

                if userid_i not in user2uid_dict:
                    user2uid_dict[userid_i] = len(user2uid_dict)

                if itemid_ij not in item2iid_dict:
                    item2iid_dict[itemid_ij] = len(item2iid_dict)

                cdd_sid_list_i = []
                for sentid_ijk in cdd_sentid_list_ij:
                    if len(sentid_ijk) == 0:
                        continue
                        
                    if sentid_ijk not in sent2sid_dict:
                        sent2sid_dict[sentid_ijk] = len(sent2sid_dict)
                    cdd_sid_list_i.append(sent2sid_dict[sentid_ijk])

                label_sentid_list_ij = data_i[userid_i][itemid_ij][1]
                label_sid_list_i = []

                for sentid_ijk in label_sentid_list_ij:
                    if len(sentid_ijk) == 0:
                        continue
                        
                    if sentid_ijk not in sent2sid_dict:
                        sent2sid_dict[sentid_ijk] = len(sent2sid_dict)
                    label_sid_list_i.append(sent2sid_dict[sentid_ijk])

                uid_i = user2uid_dict[userid_i]
                iid_i = item2iid_dict[itemid_ij]
                
                uid_list.append(uid_i)
                iid_list.append(iid_i)
                cdd_sid_list_list.append(cdd_sid_list_i)
                label_sid_list_list.append(label_sid_list_i)

        sent2sid_dict["PAD"] = len(sent2sid_dict)
        self.m_pad_sid = sent2sid_dict["PAD"]

        self.m_uid_list = uid_list
        self.m_iid_list = iid_list
        self.m_cdd_sid_list_list = cdd_sid_list_list
        self.m_label_sid_list_list = label_sid_list_list        

        vocab_obj = Vocab()
        vocab_obj.f_set_user2uid_vocab(user2uid_dict)
        vocab_obj.f_set_item2iid_vocab(item2iid_dict)
        vocab_obj.f_set_sent2sid_vocab(sent2sid_dict)

        return vocab_obj

    def load_sent_embed(self, vocab, sent_embed_file):
        ### sid 2 embed
        sid2embed_dict = {}
        sent2sid_dict = vocab.sent2sid_dict

        sent_embed = readJson(sent_embed_file)
        sent_embed_num = len(sent_embed)
        print("sent num", sent_embed_num)

        #### sent_embed {sentid: embed}
        for i in range(sent_embed_num):
            data_i = sent_embed[i]

            sentid_i = list(data_i.keys())[0]
            sentembed_i = data_i[sentid_i]

            if sentid_i not in sent2sid_dict:
                print("error missing sent", sentid_i)
                continue
            
            sid_i = sent2sid_dict[sentid_i]
            if sid_i not in sid2embed_dict:
                sid2embed_dict[sid_i] = sentembed_i
        
        vocab.f_set_embed_vocab(None, sid2embed_dict)

    def load_feature_embed(self, vocab, feature_embed_file):
        fid2embed_dict = {}
        feature2fid_dict = vocab.feature2fid_dict

        feature_embed = readJson(feature_embed_file)
        feature_embed_num = len(feature_embed)
        print("feature_embed_num", feature_embed_num)

        for i in range(feature_embed_num):
            data_i = feature_embed[i]

            featureid_i = list(data_i.keys())[0]
            featureembed_i = data_i[featureid_i]

            if featureid_i not in feature2fid_dict:
                print("error missing feature", featureid_i)
                continue

            fid_i = feature2fid_dict[featureid_i]
            if fid_i not in fid2embed_dict:
                fid2embed_dict[fid_i] = featureembed_i

        vocab.f_set_embed_vocab(fid2embed_dict, None)

    def load_train_data(self, useritem_candidate_label_sent_file, sent_feature_file, user_feature_file, item_feature_file, feature_embed_file, sent_embed_file):

        vocab_obj = self.load_useritem_cdd_label_sent(useritem_candidate_label_sent_file)
        self.load_sent_feature(vocab_obj, sent_feature_file)
        self.load_user_feature(vocab_obj, user_feature_file)
        self.load_item_feature(vocab_obj, item_feature_file)

        self.load_feature_embed(vocab_obj, feature_embed_file)
        self.load_sent_embed(vocab_obj, sent_embed_file)

        print("... load train data ...", len(self.m_uid_list), len(self.m_iid_list), len(self.m_cdd_sid_list_list))

        return vocab_obj

    def load_useritem_cdd_label_sent_eval(self, vocab, useritem_candidate_label_sent_file):
        #### read pair data 
        user2uid_dict = vocab.m_user2uid
        item2iid_dict = vocab.m_item2iid
        sent2sid_dict = vocab.m_sent2sid

        uid_list = []
        iid_list = []
        cdd_sid_list_list = []
        label_sid_list_list = []

        #### useritem_sent {userid: {itemid: [cdd_sentid] [label_sentid]}}
        useritem_cdd_label_sent = readJson(useritem_candidate_label_sent_file)
        useritem_cdd_label_sent_num = len(useritem_cdd_label_sent)
        print("useritem_cdd_label_sent_num", useritem_cdd_label_sent_num)

        for i in range(useritem_cdd_label_sent_num):
            data_i = useritem_cdd_label_sent[i]

            userid_i = list(data_i.keys())[0]
            itemid_list_i = list(data_i[userid_i].keys())

            for itemid_ij in itemid_list_i:
                cdd_sentid_list_ij = data_i[userid_i][itemid_ij][0]

                if userid_i not in user2uid_dict:
                    print("user id missing", userid_i)
                    continue

                if itemid_ij not in item2iid_dict:
                    print("item id missing", itemid_ij)
                    continue

                cdd_sid_list_i = []
                for sentid_ijk in cdd_sentid_list_ij:
                    if len(sentid_ijk) == 0:
                        continue
                        
                    if sentid_ijk not in sent2sid_dict:
                        print("sent id missing", sentid_ijk)
                        continue

                    cdd_sid_list_i.append(sent2sid_dict[sentid_ijk])

                label_sentid_list_ij = data_i[userid_i][itemid_ij][1]
                label_sid_list_i = []

                for sentid_ijk in label_sentid_list_ij:
                    if len(sentid_ijk) == 0:
                        continue
                        
                    if sentid_ijk not in sent2sid_dict:
                        print("sent id missing", sentid_ijk)
                        continue
                        
                    label_sid_list_i.append(sent2sid_dict[sentid_ijk])

                uid_i = user2uid_dict[userid_i]
                iid_i = item2iid_dict[itemid_ij]
                
                uid_list.append(uid_i)
                iid_list.append(iid_i)
                cdd_sid_list_list.append(cdd_sid_list_i)
                label_sid_list_list.append(label_sid_list_i)

        sent2sid_dict["PAD"] = len(sent2sid_dict)
        self.m_pad_sid = sent2sid_dict["PAD"]

        self.m_uid_list = uid_list
        self.m_iid_list = iid_list
        self.m_cdd_sid_list_list = cdd_sid_list_list
        self.m_label_sid_list_list = label_sid_list_list        

    def load_sent_feature_eval(self, vocab, sent_feature_file):
        ### sent_feature {sentid: {featureid: feature tf-idf}}
        sid2fid2tfidf_dict = {}

        sentid2fid2tfidf = readJson(sent_feature_file)
        sent_num = len(sentid2fid2tfidf)

        sent2sid_dict = vocab.m_sent2sid

        feature2fid_dict = vocab.m_feature2fid

        if sent_num != vocab.sent_num:
            print("sent num error", sent_num, vocab.sent_num)

        for i in range(sent_num):
            data_i = sentid2fid2tfidf[i]

            sentid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[sentid_i]

            sid_i = sent2sid_dict[sentid_i]
            if sid_i not in sid2fid2tfidf_dict:
                sid2fid2tfidf_dict[sid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    print("error missing feature", feautreid_ij)
                    continue
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                sid2fid2tfidf_dict[sid_i][fid_ij] = tfidf_ij

        self.m_sid2fid2tfidf_dict = sid2fid2tfidf_dict

    def load_eval_data(self, vocab, useritem_candidate_label_sent_file, sent_feature_file, user_feature_file, item_feature_file, feature_embed_file, sent_embed_file):
        self.load_useritem_cdd_label_sent_eval(vocab, useritem_candidate_label_sent_file)
        self.load_sent_feature(vocab, sent_feature_file)
        self.load_user_feature(vocab, user_feature_file)
        self.load_item_feature(vocab, item_feature_file)
        
    def __len__(self):
        return len(self.m_uid_list)

    def add_feature_node(self, G, uid, iid, sid_list):
        fid2nid = {}
        nid2fid = {}

        nid = 0
        fid2tfidf_dict_user = self.m_uid2fid2tfidf_dict[uid]
        for fid in fid2tfidf_dict_user:
            if fid not in fid2nid.keys():
                fid2nid[fid] = nid
                nid2fid[nid] = fid

                nid += 1

        fid2tfidf_dict_item = self.m_iid2fid2tfidf_dict[iid]
        for fid in fid2tfidf_dict_item:
            if fid not in fid2nid.keys():
                fid2nid[fid] = nid
                nid2fid[nid] = fid

                nid += 1
        
        for sid in sid_list:
            fid2tfidf_dict_sent = self.m_sid2fid2tfidf_dict[sid]

            for fid in fid2tfidf_dict_sent:
                if fid not in fid2nid.keys():
                    fid2nid[fid] = nid
                    nid2fid[nid] = fid

                    nid += 1

        fid_node_num = len(nid2fid)

        G.add_nodes(fid_node_num)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(fid_node_num)
        G.ndata["id"] = torch.LongTensor(list(nid2fid.values()))
        G.ndata["dytpe"] = torch.zeros(fid_node_num)

        return fid2nid, nid2fid

    def create_graph(self, uid, iid, sid_list, label_list):
        G = dgl.DGLGraph()
        
        ### add feature nodes
        fid2nid, nid2fid = self.add_feature_node(G, uid, iid, sid_list)
        feature_node_num = len(fid2nid)
        
        ### add sent nodes
        sent_node_num = len(sid_list)
        sid2nid = {}
        nid2sid = {}
        for i in range(sent_node_num):
            sid_i = sid_list[i]
            nid_i = feature_node_num+i

            sid2nid[sid_i] = nid_i
            nid2sid[nid_i] = sid_i

        G.add_nodes(sent_node_num)
        G.ndata["unit"][feature_node_num:] = torch.ones(sent_node_num)
        G.ndata["dtype"][feature_node_num:] = torch.ones(sent_node_num)
        G.ndata["id"][feature_node_num:] = torch.LongTensor(list(nid2sid.values()))

        feat_sent_node_num = feature_node_num+sent_node_num

        ### add user, item nodes
        ### add user node

        user_node_num = 1
        G.add_nodes(user_node_num)
        G.ndata["unit"][feat_sent_node_num:] = torch.ones(user_node_num)
        G.ndata["dtype"][feat_sent_node_num:] = torch.ones(user_node_num)*2
        G.ndata["id"][feat_sent_node_num:] = torch.LongTensor([uid])

        uid2nid = {uid:feat_sent_node_num}
        nid2uid = {feat_sent_node_num:uid}
    
        ### add item noe
        item_node_num = 1
        G.add_nodes(item_node_num)
        G.ndata["unit"][feat_sent_node_num+user_node_num:] = torch.ones(item_node_num)
        G.ndata["dtype"][feat_sent_node_num+user_node_num:] = torch.ones(item_node_num)*3
        G.ndata["id"][feat_sent_node_num+user_node_num:] = torch.LongTensor([iid])
        
        iid2nid = {iid:feat_sent_node_num+user_node_num}
        nid2iid = {feat_sent_node_num+user_node_num:iid}
        
        ### add edges from sents to features
        for i in range(sent_node_num):
            sid = sid_list[i]
            nid_s = sid2nid[sid]
            fid2tfidf_dict_sent = self.m_sid2fid2tfidf_dict[sid]
            for fid in fid2tfidf_dict_sent:
                nid_f = fid2nid[fid]
                tfidf_sent = fid2tfidf_dict_sent[fid]
                G.add_edge(nid_f, nid_s, data={"tffrac": torch.LongTensor([tfidf_sent]), "dtype": torch.Tensor([0])})
                G.add_edge(nid_s, nid_f, data={"tffrac": torch.LongTensor([tfidf_sent]), "dtype": torch.Tensor([0])})
    
        for i in range(user_node_num):
            nid_u = uid2nid[0]
            fid2tfidf_dict_user = self.m_uid2fid2tfidf_dict[uid]

            for fid in fid2tfidf_dict_user:
                nid_f = fid2nid[fid]
                tfidf_user = fid2tfidf_dict_user[fid]
                G.add_edge(nid_f, nid_u, data={"tffrac": torch.LongTensor([tfidf_user]), "dtype": torch.Tensor([0])})
                G.add_edge(nid_u, nid_f, data={"tffrac": torch.LongTensor([tfidf_user]), "dtype": torch.Tensor([0])})

        for i in range(item_node_num):
            nid_i = iid2nid[0]
            fid2tfidf_dict_item = self.m_iid2fid2tfidf_dict[iid]

            for fid in fid2tfidf_dict_item:
                nid_f = fid2nid[fid]
                tfidf_item = fid2tfidf_dict_item[fid]
                G.add_edge(nid_f, nid_i, data={"tffrac": torch.LongTensor([tfidf_item]), "dtype": torch.Tensor([0])})
                G.add_edge(nid_i, nid_f, data={"tffrac": torch.LongTensor([tfidf_item]), "dtype": torch.Tensor([0])})
        
        G.set_e_initializer(dgl.init.zero_initializer)

        label = np.zeros(sent_node_num)
        labelid_list = [sid2nid_dict[i] for i in label_list]
        label[np.array(labelid_list)] = 1

        G.nodes[sid2nid].data["label"] = torch.LongTensor(label)

        return G

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i = idx

        uid_i = self.m_uid_list[i]
        iid_i = self.m_iid_list[i]
        cdd_sid_list_i = self.m_cdd_sid_list_list[i]
        label_sid_list_i = self.m_label_sid_list_list[i]

        G = self.create_graph(uid_i, iid_i, cdd_sid_list_i, label_sid_list_i)

        return G, i

    def get_example(self, idx):
        i = idx

        uid_i = self.m_uid_list[i]
        iid_i = self.m_iid_list[i]
        cdd_sid_list_i = self.m_cdd_sid_list_list[i]
        label_sid_list_i = self.m_label_sid_list_list[i]

        example = {"user":uid_i, "item":iid_i, "cdd_sid":cdd_sid_list_i, "label_sid":label_sid_list_i}

        return example


    

