import os
import io
import json
import torch
import numpy as np
import random
import pandas as pd
import argparse
import pickle

from torch.utils.data import dataset 
from torch.utils.data import DataLoader

# from movie import MOVIE, MOVIE_TEST
# from movie_sample import MOVIE, MOVIE_TEST
# from movie_iter import MOVIE_LOADER, MOVIE_TEST

from ratebeer import RATEBEER

class DATA():
    def __init__(self):
        print("data")
    
    def f_load_ratebeer(self, args):
        self.m_data_name = args.data_name

        sent_content_file = args.data_dir+"train/id2sentence.json"
        sent_embed_file = args.data_dir+"train/sid2sentembed.json"
        feature_embed_file = args.data_dir+"train/featureid2embedding.json"
        
        useritem_candidate_label_sen_file = args.data_dir+"train/useritem2sentids.json"
        
        user_feature_file = args.data_dir+"train/user2feature.json"
        item_feature_file = args.data_dir+"train/item2feature.json"

        sent_feature_file = args.data_dir+"train/sentence2feature.json"
        
        train_data = RATEBEER()
        vocab_obj = train_data.load_train_data(sent_content_file, sent_embed_file, feature_embed_file, useritem_candidate_label_sen_file, user_feature_file, item_feature_file, sent_feature_file)

        sent_content_file = args.data_dir+"valid/id2sentence_test.json"
        useritem_candidate_label_sen_file = args.data_dir+"valid/useritem2sentids_test.json"
        
        valid_data = RATEBEER()
        valid_data.load_eval_data(vocab_obj, train_data.m_uid2fid2tfidf_dict, train_data.m_iid2fid2tfidf_dict, train_data.m_sid2fid2tfidf_dict, sent_content_file, useritem_candidate_label_sen_file)

        batch_size = args.batch_size

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=graph_collate_fn)

        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=graph_collate_fn)

        return train_loader, valid_loader, vocab_obj


import dgl

def graph_collate_fn(samples):
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    batched_index = [index[idx] for idx in sorted_index]

    return batched_graph, batched_index    
