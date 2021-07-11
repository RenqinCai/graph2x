import os
import io
import json
import torch
import numpy as np
import random
import pandas as pd
import argparse
import pickle
from torchtext import data

# from torch.utils.data import dataset
# from torch.utils.data import DataLoader
# from torch_geometric.data import Dataset
# from torch_geometric.data import DataLoader

# from ratebeer import RATEBEER
# from ratebeer_process import Vocab
from rnn_process import Vocab


# class DATA():
#     def __init__(self, device):
#         print("data")
#         self.device = device

#     def f_load_graph_ratebeer(self, args):
#         self.m_data_name = args.data_name

#     def f_load_rnn_ratebeer(self, args):
#         self.m_data_name = args.data_name
#         # get the data directory path of the train/val/test json file
#         data_dir = args.data_dir

#         # Extract item and user data
#         # Since we don't need tokenization, sequential is set to False
#         item = data.Field(sequential=False)
#         user = data.Field(sequential=False)

#         # Extract review text data
#         # text = data.Field(
#         #     sequential=True,
#         #     tokenize='spacy',
#         #     init_token="<sos>",
#         #     eos_token="<eos>",
#         #     fix_length=MAX_LENGTH + 2,
#         #     lower=True)

#         text = data.Field(
#             sequential=True,
#             init_token="<sos>",
#             eos_token="<eos>",
#             lower=True)

#         # Extract rating data
#         # Since the rating is then fed into the embedding layer, it should also be dtype of torch.long(default)
#         rating = data.Field(
#             sequential=False,
#             use_vocab=False)

#         if args.verbose:
#             print('Loading datasets...')

#         train, val, test = data.TabularDataset.splits(
#             path=data_dir,
#             train='train_combined.json',
#             test='test_combined.json',
#             validation='valid_combined.json',
#             format='json',
#             fields={
#                 'item': ('item', item),
#                 'user': ('user', user),
#                 'review': ('text', text),
#                 'rating': ('rating', rating)
#             }
#         )

#         if args.verbose:
#             print('datasets loaded')
#         item.build_vocab(train)
#         if args.verbose:
#             print('item vocab built')
#         user.build_vocab(train)
#         if args.verbose:
#             print('user vocab built')
#         text.build_vocab(train.text, min_freq=args.min_freq, max_size=args.max_vocab)
#         if args.verbose:
#             print('text vocab built')

#         train_iter, val_iter, test_iter = data.Iterator.splits(
#             datasets=(train, val, test),
#             batch_sizes=(args.batch_size, args.batch_size_eval, args.batch_size_eval),
#             repeat=False,
#             shuffle=True,
#             sort=False,
#             device=self.device
#         )

#         print('Number of batches in 1 training epoch: {}'.format(len(train_iter)))
#         print('Number of batches in 1 validation epoch: {}'.format(len(val_iter)))
#         print('Number of batches in 1 testing epoch: {}'.format(len(test_iter)))

#         print('============== Dataset Loaded ==============')

#         # construct the vocab object
#         vocab_obj = Vocab(
#             user_vocab=user.vocab,
#             item_vocab=item.vocab,
#             text_vocab=text.vocab,
#             text_field=text
#         )

#         return train_iter, val_iter, test_iter, vocab_obj


# DATA without rating
class DATA():
    def __init__(self, device):
        print("data")
        self.device = device

    def f_load_graph_ratebeer(self, args):
        self.m_data_name = args.data_name

    def f_load_rnn_ratebeer(self, args):
        self.m_data_name = args.data_name
        # get the data directory path of the train/val/test json file
        data_dir = args.data_dir

        # Extract item and user data
        # Since we don't need tokenization, sequential is set to False
        item = data.Field(sequential=False)
        user = data.Field(sequential=False)

        # Extract review text data
        # text = data.Field(
        #     sequential=True,
        #     tokenize='spacy',
        #     init_token="<sos>",
        #     eos_token="<eos>",
        #     fix_length=MAX_LENGTH + 2,
        #     lower=True)

        text = data.Field(
            sequential=True,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True)

        # # Extract rating data
        # # Since the rating is then fed into the embedding layer, it should also be dtype of torch.long(default)
        # rating = data.Field(
        #     sequential=False,
        #     use_vocab=False)

        if args.verbose:
            print('Loading datasets...')

        train, val, test = data.TabularDataset.splits(
            path=data_dir,
            train='train.json',
            test='test.json',
            validation='valid.json',
            format='json',
            fields={
                'item': ('item', item),
                'user': ('user', user),
                'review': ('text', text)
            }
        )

        if args.verbose:
            print('datasets loaded')
        item.build_vocab(train)
        if args.verbose:
            print('item vocab built')
        user.build_vocab(train)
        if args.verbose:
            print('user vocab built')
        # text.build_vocab(train.text, min_freq=args.min_freq, max_size=args.max_vocab)
        text.build_vocab(train.text, min_freq=args.min_freq)
        if args.verbose:
            print('text vocab built')

        train_iter, val_iter, test_iter = data.Iterator.splits(
            datasets=(train, val, test),
            batch_sizes=(args.batch_size, args.batch_size_eval, args.batch_size_eval),
            repeat=False,
            shuffle=True,
            sort=False,
            device=self.device
        )

        print('Number of batches in 1 training epoch: {}'.format(len(train_iter)))
        print('Number of batches in 1 validation epoch: {}'.format(len(val_iter)))
        print('Number of batches in 1 testing epoch: {}'.format(len(test_iter)))

        print("Number of users: {}".format(user.vocab))
        print("Number of items: {}".format(item.vocab))
        print("Number of words: {}".format(text.vocab))

        print('============== Dataset Loaded ==============')

        # construct the vocab object
        vocab_obj = Vocab(
            user_vocab=user.vocab,
            item_vocab=item.vocab,
            text_vocab=text.vocab,
            text_field=text
        )

        return train_iter, val_iter, test_iter, vocab_obj
