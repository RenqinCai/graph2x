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
from text_process import Vocab


class DATA():
    def __init__(self, device):
        print("data")
        self.device = device

    def f_load_featsent(self, args):
        self.m_data_name = args.data_name
        data_dir = args.data_dir

        item = data.Field(sequential=False, use_vocab=False)
        user = data.Field(sequential=False, use_vocab=False)
        sentence = data.Field(sequential=False, use_vocab=False)
        feature = data.Field(sequential=True, batch_first=True, include_lengths=True)
        label = data.Field(sequential=False, unk_token=None, is_target=True)

        if args.verbose:
            print('Loading datasets...')

        train, val, test = data.TabularDataset.splits(
            path=data_dir,
            train='train/useritemsent2feat_multilines_sample.json',
            test='test/useritemsent2feat_multilines.json',
            validation='valid/useritemsent2feat_multilines_sample.json',
            format='json',
            fields={
                'item': ('item', item),
                'user': ('user', user),
                'sent_id': ('sentence', sentence),
                'feature': ('feature', feature),
                'label': ('label', label)
            }
        )

        if args.verbose:
            print('datasets loaded')
        # item.build_vocab(train.item)
        # if args.verbose:
        #     print('item vocab built')
        # user.build_vocab(train.user)
        # if args.verbose:
        #     print('user vocab built')
        feature.build_vocab(train.feature)
        if args.verbose:
            print('feature vocab built')
        label.build_vocab(train.label)
        if args.verbose:
            print('label vocab built')

        # During training, shuffle should be True.
        # During testing, shuffle should be False.
        if args.train:
            train_iter, val_iter, test_iter = data.Iterator.splits(
                datasets=(train, val, test),
                batch_sizes=(args.batch_size, args.batch_size_eval, args.batch_size_eval),
                repeat=False,
                shuffle=True,
                sort=False,
                device=self.device
            )
        else:
            train_iter, val_iter, test_iter = data.Iterator.splits(
                datasets=(train, val, test),
                batch_sizes=(args.batch_size, args.batch_size_eval, args.batch_size_eval),
                repeat=False,
                shuffle=False,
                sort=False,
                device=self.device
            )

        print('Number of batches in 1 training epoch: {}'.format(len(train_iter)))
        print('Number of batches in 1 validation epoch: {}'.format(len(val_iter)))
        print('Number of batches in 1 testing epoch: {}'.format(len(test_iter)))

        print('============== Dataset Loaded ==============')

        # construct the vocab object
        vocab_obj = Vocab(
            text_vocab=feature.vocab,
            text_field=feature
        )

        return train_iter, val_iter, test_iter, vocab_obj

    # def f_load_featsent_1(self, data_name, data_dir, batch_size, batch_size_eval, verbose=True):
    #     """ used for debug
    #     """
    #     self.m_data_name = data_name

    #     item = data.Field(sequential=False, use_vocab=False)
    #     user = data.Field(sequential=False, use_vocab=False)
    #     sentence = data.Field(sequential=False, use_vocab=False)
    #     feature = data.Field(sequential=True, batch_first=True, include_lengths=True)
    #     label = data.Field(sequential=False, unk_token=None, is_target=True)

    #     if verbose:
    #         print('Loading datasets...')

    #     train, val, test = data.TabularDataset.splits(
    #         path=data_dir,
    #         train='train/useritemsent2feat_multilines_sample.json',
    #         test='test/useritemsent2feat_multilines.json',
    #         validation='valid/useritemsent2feat_multilines_sample.json',
    #         format='json',
    #         fields={
    #             'item': ('item', item),
    #             'user': ('user', user),
    #             'feature': ('feature', feature),
    #             'label': ('label', label),
    #             'sent_id': ('sentence', sentence)
    #         }
    #     )

    #     if verbose:
    #         print('user vocab built')
    #     feature.build_vocab(train.feature)
    #     if verbose:
    #         print('feature vocab built')
    #     label.build_vocab(train.label)
    #     if verbose:
    #         print('label vocab built')

    #     train_iter, val_iter, test_iter = data.Iterator.splits(
    #         datasets=(train, val, test),
    #         batch_sizes=(batch_size, batch_size_eval, batch_size_eval),
    #         repeat=False,
    #         shuffle=False,
    #         sort=False,
    #         device=self.device
    #     )

    #     print('Number of batches in 1 training epoch: {}'.format(len(train_iter)))
    #     print('Number of batches in 1 validation epoch: {}'.format(len(val_iter)))
    #     print('Number of batches in 1 testing epoch: {}'.format(len(test_iter)))

    #     print('============== Dataset Loaded ==============')

    #     # construct the vocab object
    #     vocab_obj = Vocab(
    #         text_vocab=feature.vocab,
    #         text_field=feature
    #     )

    #     return train_iter, val_iter, test_iter, vocab_obj
