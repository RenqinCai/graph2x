import os
import io
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
import argparse
import copy
import pickle
import string
import datetime
from collections import Counter 

class BEER(Dataset):
    def __init__(self, args):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size
    
        self.m_max_line = 1e10

        self.m_input_file = args.data_input_file
        self.m_input_abs_file = os.path.join(self.m_data_dir, self.m_input_file)

        data = []
        line_num = 0
        with open(self.m_input_abs_file) as f:
            for line in f:
                line_data = json.loads(line)

                user_i = line_data["user"]
                item_i = line_data["item"]
                candidate_i = line_data["candidate"]
                review_i = line_data["review"]

                if len(candidate_i) == 0:
                    continue
                    
                if len(review_i) == 0:
                    continue

                data.append(line_data)
        
        random.shuffle(data)
        print("before sampling data num", len(data))
        # sample_num = 20000
        # data = data[:sample_num]
        self.m_sample_num = len(data)
        print("after sampling data num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        print("batch num", self.m_batch_num)
            
        self.m_input_batch_list = data


    def __len__(self):
        return len(self.m_input_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx

        input_i = self.m_input_batch_list[i]
    
        return input_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        input_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            # input_i = sample_i[0]
            input_i = sample_i

            # input_i["candidate"] = input_i["candidate"][:10]

            # print(input_i)
            # exit()
            input_iter.append(input_i)

        return input_iter
