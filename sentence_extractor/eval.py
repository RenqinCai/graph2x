import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os

from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import get_recall

class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_device = device
        self.m_model_path = args.model_path

    def f_init_eval(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        print("eval new")
        self.f_eval_new(train_data, eval_data)

    def f_eval_new(self, train_data, eval_data):

        recall_list = []

        print('--'*10)

        topk = 20
        self.m_network.eval()
        with torch.no_grad():

            for user_batch, item_batch, mask_item_batch, itemnum_batch in eval_data:
                
                user_gpu = user_batch.to(self.m_device)

                logits = self.m_network.f_eval_forward(user_gpu)
                
                recall = get_recall(logits.cpu(), item_batch, mask_item_batch, itemnum_batch, k=topk)
                
                recall_list.append(recall)

        mean_recall = np.mean(recall_list)
        print("recall@%d:%.4f"%(topk, mean_recall))
