import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

def get_example_recall_precision(pred, target, k=1):
    recall = 0.0
    precision = 0.0

    pred = list(pred.numpy())
    target = list(target.numpy())

    true_pos = set(target) & set(pred)
    true_pos_num = len(true_pos)

    target_num = len(target)
    recall = true_pos_num*1.0/target_num

    precision = true_pos_num*1.0/k

    return recall, precision

# def get_recall(preds, targets, mask, targetnum, k=1):
#     preds = preds.view(-1, preds.size(1))

#     preds.scatter_(1, mask, float("-inf"))
#     preds[:, 0] = float("-inf")
    
#     top_vals, indices = torch.topk(preds, k, -1)

#     recall_list = []

#     for i, pred_index in enumerate(indices):
#         pred_i = list(pred_index.numpy())
#         target_i = targets[i].numpy()
#         # len_i = sum(target_i != 0)
#         num_i = targetnum[i].item()
#         target_i = list(target_i)[:num_i]
    
#         true_pos = set(target_i) & set(pred_i)
#         true_pos_num = len(true_pos)

#         recall = true_pos_num/num_i
#         recall_list.append(recall)

#     avg_recall = np.mean(recall_list)

#     return avg_recall