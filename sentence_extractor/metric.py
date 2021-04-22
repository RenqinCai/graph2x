import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn
import collections
import math


def get_example_recall_precision(pred, target, k=1):
    recall = 0.0
    precision = 0.0

    pred = list(pred.numpy())
    # target = list(target.numpy())

    true_pos = set(target) & set(pred)
    true_pos_num = len(true_pos)

    target_num = len(target)
    recall = true_pos_num*1.0/target_num

    precision = true_pos_num*1.0/k

    return recall, precision

def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order+1):
        for i in range(0, len(segment)-order+1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1

    return ngram_counts

def compute_bleu(references, hypotheses, max_order=4, smooth=False):
    matches_by_order = [0]*max_order
    possible_matches_by_order = [0]*max_order

    reference_length = 0
    hypothesis_length = 0

    for (reference, hypothesis) in zip(references, hypotheses):
        reference_length += min(len(r) for r in references)
        hypothesis_length += len(hypothesis)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        
        hyp_ngram_counts = _get_ngrams(hypothesis, max_order)
        overlap = hyp_ngram_counts & merged_ref_ngram_counts

        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        
        for order in range(1, max_order+1):
            possible_matches = len(hypothesis)-order+1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0]*max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i]+1.0)/(possible_matches_by_order[i]+1.0))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i])/possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0/max_order)*math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(hypothesis_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1-1.0/ratio)

    bleu = geo_mean*bp

    return bleu
