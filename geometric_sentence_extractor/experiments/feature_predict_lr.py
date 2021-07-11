from numpy.lib.npyio import load
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from sklearn import metrics
import json

def readJson(fname):
    data = []
    line_num = 0
    with open(fname, encoding="utf-8") as f:
        for line in f:
            # print("line", line)
            line_num += 1
            try:
                data.append(json.loads(line))
            except:
                print("error", line_num)
    return data

def load_train_feature_label(feature_label_file):
    x = []
    y = []
    user_item_pair = []

    with open(feature_label_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip().split("\t")
            x.append(line[:-2])
            y.append(line[-2])

    x = np.array(x)
    y = np.array(y)
    print("x size", x.shape)
    print("y size", y.shape)

def load_test_feature_label(feature_label_file):
    x = []
    y = []
    user_item_pair = []

    data = readJson(feature_label_file)    
    # with open(feature_label_file, "r") as f:
    #     for raw_line in f:
    #         # line = raw_line.strip().split("\t")
    data_num = len(data)

    x_ids = []
    x = []
    y = []
    topk_y = []
    for i in range(data_num):
        data_i = data[i]
        pair_index = data.keys()
        pair_value = data.values()

        feature_id_embed = pair_value["feature"]
        feature_id = feature_id_embed[:, 0]
        feature_embed = feature_id_embed[:, 1: ]

        topk_num = pair_value["topk"]
        topk_y.append(topk_num)

        x_ids.append(feature_id)
        x.append(feature_embed)

        gt_feature_id = pair_value["gt"]
        y.append(gt_feature_id)

    # x = np.array(x)
    # # y = np.array(y)
    # print("x size", x.shape)

    return x_ids, x, topk_y, y


def train_model(x, y):

    clf = LogisticRegression(random_state=0).fit(x, y)
    
    return clf

def iterate_eval_model(model, x_ids, x, y, test_topk_num):
    pair_num = len(x)
    precision_list = []
    recall_list = []
    f1_list = []
    auc = []

    for i in range(pair_num):
        x_i = x[i]
        y_i = y[i]
        xid_i = x_ids[i]
        topk_i = test_topk_num[i]

        preds_i = model.predict_proba(x_i)
        idx_i = np.argpartition(preds_i, -topk_i)[-topk_i:]  # Indices not sorted
        topk_preds_idx_i = idx_i[np.argsort(preds_i[idx_i])][::-1] 

        topk_preds_i = xid_i[topk_preds_idx_i]
        
        TP = set(topk_preds_i).intersection(set(y_i))

        precision = TP/topk_i
        recall = TP/len(y_i)

        f1 = 2*precision*recall/(precision+recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    
    return avg_precision, avg_recall, avg_f1

# def eval_model(model, x, y, topk):
#     preds = model.predict_proba(x)
    
#     idx = np.argpartition(preds, -topk)[-topk:]  # Indices not sorted

#     topk_preds = idx[np.argsort(preds[idx])][::-1] 

#     fpr, tpr, thresholds = metrics.roc_curve(y, topk_preds, pos_label=1)
#     auc = metrics.auc(fpr, tpr)

#     T = (topk_preds == y)
#     P = (topk_preds == 1)

#     TP = sum(T*P)

#     precision = TP/topk

#     recall = TP/sum(y)

#     f1 = 2*precision*recall/(precision+recall)
    
#     return precision, recall, f1, auc

train_input_file = ".txt"

train_x, train_y = load_train_feature_label(train_input_file)

lr_model = train_model(train_x, train_y)

test_input_file = ".json"

test_x_ids, test_x, test_topk_num, test_y = load_test_feature_label(test_input_file)

precision, recall, f1 = iterate_eval_model(lr_model, test_x, test_y, test_topk_num)

# precision, recall, f1, auc = eval_model(lr_model, test_x, test_y, topk)
