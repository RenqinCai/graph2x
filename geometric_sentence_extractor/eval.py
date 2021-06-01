import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# %matplotlib inline
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision
from rouge import Rouge
import dgl
import pickle

dataset_name = 'small_500'


class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        # self.m_item2iid = vocab_obj.m_item2iid
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid

        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}

        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_device = device
        self.m_model_path = args.model_path

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        testset_sent2id_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2id.json'.format(dataset_name)
        testset_sentid2feature_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2feature.json'.format(dataset_name)
        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        with open(testset_sent2id_file, 'r') as f:
            print("Load file: {}".format(testset_sent2id_file))
            self.d_testsetsent2id = json.load(f)
        with open(testset_sentid2feature_file, 'r') as f:
            print("Load file: {}".format(testset_sentid2feature_file))
            self.d_testsetsentid2feature = json.load(f)

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
        # self.f_cluster_embedding()
        self.f_eval_new(train_data, eval_data)

    def f_cluster_embedding(self):

        # self.m_iid2item = {self.m_item2iid[k]:k for k in self.m_item2iid}

        # embeds = self.m_network.m_item_embed.weight.data.cpu().numpy()
        # item_num = len(embeds)
        # labels = [self.m_iid2item[i] for i in range(item_num)]

        # tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        # new_values = tsne_model.fit_transform(embeds)

        # x = []
        # y = []
        # for value in new_values:
        #     x.append(value[0])
        #     y.append(value[1])
            
        # plt.figure(figsize=(16, 16)) 
        # for i in range(len(x)):
        #     plt.scatter(x[i],y[i])
        #     plt.annotate(labels[i],
        #                 xy=(x[i], y[i]),
        #                 xytext=(5, 2),
        #                 textcoords='offset points',
        #                 ha='right',
        #                 va='bottom')
        # plt.savefig("item_embed_tsne.png")

        # m_item_embed is a nn.Embedding layer which maps m_item_num to item_embed_size
        embeds_item = self.m_network.m_item_embed.weight.data.cpu().numpy()
        embeds_feature = self.m_network.m_feature_embed.weight.data.cpu().numpy()

        print("item embedding shape: {}".format(embeds_item.shape))
        print("feature embedding shape: {}".format(embeds_feature.shape))
        item_num = len(embeds_item)
        feature_num = len(embeds_feature)       # for small dataset, this should be 800
        # find the true item that correponding to the iid
        labels_item = [self.m_iid2item[i] for i in range(item_num)]
        labels_feature = [self.m_fid2feature[i] for i in range(feature_num)]    # this is the featureid
        # TODO: not using feature id, but using true feature
        labels_feature_text = [self.d_id2feature[labels_feature[i]] for i in range(feature_num)]

        # dump the label (item/feature) into file
        with open('../embeddings/item_labels_{}.pkl'.format(dataset_name), 'wb') as f:
            pickle.dump(labels_item, f)
        with open('../embeddings/feature_labels_{}.pkl'.format(dataset_name), 'wb') as f:
            pickle.dump(labels_feature_text, f)
        # save item/feature embeddings into file
        with open('../embeddings/item_embs_{}.npy'.format(dataset_name), 'wb') as f:
            np.save(f, embeds_item)
        print("Item embeddings saved!")
        with open('../embeddings/feature_embs_{}.npy'.format(dataset_name), 'wb') as f:
            np.save(f, embeds_feature)
        print("Feature embeddings saved!")

        for i in range(item_num):
            if np.isnan(embeds_item[i]).any():
                print("item {} has NaN embedding!".format(i))

        for i in range(feature_num):
            if np.isnan(embeds_feature[i]).any():
                print("feature {} has NaN embedding!".format(i))

        print("Skip TSNE ... ")
        # # draw the tsne clustering figure of item/feature embeddings
        # print("In tsne ... ")

        # tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        # print("Init tsne model ...")

        # new_values = tsne_model.fit_transform(embeds_item)
        # print("Finish fiting tsne!")

        # x = []
        # y = []
        # for value in new_values:
        #     x.append(value[0])
        #     y.append(value[1])

        # plt.figure(figsize=(16, 16))
        # for i in range(len(x)):
        #     plt.scatter(x[i], y[i])
        #     plt.annotate(
        #         labels_item[i],
        #         xy=(x[i], y[i]),
        #         xytext=(5, 2),
        #         textcoords='offset points',
        #         ha='right',
        #         va='bottom')
        # print("figure saved!")
        # plt.savefig("item_embed_tsne_new.png")
        print("Finish clustering")

    def f_eval_new(self, train_data, eval_data):

        recall_list = []
        precision_list = []
        F1_list = []

        rouge_1_f_list = []
        rouge_1_p_list = []
        rouge_1_r_list = []

        rouge_2_f_list = []
        rouge_2_p_list = []
        rouge_2_r_list = []

        rouge_l_f_list = []
        rouge_l_p_list = []
        rouge_l_r_list = []

        bleu_list = []
        bleu_1_list = []
        bleu_2_list = []
        bleu_3_list = []
        bleu_4_list = []

        rouge = Rouge()

        feature_recall_list = []
        feature_precision_list = []

        feature_num_per_sentence_ref = []
        feature_num_per_sentence_pred = []
        feature_num_per_review_ref = []
        feature_num_per_review_pred = []

        print('--'*10)

        debug_index = 0
        topk = 3
        topk_candidate = 20

        # already got feature2fid mapping, need the reverse
        self.m_fid2feature = {value: key for key, value in self.m_feature2fid.items()}
        # print(self.m_feture2fid)

        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))
            for i, (G, index) in enumerate(eval_data):
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                if i % 20 == 0:
                    print("... eval ... index: {}".format(i))
                    # print(G)

                # debug_index += 1
                # if debug_index > 1:
                #     break

                G = G.to(self.m_device)

                logits = self.m_network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                labels = G.ndata["label"][snode_id]

                labels = labels.float()
                node_loss = self.m_criterion(logits, labels)
                # print("node_loss", node_loss.size())

                G.nodes[snode_id].data["loss"] = node_loss
                loss = dgl.sum_nodes(G, "loss")
                loss = loss.mean()

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)

                # print("number of graphs in this batch: {}\n".format(len(glist)))

                for j in range(len(glist)):
                    hyps_j = []
                    refs_j = []

                    idx = index[j]
                    example_j = eval_data.dataset.get_example(idx)
                    label_sid_list_j = example_j["label_sid"]

                    g_j = glist[j]
# <<<<<<< HEAD
                    # snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==1)

                    # """
                    # get feature attn weight
                    # """

                    # for k in snode_id_j:
                    #     predecessors = list(g_j.predecessors(k))
                    #     edges_id = g_j.edge_ids(predecessors, k)
                        

                    # NOTE: sentence node: unit=0, dtype=1
                    snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
# >>>>>>> peng
                    N = len(snode_id_j)
                    # print("number of candidate sentence at this graph (i.e. for this user-item pair): {}".format(N))
                    # This should be the logits of this sentence nodes (snode_id_j)
                    p_sent_j = g_j.ndata["p"][snode_id_j]
                    # print('p_sent_j shape: {}'.format(p_sent_j.shape))
                    p_sent_j = p_sent_j.view(-1)
                    p_sent_j = F.sigmoid(p_sent_j)

                    # Get the top-k sentence as predicted sentences
                    topk_j, pred_idx_j = torch.topk(p_sent_j, min(topk, N))
                    pred_snode_id_j = snode_id_j[pred_idx_j]

                    # Get the candidate top-k sentences to perform a sanity check
                    topk_cand_j, pred_cand_idx_j = torch.topk(p_sent_j, min(topk_candidate, N))
                    pred_cand_snode_id_j = snode_id_j[pred_cand_idx_j]

                    # Get the candidate bottom-k sentences to preform a sanity check
                    # NOTE: bottom_cand_j are still negative numbers
                    bottomk_cand_j, pred_bottom_idx_j = torch.topk(-1*p_sent_j, min(topk_candidate, N))
                    pred_bottom_snode_id_j = snode_id_j[pred_bottom_idx_j]
                    proba_bottomk_cand_j = -1*bottomk_cand_j

                    # print("topk_j", topk_j)
                    # Need to get the label_snode_id
                    labels_j = g_j.ndata["label"][snode_id_j]       # this should be an all-zero tensor

                    # print("labels_j: ", labels_j.view(-1))
                    # print("labels_j shape: {}".format(labels_j.shape))

                    # Extract prediction (top3)
                    # print("Number of nodes selected as prediction: {}".format(len(pred_snode_id_j)))
                    pred_feature_j = []
                    for cur_pred_snode_id_j in pred_snode_id_j:
                        # print("sent node {0}'s successor: {1}".format(cur_pred_snode_id_j, g_j.successors(cur_pred_snode_id_j)))
                        pred_fid_list_j = list()
                        pred_feature_list_j = list()
                        for pred_fnode_id_j in g_j.successors(cur_pred_snode_id_j):
                            # get fid using the "raw_id"
                            pred_fid_id_j = g_j.nodes[pred_fnode_id_j].data["raw_id"]
                            pred_fid_list_j.append(pred_fid_id_j)
                            # get featureid from the fid2feature mapping
                            pred_featureid_id_j = self.m_fid2feature[pred_fid_id_j.item()]
                            # get the feature content
                            pred_feature_id_j = self.d_id2feature[pred_featureid_id_j]
                            pred_feature_list_j.append(pred_feature_id_j)
                        # print("Feature: {}".format(" & ".join(pred_feature_list_j)))
                        pred_feature_j.extend(pred_feature_list_j)
                        # add number of features into per sentence
                        feature_num_per_sentence_pred.append(len(pred_feature_list_j))
                    # add number of features per review
                    feature_num_per_review_pred.append(len(pred_feature_j))

                    # pred_idx_j = pred_idx_j.cpu().numpy()
                    pred_sid_list_j = g_j.nodes[pred_snode_id_j].data["raw_id"]
                    pred_logits_list_j = g_j.nodes[pred_snode_id_j].data["p"]

                    # Extract Top predicted candidate sentences
                    # print("Number of nodes ranking from top as prediction candidates: {}".format(len(pred_cand_idx_j)))
                    candidate_sent2prob = dict()
                    candidate_sent2features = dict()
                    # from snode_id to sid
                    pred_cand_sid_list_j = g_j.nodes[pred_cand_snode_id_j].data["raw_id"]
                    # the proba of these sentences can be found from topk_cand_j
                    assert len(topk_cand_j) == len(pred_cand_sid_list_j)
                    assert len(pred_cand_snode_id_j) == len(pred_cand_sid_list_j)
                    cand_sent_idx = 0
                    for cur_pred_snode_cand_id_j in pred_cand_snode_id_j:
                        # get feature of this sentence
                        pred_cand_fid_list_j = list()
                        pred_cand_feature_list_j = list()
                        for pred_fnode_id_j in g_j.successors(cur_pred_snode_cand_id_j):
                            pred_fid_id_j = g_j.nodes[pred_fnode_id_j].data["raw_id"]
                            pred_cand_fid_list_j.append(pred_fid_id_j)
                            # get featureid
                            pred_featureid_id_j = self.m_fid2feature[pred_fid_id_j.item()]
                            # get feature content
                            pred_feature_id_j = self.d_id2feature[pred_featureid_id_j]
                            # add feature to the list
                            pred_cand_feature_list_j.append(pred_feature_id_j)
                        """ fill in the dict
                        1. get the content of this sentence
                        2. get the probability of this sentence
                        """
                        # get the content of this sentence
                        this_sid = pred_cand_sid_list_j[cand_sent_idx]
                        this_sent_content = self.m_sid2swords[this_sid.item()]
                        # get the probability of this sentence
                        this_sent_proba = topk_cand_j[cand_sent_idx]
                        assert this_sent_content not in candidate_sent2prob
                        candidate_sent2prob[this_sent_content] = this_sent_proba
                        candidate_sent2features[this_sent_content] = pred_cand_feature_list_j
                        # counter update
                        cand_sent_idx += 1

                    # Extract Bottom predicted candidate sentences
                    bottom_sent2prob = dict()
                    bottom_sent2features = dict()
                    # from snode_id to sid
                    pred_bottom_sid_list_j = g_j.nodes[pred_bottom_snode_id_j].data["raw_id"]
                    # the proba of these sentences can be found from proba_bottomk_cand_j (in ascending order)
                    assert len(proba_bottomk_cand_j) == len(pred_bottom_sid_list_j)
                    assert len(pred_bottom_sid_list_j) == len(pred_bottom_snode_id_j)
                    bottom_sent_idx = 0
                    for cur_pred_snode_cand_id_j in pred_bottom_snode_id_j:
                        # get the feature of this sentence
                        pred_bottom_fid_list_j = list()
                        pred_bottom_feature_list_j = list()
                        for pred_fnode_id_j in g_j.successors(cur_pred_snode_cand_id_j):
                            pred_fid_id_j = g_j.nodes[pred_fnode_id_j].data['raw_id']
                            pred_bottom_fid_list_j.append(pred_fid_id_j)
                            # get featureid
                            pred_featureid_id_j = self.m_fid2feature[pred_fid_id_j.item()]
                            # get feature content
                            pred_feature_id_j = self.d_id2feature[pred_featureid_id_j]
                            # add feature to the list
                            pred_bottom_feature_list_j.append(pred_feature_id_j)
                        # 1. get the content of this sentence
                        # 2. get the probability of this sentence
                        # get the content of this sentence
                        this_sid = pred_bottom_sid_list_j[bottom_sent_idx]
                        this_sent_content = self.m_sid2swords[this_sid.item()]
                        # get the probability of this sentence
                        this_sent_proba = proba_bottomk_cand_j[bottom_sent_idx]
                        assert this_sent_content not in bottom_sent2prob
                        bottom_sent2prob[this_sent_content] = this_sent_proba
                        bottom_sent2features[this_sent_content] = pred_bottom_feature_list_j
                        # counter update
                        bottom_sent_idx += 1

                    # print("pred_logits_list_j", pred_logits_list_j)
                    # exit()
                    unode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
                    uid_j = g_j.nodes[unode_id_j].data["raw_id"]
                    true_user_id_j = self.m_uid2user[uid_j.item()]

                    inode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"] == 3)
                    iid_j = g_j.nodes[inode_id_j].data["raw_id"]
                    true_item_id_j = self.m_iid2item[iid_j.item()]

                    recall_j, precision_j = get_example_recall_precision(
                        pred_sid_list_j.cpu(), label_sid_list_j, min(topk, N))

                    # recall_list.append(recall_j)
                    # precision_list.append(precision_j)

                    # for sid_k in label_sid_list_j:
                    #     hyps_j.append(self.m_sid2swords[sid_k])

                    # for sid_k in pred_sid_list_j:
                    #     refs_j.append(self.m_sid2swords[sid_k.item()])

                    # Get the ground-truth feature
                    refs_feature_j = []
                    for sid_k in label_sid_list_j:
                        reference_sent = self.m_sid2swords[sid_k]
                        refs_j.append(reference_sent)
                        # get the feature of this sentence
                        ref_sent_id = self.d_testsetsent2id[reference_sent]
                        ref_sent_featureids = self.d_testsetsentid2feature[ref_sent_id].keys()
                        ref_sent_features = []
                        for ref_sent_fid in ref_sent_featureids:
                            ref_sent_f = self.d_id2feature[ref_sent_fid]
                            ref_sent_features.append(ref_sent_f)
                        refs_feature_j.extend(ref_sent_features)
                        # print("Ref feature: {}".format(" & ".join(ref_sent_features)))
                        # add number of features per sentence
                        feature_num_per_sentence_ref.append(len(ref_sent_features))
                    # add number of features per review
                    feature_num_per_review_ref.append(len(refs_feature_j))

                    for sid_k in pred_sid_list_j:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    # compute recall score of features
                    recall_f_j, precision_f_j = get_feature_recall_precision(pred_feature_j, refs_feature_j)

                    hyps_j = " ".join(hyps_j)
                    refs_j = " ".join(refs_j)

                    # if uid_j.item() == 0:
                    #     continue

                    with open('../result/eval_logging_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(true_user_id_j))
                        f.write("item id: {}\n".format(true_item_id_j))
                        f.write("hyps_j: {}\n".format(hyps_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        f.write("pred features: {}\n".format(" & ".join(pred_feature_j)))
                        f.write("refs features: {}\n".format(" & ".join(refs_feature_j)))
                        f.write("feature recall: {}\n".format(recall_f_j))
                        f.write("feature precision: {}\n".format(precision_f_j))
                        f.write("========================================\n")

                    with open('../result/eval_logging_top_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(true_user_id_j))
                        f.write("item id: {}\n".format(true_item_id_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        for key, value in candidate_sent2prob.items():
                            # key is the sentence content
                            # value is the probability of this sentence
                            f.write("candidate sentence: {}\n".format(key))
                            f.write("proba: {}\n".format(value))
                            # also retrieve the feature of this sentence
                            cur_fea_list = candidate_sent2features[key]
                            f.write("feature: {}\n".format(" & ".join(cur_fea_list)))
                            f.write("----:----:----:----:----:----:----:----:\n")
                        f.write("========================================\n")

                    with open('../result/eval_logging_bottom_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(true_user_id_j))
                        f.write("item id: {}\n".format(true_item_id_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        for key, value in bottom_sent2prob.items():
                            # key is the sentence content
                            # value is the probability of this sentence
                            f.write("candidate sentence: {}\n".format(key))
                            f.write("proba: {}\n".format(value))
                            # also retrieve the feature of this sentence
                            cur_fea_list = bottom_sent2features[key]
                            f.write("feature: {}\n".format(" & ".join(cur_fea_list)))
                            f.write("----:----:----:----:----:----:----:----:\n")
                        f.write("========================================\n")

                    scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)

                    rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                    rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                    rouge_1_p_list.append(scores_j["rouge-1"]["p"])

                    rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                    rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                    rouge_2_p_list.append(scores_j["rouge-2"]["p"])

                    rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                    rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                    rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                    bleu_scores_j = compute_bleu([refs_j], [hyps_j])
                    bleu_list.append(bleu_scores_j)

                    feature_recall_list.append(recall_f_j)
                    feature_precision_list.append(precision_f_j)

                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_bleu([refs_j], [hyps_j])

                    # bleu_1_scores_j = compute_bleu_order([refs_j], [hyps_j], order=1)
                    bleu_1_list.append(bleu_1_scores_j)

                    # bleu_2_scores_j = compute_bleu_order([refs_j], [hyps_j], order=2)
                    bleu_2_list.append(bleu_2_scores_j)

                    # bleu_3_scores_j = compute_bleu_order([refs_j], [hyps_j], order=3)
                    bleu_3_list.append(bleu_3_scores_j)

                    # bleu_4_scores_j = compute_bleu_order([refs_j], [hyps_j], order=4)
                    bleu_4_list.append(bleu_4_scores_j)

                # exit()
                # break

        self.m_mean_eval_rouge_1_f = np.mean(rouge_1_f_list)
        self.m_mean_eval_rouge_1_r = np.mean(rouge_1_r_list)
        self.m_mean_eval_rouge_1_p = np.mean(rouge_1_p_list)

        self.m_mean_eval_rouge_2_f = np.mean(rouge_2_f_list)
        self.m_mean_eval_rouge_2_r = np.mean(rouge_2_r_list)
        self.m_mean_eval_rouge_2_p = np.mean(rouge_2_p_list)

        self.m_mean_eval_rouge_l_f = np.mean(rouge_l_f_list)
        self.m_mean_eval_rouge_l_r = np.mean(rouge_l_r_list)
        self.m_mean_eval_rouge_l_p = np.mean(rouge_l_p_list)

        self.m_mean_eval_bleu = np.mean(bleu_list)
        self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
        self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
        self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
        self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        self.m_recall_feature = np.mean(feature_recall_list)
        self.m_precision_feature = np.mean(feature_precision_list)

        self.m_mean_f_num_per_sent_pred = np.mean(feature_num_per_sentence_pred)
        self.m_mean_f_num_per_sent_ref = np.mean(feature_num_per_sentence_ref)
        self.m_mean_f_num_per_review_pred = np.mean(feature_num_per_review_pred)
        self.m_mean_f_num_per_review_ref = np.mean(feature_num_per_review_ref)

        # print("NLL_loss:%.4f"%(self.m_mean_eval_loss))
        print("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f" % (
            self.m_mean_eval_rouge_1_f,
            self.m_mean_eval_rouge_1_p,
            self.m_mean_eval_rouge_1_r,
            self.m_mean_eval_rouge_2_f,
            self.m_mean_eval_rouge_2_p,
            self.m_mean_eval_rouge_2_r,
            self.m_mean_eval_rouge_l_f,
            self.m_mean_eval_rouge_l_p,
            self.m_mean_eval_rouge_l_r))
        print("bleu:%.4f" % (self.m_mean_eval_bleu))
        print("bleu-1:%.4f" % (self.m_mean_eval_bleu_1))
        print("bleu-2:%.4f" % (self.m_mean_eval_bleu_2))
        print("bleu-3:%.4f" % (self.m_mean_eval_bleu_3))
        print("bleu-4:%.4f" % (self.m_mean_eval_bleu_4))
        print("feature recall: {}".format(self.m_recall_feature))
        print("feature precision: {}".format(self.m_precision_feature))
        print("feature num per sentence [pred]: {}".format(self.m_mean_f_num_per_sent_pred))
        print("feature num per sentence [ref]: {}".format(self.m_mean_f_num_per_sent_ref))
        print("feature num per review [pred]: {}".format(self.m_mean_f_num_per_review_pred))
        print("feature num per review [ref]: {}".format(self.m_mean_f_num_per_review_ref))
        print("total number of sentence [pred]: {}".format(len(feature_num_per_sentence_pred)))
        print("total number of sentence [ref]: {}".format(len(feature_num_per_sentence_ref)))
        print("total number of review [pred]: {}".format(len(feature_num_per_review_pred)))
        print("total number of review [ref]: {}".format(len(feature_num_per_review_ref)))

        with open('../result/eval_metrics_{}.txt'.format(dataset_name), 'w') as f:
            print("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f \n"%(
                self.m_mean_eval_rouge_1_f,
                self.m_mean_eval_rouge_1_p,
                self.m_mean_eval_rouge_1_r,
                self.m_mean_eval_rouge_2_f,
                self.m_mean_eval_rouge_2_p,
                self.m_mean_eval_rouge_2_r,
                self.m_mean_eval_rouge_l_f,
                self.m_mean_eval_rouge_l_p,
                self.m_mean_eval_rouge_l_r), file=f)
            print("bleu:%.4f\n" % (self.m_mean_eval_bleu), file=f)
            print("bleu-1:%.4f\n" % (self.m_mean_eval_bleu_1), file=f)
            print("bleu-2:%.4f\n" % (self.m_mean_eval_bleu_2), file=f)
            print("bleu-3:%.4f\n" % (self.m_mean_eval_bleu_3), file=f)
            print("bleu-4:%.4f\n" % (self.m_mean_eval_bleu_4), file=f)
            print("feature recall: {}\n".format(self.m_recall_feature), file=f)
            print("feature precision: {}\n".format(self.m_precision_feature), file=f)
