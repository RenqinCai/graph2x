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
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision, get_sentence_bleu
from rouge import Rouge
import dgl
import pickle

dataset_name = 'medium_500'
label_format = 'soft_label'
use_blocking = True


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

        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        testset_sent2id_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2id.json'.format(dataset_name)
        # testset_sentid2feature_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2feature.json'.format(dataset_name)
        trainset_useritem_pair_file = '../../Dataset/ratebeer/{}/train/useritem_pairs.json'.format(dataset_name)
        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        with open(testset_sent2id_file, 'r') as f:
            print("Load file: {}".format(testset_sent2id_file))
            self.d_testsetsent2id = json.load(f)
        # with open(testset_sentid2feature_file, 'r') as f:
        #     print("Load file: {}".format(testset_sentid2feature_file))
        #     self.d_testsetsentid2feature = json.load(f)
        with open(trainset_useritem_pair_file, 'r') as f:
            print("Load file: {}".format(trainset_useritem_pair_file))
            self.d_trainset_useritempair = json.load(f)

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

        cnt_useritem_batch = 0
        train_test_overlap_cnt = 0
        train_test_differ_cnt = 0
        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))

            for graph_batch in eval_data:

                if cnt_useritem_batch % 10 == 0:
                    print("... eval ... ", cnt_useritem_batch)
                cnt_useritem_batch += 1

                graph_batch = graph_batch.to(self.m_device)

                # logits: batch_size*max_sen_num
                logits, sids, masks, target_sids = self.m_network.eval_forward(graph_batch)
                batch_size = logits.size(0)

                if use_blocking:
                    # use n-gram blocking
                    # get all the sentence content
                    batch_sents_content = []
                    assert len(sids) == logits.size(0)      # this is the batch size
                    for i in range(batch_size):
                        cur_sents_content = []
                        assert len(sids[i]) == len(sids[0])
                        for cur_sid in sids[i]:
                            cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
                        batch_sents_content.append(cur_sents_content)
                    assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
                    # 1. get the top-k predicted sentences which form the hypothesis
                    ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(batch_sents_content, logits, n_win=3, k=3)
                    ngram_block_pred_snids = ngram_block_pred_snids.to(self.m_device)
                    pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
                    topk_logits = ngram_block_pred_proba
                    # 2. get the top-20 predicted sentences' content and proba
                    top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(batch_sents_content, logits, n_win=3, k=topk_candidate)
                    top_cdd_pred_snids = top_cdd_pred_snids.to(self.m_device)
                    top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
                    # 3. get the bottom-20 predicted sentences' content and proba
                    reverse_logits = (1-logits)*masks
                    bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_logits, topk_candidate, dim=1)
                    bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)
                else:
                    # 1. get the top-k predicted sentences which form the hypothesis
                    topk_logits, topk_pred_snids = torch.topk(logits, topk, dim=1)
                    # topk sentence index
                    # pred_sids: shape: (batch_size, topk_sent)
                    pred_sids = sids.gather(dim=1, index=topk_pred_snids)
                    # 2. get the top-20 predicted sentences' content and proba
                    top_cdd_logits, top_cdd_pred_snids = torch.topk(logits, topk_candidate, dim=1)
                    top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
                    # 3. get the bottom-20 predicted sentences' content and proba
                    reverse_logits = (1-logits)*masks
                    bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_logits, topk_candidate, dim=1)
                    bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                for j in range(batch_size):
                    refs_j_list = []
                    hyps_j_list = []

                    for sid_k in target_sids[j]:
                        refs_j_list.append(self.m_sid2swords[sid_k.item()])

                    for sid_k in pred_sids[j]:
                        hyps_j_list.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j_list)
                    refs_j = " ".join(refs_j_list)

                    userid_j = userid[j].item()
                    itemid_j = itemid[j].item()
                    # get the true user/item id
                    true_userid_j = self.m_uid2user[userid_j]
                    true_itemid_j = self.m_iid2item[itemid_j]
                    # check whether this user-item pair appears in the trainset
                    if true_userid_j in self.d_trainset_useritempair:
                        if true_itemid_j in self.d_trainset_useritempair[true_userid_j]:
                            # this user-item pair already appeared in the trainset, ignore this user-item pair.
                            train_test_overlap_cnt += 1
                            continue
                        else:
                            train_test_differ_cnt += 1
                    else:
                        raise Exception("user: {} not in trainset but in testset!".format(true_userid_j))

                    with open('../result/eval_logging_{0}_{1}.txt'.format(dataset_name, label_format), 'a') as f:
                        f.write("user id: {}\n".format(true_userid_j))
                        f.write("item id: {}\n".format(true_itemid_j))
                        f.write("hyps_j: {}\n".format(hyps_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        f.write("probas: {}\n".format(topk_logits[j]))
                        if use_blocking:
                            f.write("rank: {}\n".format(ngram_block_pred_rank[j]))
                        f.write("========================================\n")

                    top_cdd_hyps_j = []
                    top_cdd_probs_j = top_cdd_logits[j]
                    for sid_k in top_cdd_pred_sids[j]:
                        top_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])

                    with open('../result/eval_logging_top_{0}_{1}.txt'.format(dataset_name, label_format), 'a') as f:
                        f.write("user id: {}\n".format(true_userid_j))
                        f.write("item id: {}\n".format(true_itemid_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        for k in range(topk_candidate):
                            # key is the sentence content
                            # value is the probability of this sentence
                            f.write("candidate sentence: {}\n".format(top_cdd_hyps_j[k]))
                            f.write("prob: {}\n".format(top_cdd_probs_j[k].item()))
                            # also retrieve the feature of this sentence
                            f.write("----:----:----:----:----:----:----:----:\n")
                        f.write("========================================\n")

                    bottom_cdd_hyps_j = []
                    bottom_cdd_probs_j = 1-bottom_cdd_logits[j]
                    for sid_k in bottom_cdd_pred_sids[j]:
                        bottom_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])

                    with open('../result/eval_logging_bottom_{0}_{1}.txt'.format(dataset_name, label_format), 'a') as f:
                        f.write("user id: {}\n".format(true_userid_j))
                        f.write("item id: {}\n".format(true_itemid_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        for k in range(topk_candidate):
                            # key is the sentence content
                            # value is the probability of this sentence
                            f.write("candidate sentence: {}\n".format(bottom_cdd_hyps_j[k]))
                            f.write("prob: {}\n".format(bottom_cdd_probs_j[k].item()))
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

                    # bleu_scores_j = compute_bleu([refs_j], [hyps_j])
                    bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                    bleu_list.append(bleu_scores_j)

                    # bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_bleu([refs_j], [hyps_j])
                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())

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

        assert len(rouge_1_f_list) == train_test_differ_cnt

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

        # self.m_recall_feature = np.mean(feature_recall_list)
        # self.m_precision_feature = np.mean(feature_precision_list)

        # self.m_mean_f_num_per_sent_pred = np.mean(feature_num_per_sentence_pred)
        # self.m_mean_f_num_per_sent_ref = np.mean(feature_num_per_sentence_ref)
        # self.m_mean_f_num_per_review_pred = np.mean(feature_num_per_review_pred)
        # self.m_mean_f_num_per_review_ref = np.mean(feature_num_per_review_ref)

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
        # print("feature recall: {}".format(self.m_recall_feature))
        # print("feature precision: {}".format(self.m_precision_feature))
        # print("feature num per sentence [pred]: {}".format(self.m_mean_f_num_per_sent_pred))
        # print("feature num per sentence [ref]: {}".format(self.m_mean_f_num_per_sent_ref))
        # print("feature num per review [pred]: {}".format(self.m_mean_f_num_per_review_pred))
        # print("feature num per review [ref]: {}".format(self.m_mean_f_num_per_review_ref))
        # print("total number of sentence [pred]: {}".format(len(feature_num_per_sentence_pred)))
        # print("total number of sentence [ref]: {}".format(len(feature_num_per_sentence_ref)))
        # print("total number of review [pred]: {}".format(len(feature_num_per_review_pred)))
        # print("total number of review [ref]: {}".format(len(feature_num_per_review_ref)))

        with open('../result/eval_metrics_{0}_{1}.txt'.format(dataset_name, label_format), 'w') as f:
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
            # print("feature recall: {}\n".format(self.m_recall_feature), file=f)
            # print("feature precision: {}\n".format(self.m_precision_feature), file=f)
            print("Total number of user-item on testset (not appear in trainset): {}\n".format(train_test_differ_cnt), file=f)
            print("Total number of user-item on testset (appear in trainset): {}\n".format(train_test_overlap_cnt), file=f)

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """ ngram blocking
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param n_win:   ngram window size, i.e. which n-gram we are using. n_win can be 2,3,4,...
        :param k:       we are selecting the top-k sentences

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx = []
        # TODO: Also return the probability(i.e. logit) of the selected sentences
        batch_select_proba = []
        batch_select_rank = []
        assert len(sents) == len(p_sent)
        assert len(sents) == batch_size
        assert len(sents[0]) == len(p_sent[0])
        # print(sents)
        # print("batch size (sents): {}".format(len(sents)))
        for i in range(len(sents)):
            # print(len(sents[i]))
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        # print(p_sent)
        # print(p_sent.shape)
        for batch_idx in range(batch_size):
            ngram_list = []
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx = []
            select_proba = []
            select_rank = []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except:
                    print("i: {0} \t idx: {1}".format(batch_idx, idx))
                cur_tokens = cur_sent.split()
                overlap_flag = False
                cur_sent_ngrams = []
                for i in range(len(cur_tokens)-n_win+1):
                    this_ngram = " ".join(cur_tokens[i:(i+n_win)])
                    if this_ngram in ngram_list:
                        overlap_flag = True
                        break
                    else:
                        cur_sent_ngrams.append(this_ngram)
                if not overlap_flag:
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    ngram_list.extend(cur_sent_ngrams)
                    if len(select_idx) >= k:
                        break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # convert list to torch tensor
        batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank
