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
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision, get_recall_precision_f1, get_sentence_bleu, get_recall_precision_f1_random
from rouge import Rouge
from nltk.translate import bleu_score
import dgl
import pickle
import random

dataset_name = 'medium_500_pure'
label_format = 'soft_label'
use_blocking = False            # whether using 3-gram blocking or not
use_filtering = False             # whether using bleu score based filtering or not
save_predict = False
random_sampling = True
random_features = False
get_statistics = False
save_sentence_selected = True
save_feature_selected = True
bleu_filter_value = 0.25


class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid

        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}

        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))
        if use_blocking:
            print("Using tri-gram blocking.")
        elif use_filtering:
            print("Using bleu-based filtering.")
        elif random_sampling:
            print("Random sampling.")
        else:
            print("Use the original scores.")

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        trainset_id2sent_file = '../../Dataset/ratebeer/{}/train/sentence/id2sentence.json'.format(dataset_name)
        testset_id2sent_file = '../../Dataset/ratebeer/{}/test/sentence/id2sentence.json'.format(dataset_name)
        # testset_sentid2feature_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2feature.json'.format(dataset_name)
        # trainset_useritem_pair_file = '../../Dataset/ratebeer/{}/train/useritem_pairs.json'.format(dataset_name)
        testset_useritem_cdd_withproxy_file = '../../Dataset/ratebeer/{}/test/useritem2sentids_withproxy.json'.format(dataset_name)
        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        with open(trainset_id2sent_file, 'r') as f:
            print("Load file: {}".format(trainset_id2sent_file))
            self.d_trainset_id2sent = json.load(f)
        with open(testset_id2sent_file, 'r') as f:
            print("Load file: {}".format(testset_id2sent_file))
            self.d_testset_id2sent = json.load(f)
        # with open(testset_sentid2feature_file, 'r') as f:
        #     print("Load file: {}".format(testset_sentid2feature_file))
        #     self.d_testsetsentid2feature = json.load(f)
        # with open(trainset_useritem_pair_file, 'r') as f:
        #     print("Load file: {}".format(trainset_useritem_pair_file))
        #     self.d_trainset_useritempair = json.load(f)
        with open(testset_useritem_cdd_withproxy_file, 'r') as f:
            print("Load file: {}".format(testset_useritem_cdd_withproxy_file))
            self.d_testset_useritem_cdd_withproxy = json.load(f)

        # save sid2words mapping
        if save_predict:
            self.this_DIR = '../data_postprocess/{}'.format(dataset_name)
            if not os.path.isdir(self.this_DIR):
                os.makedirs(self.this_DIR)
                print("create folder: {}".format(self.this_DIR))
            else:
                print("{} folder already exists.".format(self.this_DIR))
            this_mapping_file = os.path.join(self.this_DIR, 'sid2swords.pickle')
            with open(this_mapping_file, 'wb') as handle:
                pickle.dump(self.m_sid2swords, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        # not using feature id, but using true feature
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

        # recall_list = []
        # precision_list = []
        # F1_list = []

        f_recall_list = []
        f_precision_list = []
        f_F1_list = []
        f_auc_list = []

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

        num_sents_per_target_review = []        # number of sentences for each ui-pair's gt review
        num_features_per_target_review = []     # number of features for each ui-pair's gt review
        num_unique_features_per_target = []     # number of unique features per ui-pair'g gt review
        num_sents_per_proxy_review = []         # number of sentences for each ui-pair's proxies
        num_features_per_proxy_review = []      # number of features for each ui-pair's proxies
        num_unique_features_per_proxy = []      # number of unique features per ui-pair's gt review

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
        save_logging_cnt = 0
        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))

            for graph_batch in eval_data:
            # for graph_batch in train_data:
                if cnt_useritem_batch % 100 == 0:
                    print("... eval ... ", cnt_useritem_batch)

                graph_batch = graph_batch.to(self.m_device)

                # logits: batch_size*max_sen_num
                s_logits, sids, s_masks, target_sids, f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)

                # Save the predict logits and sids
                if save_predict:
                    userid_batch = graph_batch.u_rawid
                    itemid_batch = graph_batch.i_rawid
                    for i in range(batch_size):
                        current_result_dict = {}
                        current_result_dict['user_id'] = userid_batch[i].item()
                        current_result_dict['item_id'] = itemid_batch[i].item()
                        assert len(s_logits[i]) == len(sids[i])
                        assert len(s_logits[i]) == len(s_masks[i])
                        triple_data_list = []
                        for pos in range(len(s_logits[i])):
                            triple_data_list.append(
                                [s_logits[i][pos].item(), sids[i][pos].item(), s_masks[i][pos].item()])
                        current_result_dict['predict_data'] = triple_data_list
                        current_target_sent_sids = []
                        for this_sid in target_sids[i]:
                            current_target_sent_sids.append(this_sid.item())
                        current_result_dict['target'] = current_target_sent_sids

                        # save current_result_dict into json file
                        model_ckpt_name = self.m_model_file.split('.')[0]
                        this_json_file = os.path.join(self.this_DIR, 'result_{}.json'.format(model_ckpt_name))
                        with open(this_json_file, 'a') as f:
                            json.dump(current_result_dict, f)
                            f.write("\n")
                    continue

                if random_sampling:
                    userid_batch = graph_batch.u_rawid
                    itemid_batch = graph_batch.i_rawid
                    for i in range(batch_size):
                        # current_result_dict = {}
                        # current_result_dict['user_id'] = self.m_uid2user[userid_batch[i].item()]
                        # current_result_dict['item_id'] = self.m_iid2item[itemid_batch[i].item()]
                        assert s_logits[i].size(0) == sids[i].size(0)
                        assert s_logits[i].size(0) == s_masks[i].size(0)
                        current_cdd_sent_sids = []
                        current_target_sent_sids = []
                        num_sent = sum(s_masks[i]).item()
                        for pos in range(num_sent):
                            current_cdd_sent_sids.append(sids[i][pos].item())
                        for this_sid in target_sids[i]:
                            current_target_sent_sids.append(this_sid.item())
                        # randomly sample 3 sentences
                        sampled_cdd_sent_sids = random.sample(current_cdd_sent_sids, 3)
                        # get the content
                        refs_j_list = []
                        hyps_j_list = []
                        for sid_cur in current_target_sent_sids:
                            refs_j_list.append(self.m_sid2swords[sid_cur])
                        for sid_cur in sampled_cdd_sent_sids:
                            hyps_j_list.append(self.m_sid2swords[sid_cur])
                        hyps_j = " ".join(hyps_j_list)
                        refs_j = " ".join(refs_j_list)
                        num_sents_per_target_review.append(len(current_target_sent_sids))

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
                    cnt_useritem_batch += 1
                    continue

                elif get_statistics:
                    for i in range(batch_size):
                        this_g = graph_batch[i]
                        labels_feature = this_g.f_label
                        print("shape of feature labels: {}".format(labels_feature.shape))
                        num_features_per_target_review.append(torch.sum(labels_feature).item())
                    continue

                elif use_blocking:
                    # use n-gram blocking
                    # get all the sentence content
                    batch_sents_content = []
                    assert len(sids) == s_logits.size(0)      # this is the batch size
                    for i in range(batch_size):
                        cur_sents_content = []
                        assert len(sids[i]) == len(sids[0])
                        for cur_sid in sids[i]:
                            cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
                        batch_sents_content.append(cur_sents_content)
                    assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
                    sids = sids.cpu()
                    masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
                    # 1. get the top-k predicted sentences which form the hypothesis
                    ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
                        batch_sents_content, masked_s_logits, n_win=3, k=3)
                    pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
                    topk_logits = ngram_block_pred_proba
                    # 2. get the top-20 predicted sentences' content and proba
                    top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(
                        batch_sents_content, masked_s_logits, n_win=3, k=topk_candidate)
                    top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
                    # 3. get the bottom-20 predicted sentences' content and proba
                    reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
                    bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_candidate, dim=1)
                    bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

                elif use_filtering:
                    # use bleu-based filtering
                    # get all the sentence content
                    batch_sents_content = []
                    assert len(sids) == s_logits.size(0)      # this is the batch size
                    for i in range(batch_size):
                        cur_sents_content = []
                        assert len(sids[i]) == len(sids[0])
                        for cur_sid in sids[i]:
                            cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
                        batch_sents_content.append(cur_sents_content)
                    assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
                    sids = sids.cpu()
                    masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
                    # 1. get the top-k predicted sentences which form the hypothesis
                    bleu_filter_pred_snids, bleu_filter_pred_proba, bleu_filter_pred_rank = self.bleu_filtering(
                        batch_sents_content, masked_s_logits, k=3, filter_value=bleu_filter_value)
                    pred_sids = sids.gather(dim=1, index=bleu_filter_pred_snids)
                    topk_logits = bleu_filter_pred_proba
                    # 2. get the top-20 predicted sentences' content and proba
                    top_cdd_pred_snids, top_cdd_logits, _ = self.bleu_filtering(
                        batch_sents_content, masked_s_logits, k=topk_candidate, filter_value=bleu_filter_value)
                    top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
                    # 3. get the bottom-20 predicted sentences' content and proba
                    reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
                    bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_candidate, dim=1)
                    bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

                else:
                    # incase some not well-trained model will predict the logits for all sentences as 0.0, we apply masks on it
                    masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
                    # 1. get the top-k predicted sentences which form the hypothesis
                    topk_logits, topk_pred_snids = torch.topk(masked_s_logits, topk, dim=1)
                    # topk sentence index
                    # pred_sids: shape: (batch_size, topk_sent)
                    sids = sids.cpu()
                    pred_sids = sids.gather(dim=1, index=topk_pred_snids)
                    # 2. get the top-20 predicted sentences' content and proba
                    top_cdd_logits, top_cdd_pred_snids = torch.topk(masked_s_logits, topk_candidate, dim=1)
                    top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
                    # 3. get the bottom-20 predicted sentences' content and proba
                    reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
                    bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_candidate, dim=1)
                    bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                f_num = []

                for j in range(batch_size):
                    g = graph_batch[j]
                    f_num.append(sum(g.f_label).item())

                # if cnt_useritem_batch % 10 == 0:
                #     print(f_num)
                #     print(np.mean(f_num))
                cnt_useritem_batch += 1

                # batch_save_flag = (random.random() <= 0.1)
                if cnt_useritem_batch <= 100:
                    batch_save_flag = True
                else:
                    batch_save_flag = False
                if batch_save_flag:
                    save_logging_cnt += 1
                else:
                    pass
                    # break

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

                    proxy_j_list = []
                    for sid_k in self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][-1]:
                        proxy_j_list.append(self.d_trainset_id2sent[sid_k])
                    proxy_j = " ".join(proxy_j_list)

                    # add the number of sentences of the ground-truth review
                    num_sents_per_target_review.append(len(refs_j_list))


                    # check whether this user-item pair appears in the trainset
                    # if true_userid_j in self.d_trainset_useritempair:
                    #     if true_itemid_j in self.d_trainset_useritempair[true_userid_j]:
                    #         # this user-item pair already appeared in the trainset, ignore this user-item pair.
                    #         train_test_overlap_cnt += 1
                    #         continue
                    #     else:
                    #         train_test_differ_cnt += 1
                    # else:
                    #     raise Exception("user: {} not in trainset but in testset!".format(true_userid_j))

                    if save_sentence_selected and batch_save_flag:
                        predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_{0}_{1}.txt'.format(dataset_name, label_format))
                        with open(predict_log_file, 'a') as f:
                            f.write("user id: {}\n".format(true_userid_j))
                            f.write("item id: {}\n".format(true_itemid_j))
                            f.write("hyps_j: {}\n".format(hyps_j))
                            f.write("refs_j: {}\n".format(refs_j))
                            f.write("probas: {}\n".format(topk_logits[j]))
                            if use_blocking:
                                f.write("rank: {}\n".format(ngram_block_pred_rank[j]))
                            elif use_filtering:
                                f.write("rank: {}\n".format(bleu_filter_pred_rank[j]))
                            f.write("========================================\n")
                        # top-ranked sentences
                        top_cdd_hyps_j = []
                        top_cdd_probs_j = top_cdd_logits[j]
                        for sid_k in top_cdd_pred_sids[j]:
                            top_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])
                        top_predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_top_{0}_{1}.txt'.format(dataset_name, label_format))
                        with open(top_predict_log_file, 'a') as f:
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
                        # bottom-ranked sentences
                        bottom_cdd_hyps_j = []
                        bottom_cdd_probs_j = 1-bottom_cdd_logits[j]
                        for sid_k in bottom_cdd_pred_sids[j]:
                            bottom_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])
                        bottom_predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_bottom_{0}_{1}.txt'.format(dataset_name, label_format))
                        with open(bottom_predict_log_file, 'a') as f:
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

                    ### get feature prediction performance
                    # # f_logits, fids, f_masks, target_f_labels
                    # f_logits_j = f_logits[j]
                    # fid_j = fids[j].cpu()
                    # mask_f_j = f_masks[j].cpu()
                    # target_f_labels_j = target_f_labels[j].cpu()

                    # print("f_logits_j, shape: {}".format(f_logits_j.shape))
                    # print(f_logits_j)
                    # print("fid_j, shape: {}".format(fid_j.shape))
                    # print(fid_j)
                    # print("mask_f_j, shape: {}".format(mask_f_j.shape))
                    # print(mask_f_j)
                    # print("target_f_labels_j, shape: {}".format(target_f_labels_j.shape))
                    # print(target_f_labels_j.squeeze())

                    # assert sum(mask_f_j) == target_f_labels_j.size(0)

                    # f_num_j = target_f_labels_j.size(0)
                    # mask_f_logits_j = f_logits_j[:f_num_j].cpu()

                    # if not random_features:
                    #     f_prec_j, f_recall_j, f_f1_j, f_auc_j, topk_pred_f_j = get_recall_precision_f1(mask_f_logits_j, target_f_labels_j)
                    # else:
                    #     f_prec_j, f_recall_j, f_f1_j, f_auc_j, topk_pred_f_j = get_recall_precision_f1_random(mask_f_logits_j, target_f_labels_j)
                    # f_precision_list.append(f_prec_j)
                    # f_recall_list.append(f_recall_j)
                    # f_F1_list.append(f_f1_j)
                    # f_auc_list.append(f_auc_j)

                    # # get the index of the feature labels (1)
                    # target_fid_index_j = (target_f_labels_j.squeeze() == 1).nonzero(as_tuple=True)[0]
                    # # get the fid of the feature labels
                    # # target_fid_j = fid_j[target_fid_index_j]
                    # target_fid_j = torch.gather(fid_j, dim=0, index=target_fid_index_j)
                    # # get the featureid of the feature labels
                    # target_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in target_fid_j]
                    # # get the feature word of the feature labels
                    # target_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in target_featureid_j]
                    # # get the index of the predicted features
                    # assert sum(topk_pred_f_j).item() == 26
                    # top_pred_fid_index_j = (topk_pred_f_j == 1).nonzero(as_tuple=True)[0]
                    # top_pred_fid_j = torch.gather(fid_j, dim=0, index=top_pred_fid_index_j)
                    # top_pred_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in top_pred_fid_j]
                    # top_pred_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in top_pred_featureid_j]
                    # # get the overlapping features
                    # target_featureid_j_set = set(target_featureid_j)
                    # top_pred_featureid_j_set = set(top_pred_featureid_j)
                    # overlap_featureid_j_set = target_featureid_j_set.intersection(top_pred_featureid_j_set)
                    # overlap_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureid_j_set]

                    # if save_feature_selected and batch_save_flag:
                    #     predict_features_file = os.path.join(
                    #         self.m_eval_output_path,
                    #         'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                    #     with open(predict_features_file, 'a') as f:
                    #         f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                    #         f.write("refs: {}\n".format(refs_j))
                    #         f.write("hyps: {}\n".format(hyps_j))
                    #         f.write("proxy: {}\n".format(proxy_j))
                    #         f.write("target features: {}\n".format(target_featureword_j))
                    #         f.write("top predict features: {}\n".format(top_pred_featureword_j))
                    #         f.write("overlappings: {}\n".format(overlap_featureword_j))
                    #         f.write("Number of features in target: {}\n".format(len(target_featureid_j_set)))
                    #         f.write("Number of features in top-pred: {}\n".format(len(top_pred_featureid_j_set)))
                    #         f.write("Number of feature overlap: {}\n".format(len(overlap_featureid_j_set)))
                    #         f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                    #         f.write("==------==------==------==------==------==------==\n")
                    #     # print("User: {0}\tItem: {1}".format(true_userid_j, true_itemid_j))
                    #     # print("refs: {}".format(refs_j))
                    #     # print("hyps: {}".format(hyps_j))
                    #     # print("proxy: {}".format(proxy_j))
                    #     # print("target features: {}".format(target_featureword_j))
                    #     # print("top predict features: {}".format(top_pred_featureword_j))

                # exit()

        # self.m_mean_feature_num_per_review = np.mean(num_features_per_target_review)
        # print("Mean number of features per review: {}".format(self.m_mean_feature_num_per_review))

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

        # if len(num_sents_per_target_review) != 0:
        #     self.m_mean_num_sents_per_target_review = np.mean(num_sents_per_target_review)
        #     print("Number of sentences for each target review (on average): {}".format(
        #         self.m_mean_num_sents_per_target_review))

        # self.m_mean_f_precision = np.mean(f_precision_list)
        # self.m_mean_f_recall = np.mean(f_recall_list)
        # self.m_mean_f_f1 = np.mean(f_F1_list)
        # self.m_mean_f_auc = np.mean(f_auc_list)

        print("Totally {0} batches ({1} data instances).\nAmong them, {2} batches are saved into logging files.".format(
            len(eval_data), len(bleu_list), save_logging_cnt))

        # print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
        #     self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc))

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

        metric_log_file = os.path.join(self.m_eval_output_path, 'eval_metrics_{0}_{1}.txt'.format(dataset_name, label_format))
        with open(metric_log_file, 'w') as f:
            # print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f \n" % (
            #     self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc), file=f)
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
            # if len(num_sents_per_target_review) != 0:
            #     print("Number of sentences for each target review (on average): {}".format(
            #         self.m_mean_num_sents_per_target_review), file=f)

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

    def bleu_filtering(self, sents, p_sent, k, filter_value=0.25):
        """ bleu filtering
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param k:       we are selecting the top-k sentences
        :param filter_value: the boundary value of bleu-2 + bleu-3 that defines whether we should filter a sentence

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx = []
        batch_select_proba = []
        batch_select_rank = []
        assert len(sents) == len(p_sent)
        assert len(sents[0]) == len(p_sent[0])
        for i in range(len(sents)):
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        for batch_idx in range(batch_size):
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx = []
            select_proba = []
            select_rank = []
            select_sents = []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except:
                    print("Error! batch: {0} \t idx: {1}".format(batch_idx, idx))
                if len(select_sents) == 0:
                    # add current sentence into the selected sentences
                    select_sents.append(cur_sent)
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    if len(select_idx) >= k:
                        break
                else:
                    # compute bleu score
                    this_ref_sents = []
                    for this_sent in select_sents:
                        this_ref_sents.append(this_sent.split())
                    this_hypo_sent = cur_sent.split()
                    sf = bleu_score.SmoothingFunction()
                    bleu_1 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[1.0, 0.0, 0.0, 0.0])
                    bleu_2 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[0.5, 0.5, 0.0, 0.0])
                    bleu_3 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[1.0/3, 1.0/3, 1.0/3, 0.0])
                    if (bleu_2 + bleu_3) < filter_value:
                        # add current sentence into the selected sentences
                        select_sents.append(cur_sent)
                        select_idx.append(idx)
                        select_proba.append(p_sent[batch_idx][idx])
                        select_rank.append(idx_rank)
                        if len(select_idx) >= k:
                            break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # convert list to torch tensor
        batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank
