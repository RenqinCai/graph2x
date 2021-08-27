import os
import json
import pickle
import random
import datetime

import torch
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from numpy.core.numeric import indices

from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision, get_recall_precision_f1, get_sentence_bleu, get_recall_precision_f1_random
from rouge import Rouge
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu


dataset_name = 'medium_500_pure'
label_format = 'soft_label'
# methods to select predicted sentence
use_blocking = False        # whether using 3-gram blocking or not
use_filtering = False       # whether using bleu score based filtering or not
use_trigram_feat_unigram_blocking = True
random_sampling = False
bleu_filter_value = 0.25

save_predict = False
get_statistics = False
save_sentence_selected = False
save_feature_selected = False
random_features = False

save_hyps_refs = True
compute_rouge_score = True
compute_bleu_score = True


class EVAL_PRED(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_batch_size_eval = args.batch_size_eval
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid
        self.m_sent2sid = vocab_obj.m_sent2sid
        self.m_train_sent_num = vocab_obj.m_train_sent_num

        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}
        # get sid to sent_id mapping
        self.m_sid2sentid = {self.m_sent2sid[k]: k for k in self.m_sent2sid}

        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path
        self.select_s_topk = args.select_topk_s
        self.m_pred_scores_file = os.path.join(self.m_eval_output_path, "pred_scores.json")

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))
        if use_blocking:
            print("Using trigram blocking.")
        elif use_filtering:
            print("Using bleu-based filtering.")
        elif use_trigram_feat_unigram_blocking:
            print("Using trigram blocking + feature unigram blocking.")
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
        trainset_user2featuretf_file = '../../Dataset/ratebeer/{}/train/user/user2featuretf.json'.format(dataset_name)
        trainset_item2featuretf_file = '../../Dataset/ratebeer/{}/train/item/item2featuretf.json'.format(dataset_name)
        trainset_sentid2featuretf_file = '../../Dataset/ratebeer/{}/train/sentence/sentence2featuretf.json'.format(dataset_name)
        testset_sentid2featuretf_file = '../../Dataset/ratebeer/{}/test/sentence/sentence2featuretf.json'.format(dataset_name)
        trainset_user2sentid_file = '../../Dataset/ratebeer/{}/train/user/user2sentids.json'.format(dataset_name)
        trainset_item2sentid_file = '../../Dataset/ratebeer/{}/train/item/item2sentids.json'.format(dataset_name)
        trainset_sentid2featuretfidf_file = '../../Dataset/ratebeer/{}/train/sentence/sentence2feature.json'.format(dataset_name)
        # Load the combined train/test set
        trainset_combined_file = '../../Dataset/ratebeer/{}/train_combined.json'.format(dataset_name)
        testset_combined_file = '../../Dataset/ratebeer/{}/test_combined.json'.format(dataset_name)
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
        # Load trainset user to feature tf-value dict
        with open(trainset_user2featuretf_file, 'r') as f:
            print("Load file: {}".format(trainset_user2featuretf_file))
            self.d_trainset_user2featuretf = json.load(f)
        # Load trainset item to feature tf-value dict
        with open(trainset_item2featuretf_file, 'r') as f:
            print("Load file: {}".format(trainset_item2featuretf_file))
            self.d_trainset_item2featuretf = json.load(f)
        # Load trainset sentence id to feature tf-value dict
        with open(trainset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(trainset_sentid2featuretf_file))
            self.d_trainset_sentid2featuretf = json.load(f)
        # Load testset sentence id to feature tf-value dict
        with open(testset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(testset_sentid2featuretf_file))
            self.d_testset_sentid2featuretf = json.load(f)
        # Load trainset user to sentence id dict
        with open(trainset_user2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_user2sentid_file))
            self.d_trainset_user2sentid = json.load(f)
        # Load trainset item to sentence id dict
        with open(trainset_item2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_item2sentid_file))
            self.d_trainset_item2sentid = json.load(f)
        # Load trainset sentence id to feature tf-idf value dict
        with open(trainset_sentid2featuretfidf_file, 'r') as f:
            print("Load file: {}".format(trainset_sentid2featuretfidf_file))
            self.d_trainset_sentid2featuretfidf = json.load(f)
        # Get trainset sid2featuretf dict
        # Load train/test combined review for standard evaluation
        self.d_trainset_combined = dict()
        with open(trainset_combined_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                userid = line_data['user']
                itemid = line_data['item']
                review_text = line_data['review']
                if userid not in self.d_trainset_combined:
                    self.d_trainset_combined[userid] = dict()
                    self.d_trainset_combined[userid][itemid] = review_text
                else:
                    assert itemid not in self.d_trainset_combined[userid]
                    self.d_trainset_combined[userid][itemid] = review_text
        self.d_testset_combined = dict()
        with open(testset_combined_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                userid = line_data['user']
                itemid = line_data['item']
                review_text = line_data['review']
                if userid not in self.d_testset_combined:
                    self.d_testset_combined[userid] = dict()
                    self.d_testset_combined[userid][itemid] = review_text
                else:
                    assert itemid not in self.d_testset_combined[userid]
                    self.d_testset_combined[userid][itemid] = review_text

    def f_eval(self, train_data, eval_data):
        print("eval new")
        self.f_eval_new(train_data, eval_data)

    def f_eval_new(self, train_data, eval_data):
        """ TODO:
        1. Save Predict/Selected sentences and Reference sentences to compute BLEU using the perl script.
        2. Add mojority vote based baselines.
        3. Seperate code chunks into functions.
        """
        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []

        rouge = Rouge()
        num_empty_hyps = 0
        num_too_long_hyps = 0

        num_sents_per_target_review = []        # number of sentences for each ui-pair's gt review
        # num_features_per_target_review = []     # number of features for each ui-pair's gt review
        # num_unique_features_per_target = []     # number of unique features per ui-pair'g gt review
        # num_sents_per_proxy_review = []         # number of sentences for each ui-pair's proxies
        # num_features_per_proxy_review = []      # number of features for each ui-pair's proxies
        # num_unique_features_per_proxy = []      # number of unique features per ui-pair's gt review

        train_ui_pair_saved_cnt = 0
        test_ui_pair_saved_cnt = 0

        print('--'*10)

        # debug_index = 0
        s_topk = self.select_s_topk
        s_topk_candidate = 3

        cnt_useritem_pair = 0
        cnt_useritem_batch = 0
        # train_test_overlap_cnt = 0
        # train_test_differ_cnt = 0
        save_logging_cnt = 0

        # read pred sentence scores file:








        with torch.no_grad():
            print("Number of training data: {}".format(len(train_data)))
            print("Number of evaluation data: {}".format(len(eval_data)))
            print("Number of topk selected sentences: {}".format(s_topk))
            # Perform Evaluation on eval_data / train_data
            for graph_batch in eval_data:
                if cnt_useritem_batch % 100 == 0:
                    print("... eval ... ", cnt_useritem_batch)

                graph_batch = graph_batch.to(self.m_device)

                # logits: batch_size*max_sen_num
                s_logits, sids, s_masks, target_sids = self.m_network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)

                if random_sampling:
                    pass

                elif get_statistics:
                    pass

                elif use_blocking:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate
                    )
                elif use_trigram_feat_unigram_blocking:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.trigram_unigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, n_win=3, topk=s_topk, topk_cdd=s_topk_candidate
                    )
                elif use_filtering:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.bleu_filtering_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate, bleu_bound=bleu_filter_value
                    )
                else:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.origin_blocking_sent_prediction(
                        s_logits, sids, s_masks, topk=s_topk, topk_cdd=s_topk_candidate
                    )

                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                # Decide the batch_save_flag. To get shorted results, we only print the first several batches' results
                cnt_useritem_batch += 1
                if cnt_useritem_batch <= MAX_batch_output:
                    batch_save_flag = True
                else:
                    batch_save_flag = False
                # Whether to break or continue(i.e. pass) when the batch_save_flag is false
                if batch_save_flag:
                    save_logging_cnt += 1
                else:
                    # pass or break. pass will continue evaluating full batch testing set, break will only
                    # evaluate the first several batches of the testing set.
                    pass
                    # break

                for j in range(batch_size):
                    userid_j = userid[j].item()
                    itemid_j = itemid[j].item()
                    # get the true user/item id
                    true_userid_j = self.m_uid2user[userid_j]
                    true_itemid_j = self.m_iid2item[itemid_j]

                    refs_j_list = []
                    hyps_j_list = []
                    hyps_featureid_j_list = []
                    for sid_k in target_sids[j]:
                        refs_j_list.append(self.m_sid2swords[sid_k.item()])

                    for sid_k in s_pred_sids[j]:
                        hyps_j_list.append(self.m_sid2swords[sid_k.item()])
                        hyps_featureid_j_list.extend(self.d_trainset_sid2feature[sid_k.item()])

                    hyps_num_unique_features = len(set(hyps_featureid_j_list))

                    hyps_j = " ".join(hyps_j_list)
                    refs_j = " ".join(refs_j_list)

                    # proxy_j_list = []
                    # # get the proxy's sentences' id.
                    # # NOTE: proxy sentences' id is on the trainset
                    # for sid_k in self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][-1]:
                    #     proxy_j_list.append(self.d_trainset_id2sent[sid_k])
                    # proxy_j = " ".join(proxy_j_list)

                    # add the number of sentences of the gt review
                    # num_sents_per_target_review.append(len(refs_j_list))
                    # add the number of sentences of the proxy review
                    # num_sents_per_proxy_review.append(len(proxy_j_list))

                    # Get the featureid and feature logits
                    f_logits_j = f_logits[j]
                    fid_j = fids[j].cpu()
                    hidden_f_batch_j = hidden_f_batch[j].cpu()
                    # print("f_logits_j: {}".format(f_logits_j.shape))
                    # print("fid_j: {}".format(fid_j.shape))
                    # print("hidden_f_batch_j: {}".format(hidden_f_batch_j.shape))
                    # mask_f_j = f_masks[j].cpu()
                    target_f_labels_j = target_f_labels[j].cpu()
                    # print("target f albels, shape: {}".format(target_f_labels_j.shape))
                    # print("target f labels: {}".format(target_f_labels_j.squeeze()))
                    f_num_j = target_f_labels_j.size(0)
                    mask_f_logits_j = f_logits_j[:f_num_j].cpu()
                    mask_fid_j = fid_j[:f_num_j]
                    mask_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in mask_fid_j]
                    mask_hidden_f_j = hidden_f_batch_j[:f_num_j]
                    # print("mask_f_logits: {}".format(mask_f_logits_j.shape))
                    # print("mask_fid_j: {}".format(mask_fid_j.shape))
                    # print("mask_hidden_f_j: {}".format(mask_hidden_f_j.shape))

                    cnt_useritem_pair += 1

                    if save_sentence_selected and batch_save_flag:
                        self.save_predict_sentences(
                            true_userid=true_userid_j,
                            true_itemid=true_itemid_j,
                            refs_sent=refs_j,
                            hyps_sent=hyps_j,
                            topk_logits=s_topk_logits[j],
                            pred_sids=s_pred_sids[j],
                            top_cdd_logits=s_top_cdd_logits[j],
                            top_cdd_pred_sids=s_top_cdd_pred_sids[j],
                            bottom_cdd_logits=s_bottom_cdd_logits[j],
                            bottom_cdd_pred_sids=s_bottom_cdd_pred_sids[j],
                            s_topk_candidate=s_topk_candidate
                        )

                    if save_hyps_refs:
                        # Compute ROUGE/BLEU score
                        # Save refs and selected hyps into file
                        refs_file = os.path.join(self.m_eval_output_path, 'reference.txt')
                        hyps_file = os.path.join(self.m_eval_output_path, 'hypothesis.txt')
                        refs_json_file = os.path.join(self.m_eval_output_path, 'refs.json')
                        hyps_json_file = os.path.join(self.m_eval_output_path, 'hyps.json')
                        # Get the true combined reference text in data
                        true_combined_ref = self.d_testset_combined[true_userid_j][true_itemid_j]
                        # write reference raw text
                        with open(refs_file, 'a') as f_ref:
                            # f_ref.write(refs_j)
                            f_ref.write(true_combined_ref)
                            f_ref.write("\n")
                        # write reference raw text with user/item id
                        with open(refs_json_file, 'a') as f_ref_json:
                            # cur_ref_json = {
                            #     'user': true_userid_j, 'item': true_itemid_j, 'text': refs_j
                            # }
                            cur_ref_json = {
                                'user': true_userid_j, 'item': true_itemid_j, 'text': true_combined_ref
                            }
                            json.dump(cur_ref_json, f_ref_json)
                            f_ref_json.write("\n")
                        if use_majority_vote_popularity:
                            cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                            hyps_pop, _, _, _ = self.majority_vote_popularity(
                                true_userid_j, true_itemid_j, cur_cdd_sents, topk=s_topk)
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_pop)
                                f_hyp.write("\n")
                            with open(hyps_json_file, 'a') as f_hyp_json:
                                cur_hyp_json = {
                                    'user': true_userid_j, 'item': true_itemid_j, 'text': hyps_pop
                                }
                                json.dump(cur_hyp_json, f_hyp_json)
                                f_hyp_json.write("\n")
                        elif use_majority_vote_popularity_itemside:
                            cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                            # Get item-side feature ids
                            item_featureids = set(self.d_trainset_item2featuretf[true_itemid_j].keys())
                            hyps_pop, _, _, _ = self.majority_vote_popularity_itemside(
                                true_userid_j, true_itemid_j, cur_cdd_sents, item_featureids, topk=s_topk)
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_pop)
                                f_hyp.write("\n")
                            with open(hyps_json_file, 'a') as f_hyp_json:
                                cur_hyp_json = {
                                    'user': true_userid_j, 'item': true_itemid_j, 'text': hyps_pop
                                }
                                json.dump(cur_hyp_json, f_hyp_json)
                                f_hyp_json.write("\n")
                        elif use_majority_vote_feature_score:
                            cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                            hyps_f_score, _, _, _ = self.majority_vote_predicted_feature(
                                true_userid_j, true_itemid_j, cur_cdd_sents, mask_f_logits_j, mask_featureid_j, topk=s_topk)
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_f_score)
                                f_hyp.write("\n")
                            with open(hyps_json_file, 'a') as f_hyp_json:
                                cur_hyp_json = {
                                    'user': true_userid_j, 'item': true_itemid_j, 'text': hyps_f_score
                                }
                                json.dump(cur_hyp_json, f_hyp_json)
                                f_hyp_json.write("\n")
                        else:
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_j)
                                f_hyp.write("\n")
                            with open(hyps_json_file, 'a') as f_hyp_json:
                                cur_hyp_json = {
                                    'user': true_userid_j, 'item': true_itemid_j, 'text': hyps_j
                                }
                                json.dump(cur_hyp_json, f_hyp_json)
                                f_hyp_json.write("\n")

                    if use_majority_vote_popularity and not save_hyps_refs:
                        cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                        hyps_pop, _, topk_cdd_scores, hyps_sent_feature_scores = self.majority_vote_popularity(
                            true_userid_j, true_itemid_j, cur_cdd_sents, topk=s_topk)
                        popu_log_file = os.path.join(self.m_eval_output_path, 'popularity_majority_vote.txt')
                        with open(popu_log_file, 'a') as f_popu:
                            f_popu.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                            f_popu.write("Refs: {}\n".format(refs_j))
                            f_popu.write("Hyps: {}\n".format(hyps_pop))
                            f_popu.write("Hyps sent scores: {}\n".format(topk_cdd_scores.numpy().tolist()))
                            # write feature weighted scores
                            for featureid_score_dict in hyps_sent_feature_scores:
                                featureword_score_dict = dict()
                                for key, value in featureid_score_dict.items():
                                    assert isinstance(key, str)
                                    featureid = key
                                    featureword = self.d_id2feature[featureid]
                                    featureword_score_dict[featureword] = value
                                # write this featureword-score dict into file
                                f_popu.write(json.dumps(featureword_score_dict))
                                f_popu.write("\n")
                            f_popu.write("========================================\n")

                    if compute_rouge_score:
                        try:
                            scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)
                        except:
                            if hyps_j == '':
                                hyps_j = '<unk>'
                                scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)
                                num_empty_hyps += 1
                            else:
                                # hyps may be too long, then we truncate it to be half
                                hyps_j_trunc = " ".join(hyps_j_list[0:int(s_topk/2)])
                                scores_j = rouge.get_scores(hyps_j_trunc, refs_j, avg=True)
                                num_too_long_hyps += 1

                        rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                        rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                        rouge_1_p_list.append(scores_j["rouge-1"]["p"])

                        rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                        rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                        rouge_2_p_list.append(scores_j["rouge-2"]["p"])

                        rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                        rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                        rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                    if compute_bleu_score:
                        bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                        bleu_list.append(bleu_scores_j)

                        bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())

                        bleu_1_list.append(bleu_1_scores_j)
                        bleu_2_list.append(bleu_2_scores_j)
                        bleu_3_list.append(bleu_3_scores_j)
                        bleu_4_list.append(bleu_4_scores_j)

                # exit()

        if compute_rouge_score:
            self.m_mean_eval_rouge_1_f = np.mean(rouge_1_f_list)
            self.m_mean_eval_rouge_1_r = np.mean(rouge_1_r_list)
            self.m_mean_eval_rouge_1_p = np.mean(rouge_1_p_list)

            self.m_mean_eval_rouge_2_f = np.mean(rouge_2_f_list)
            self.m_mean_eval_rouge_2_r = np.mean(rouge_2_r_list)
            self.m_mean_eval_rouge_2_p = np.mean(rouge_2_p_list)

            self.m_mean_eval_rouge_l_f = np.mean(rouge_l_f_list)
            self.m_mean_eval_rouge_l_r = np.mean(rouge_l_r_list)
            self.m_mean_eval_rouge_l_p = np.mean(rouge_l_p_list)

        if compute_bleu_score:
            self.m_mean_eval_bleu = np.mean(bleu_list)
            self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
            self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
            self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
            self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        print("Totally {0} batches ({1} data instances).\nAmong them, {2} batches are saved into logging files.".format(
            len(eval_data), cnt_useritem_pair, save_logging_cnt
        ))
        print("Number of too long hypothesis: {}".format(num_too_long_hyps))

        if compute_rouge_score and compute_bleu_score:
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

            metric_log_file = os.path.join(self.m_eval_output_path, 'eval_metrics_{0}_{1}.txt'.format(dataset_name, label_format))
            with open(metric_log_file, 'w') as f:
                print("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f \n" % (
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


    def trigram_feat_unigram_blocking(self, sents, p_sent, n_win=3, topk=5, use_feat_freq_in_sent=False):
        """ a combination of trigram blocking and soft feature-unigram blocking
        :param sents:   batch of list of candidate sentence, each candidate sentence is a string.
                        shape: (batch_size, sent_num)
        :param p_sent:  torch tensor. batch of predicted scores of each candidate sentence.
                        shape: (batch_size, sent_num)
        :param topk:    we are selecting the top-k sentences.
        :param use_feat_freq_in_sent:  when compute the unigram feature word blocking,
                        using the frequency of the feature word in the sentence or only set the frequency
                        to be 1 when a feature appears in the sentence (regardless of real freq in that sent).

        :return:        selected index of sids
        """

        batch_size = p_sent.size(0)
        batch_select_idx, batch_select_proba, batch_select_rank = [], [], []
        feat_overlap_threshold = 1
        # 1. Perform trigram blocking, get the top-100 predicted sentences
        batch_select_idx_trigram, batch_select_proba_trigram, batch_select_rank_trigram = self.ngram_blocking(
            sents=sents, p_sent=p_sent, n_win=n_win, k=100
        )
        # 2. Perform feature-unigram blocking
        for batch_idx in range(batch_size):
            feat_word_freq = dict()
            select_idx, select_proba, select_rank = [], [], []
            for idx, sent_idx in enumerate(batch_select_idx_trigram[batch_idx]):
                cur_sent = sents[batch_idx][sent_idx]
                cur_words = cur_sent.split()
                block_flag = False
                cur_feature_words = dict()
                for word in cur_words:
                    # check if this word is feature word
                    if word in self.d_feature2id.keys():
                        if word in cur_feature_words:
                            cur_feature_words[word] += 1
                        else:
                            cur_feature_words[word] = 1
                if use_feat_freq_in_sent:
                    for word, freq in cur_feature_words.items():
                        if word in feat_word_freq:
                            if freq + feat_word_freq[word] > feat_overlap_threshold:
                                block_flag = True
                                break
                        else:
                            if freq > 2:
                                block_flag = True
                                break
                    if not block_flag:
                        select_idx.append(sent_idx)
                        select_proba.append(batch_select_proba_trigram[batch_idx][idx])
                        select_rank.append(batch_select_rank_trigram[batch_idx][idx])
                        for word, freq in cur_feature_words.items():
                            if word in feat_word_freq:
                                feat_word_freq[word] += freq
                            else:
                                feat_word_freq[word] = freq
                else:
                    for word in cur_feature_words.keys():
                        if word in feat_word_freq:
                            if feat_word_freq[word] == feat_overlap_threshold:
                                block_flag = True
                                break
                    if not block_flag:
                        select_idx.append(sent_idx)
                        select_proba.append(batch_select_proba_trigram[batch_idx][idx])
                        select_rank.append(batch_select_rank_trigram[batch_idx][idx])
                        for word in cur_feature_words.keys():
                            if word in feat_word_freq:
                                feat_word_freq[word] += 1
                            else:
                                feat_word_freq[word] = 1
                        if len(select_idx) >= topk:
                            break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # # convert list to torch tensor, which is used for later gather element by index
        # batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def ngram_blocking(self, sents, p_sent, n_win, k, use_topk=True):
        """ ngram blocking
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param n_win:   ngram window size, i.e. which n-gram we are using. n_win can be 2,3,4,...
        :param k:       we are selecting the top-k sentences

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx, batch_select_proba, batch_select_rank = [], [], []
        assert len(sents) == len(p_sent)
        assert len(sents) == batch_size
        assert len(sents[0]) == len(p_sent[0])
        for i in range(len(sents)):
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        for batch_idx in range(batch_size):
            ngram_list = []
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx, select_proba, select_rank = [], [], []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except KeyError:
                    print("Error! i: {0} \t idx: {1}".format(batch_idx, idx))
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
                    if p_sent[batch_idx][idx] < 0.0:
                        # this suggest that this idx is already the pad idx
                        break
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    ngram_list.extend(cur_sent_ngrams)
                    if use_topk and len(select_idx) >= k:
                        break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # # convert list to torch tensor
        # batch_select_idx = torch.LongTensor(batch_select_idx)
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
                except KeyError:
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

    def origin_blocking_sent_prediction(self, s_logits, sids, s_masks, topk=3, topk_cdd=20):
        # incase some not well-trained model will predict the logits for all sentences as 0.0, we apply masks on it
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        # 1. get the top-k predicted sentences which form the hypothesis
        topk_logits, topk_pred_snids = torch.topk(masked_s_logits, topk, dim=1)
        # topk sentence index
        # pred_sids: shape: (batch_size, topk_sent)
        sids = sids.cpu()
        pred_sids = sids.gather(dim=1, index=topk_pred_snids)
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_logits, top_cdd_pred_snids = torch.topk(masked_s_logits, topk_cdd, dim=1)
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def trigram_blocking_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, topk_cdd=20):
        # use n-gram blocking
        # get all the sentence content
        batch_sents_content = []
        assert len(sids) == s_logits.size(0)      # this is the batch size
        for i in range(batch_size):
            cur_sents_content = []
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        sids = sids.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk
        )
        # pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
        pred_sids = []
        for i in range(batch_size):
            pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(ngram_block_pred_snids[i])))
        topk_logits = ngram_block_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk_cdd
        )
        # top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        top_cdd_pred_sids = []
        for i in range(batch_size):
            top_cdd_pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(top_cdd_pred_snids[i])))
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def trigram_unigram_blocking_sent_prediction(self, s_logits, sids, s_masks, n_win=3, topk=5, topk_cdd=20):
        """use trigram blocking and soft unigram feature word blocking
        :param: s_logits:
        :param: sids:
        :param: s_masks:
        :param: topk:      select the top-k sentence. default: 5
        :param: topk_cdd:  sanity check. select the top-k candidate sentences, used to tune topk. default: 20
        """
        batch_sents_content = []
        assert sids.size(0) == s_logits.size(0)     # this is the batch_size
        batch_size = sids.size(0)
        for i in range(batch_size):
            cur_sents_content = []
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        sids = sids.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        trigram_feat_block_pred_snids, trigram_feat_block_pred_proba, trigram_feat_block_pred_rank = self.trigram_feat_unigram_blocking(
            sents=batch_sents_content, p_sent=masked_s_logits, n_win=n_win, topk=topk, use_feat_freq_in_sent=False
        )
        pred_sids = []
        for i in range(batch_size):
            pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(trigram_feat_block_pred_snids[i])))
        topk_logits = trigram_feat_block_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.trigram_feat_unigram_blocking(
            sents=batch_sents_content, p_sent=masked_s_logits, n_win=n_win, topk=topk_cdd, use_feat_freq_in_sent=False
        )
        # top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        top_cdd_pred_sids = []
        for i in range(batch_size):
            top_cdd_pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(top_cdd_pred_snids[i])))
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def bleu_filtering_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, topk_cdd=20, bleu_bound=0.25):
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
            batch_sents_content, masked_s_logits, k=topk, filter_value=bleu_bound)
        pred_sids = sids.gather(dim=1, index=bleu_filter_pred_snids)
        topk_logits = bleu_filter_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.bleu_filtering(
            batch_sents_content, masked_s_logits, k=topk_cdd, filter_value=bleu_bound)
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def save_predict_sentences(self, true_userid, true_itemid, refs_sent, hyps_sent, topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids, s_topk_candidate=20):
        # top-predicted/selected sentences
        predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_{0}_{1}.txt'.format(dataset_name, label_format))
        with open(predict_log_file, 'a') as f:
            f.write("user id: {}\n".format(true_userid))
            f.write("item id: {}\n".format(true_itemid))
            f.write("hyps: {}\n".format(hyps_sent))
            f.write("refs: {}\n".format(refs_sent))
            f.write("probas: {}\n".format(topk_logits))
            # if use_blocking:
            #     f.write("rank: {}\n".format(ngram_block_pred_rank[j]))
            # elif use_filtering:
            #     f.write("rank: {}\n".format(bleu_filter_pred_rank[j]))
            f.write("========================================\n")
        # top-ranked sentences
        top_cdd_hyps_j = []
        top_cdd_probs_j = top_cdd_logits
        for sid_k in top_cdd_pred_sids:
            top_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])
        top_predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_top_{0}_{1}.txt'.format(dataset_name, label_format))
        with open(top_predict_log_file, 'a') as f:
            f.write("user id: {}\n".format(true_userid))
            f.write("item id: {}\n".format(true_itemid))
            f.write("refs: {}\n".format(refs_sent))
            for k in range(s_topk_candidate):
                # key is the sentence content
                # value is the probability of this sentence
                f.write("candidate sentence: {}\n".format(top_cdd_hyps_j[k]))
                f.write("prob: {}\n".format(top_cdd_probs_j[k].item()))
                # also retrieve the feature of this sentence
                f.write("----:----:----:----:----:----:----:----:\n")
            f.write("========================================\n")
        # bottom-ranked sentences
        bottom_cdd_hyps_j = []
        bottom_cdd_probs_j = 1-bottom_cdd_logits
        for sid_k in bottom_cdd_pred_sids:
            bottom_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])
        bottom_predict_log_file = os.path.join(self.m_eval_output_path, 'eval_logging_bottom_{0}_{1}.txt'.format(dataset_name, label_format))
        with open(bottom_predict_log_file, 'a') as f:
            f.write("user id: {}\n".format(true_userid))
            f.write("item id: {}\n".format(true_itemid))
            f.write("refs_j: {}\n".format(refs_sent))
            for k in range(s_topk_candidate):
                # key is the sentence content
                # value is the probability of this sentence
                f.write("candidate sentence: {}\n".format(bottom_cdd_hyps_j[k]))
                f.write("prob: {}\n".format(bottom_cdd_probs_j[k].item()))
                f.write("----:----:----:----:----:----:----:----:\n")
            f.write("========================================\n")

    def combine_featuretf(self, user_featuretf, item_featuretf):
        """ Add 2 dict together to get the feature tf-value on this user and this item
        :param: user_featuretf: user-side feature frequency (i.e. tf) dict
                                key: featureid, value: frequency (i.e. tf) of this featureid
        :param: item_featuretf: item-side feature frequency (i.e. tf) dict
                                key: featureid, value: frequency (i.e. tf) of this featureid
        return: useritem_featuretf, key: featureid, value: frequency(i.e. tf) of this featureid
        """

        useritem_featuretf = dict()
        for key, value in user_featuretf.items():
            feature_id = key
            assert isinstance(feature_id, str)
            feature_tf = value
            assert isinstance(feature_tf, int)
            assert feature_id not in useritem_featuretf
            useritem_featuretf[feature_id] = feature_tf
        for key, value in item_featuretf.items():
            feature_id = key
            assert isinstance(feature_id, str)
            feature_tf = value
            assert isinstance(feature_tf, int)
            if feature_id not in useritem_featuretf:
                useritem_featuretf[feature_id] = feature_tf
            else:
                useritem_featuretf[feature_id] += feature_tf

        return useritem_featuretf

    def get_sid2featuretf_train(self, trainset_sentid2featuretf, sent2sid):
        """ Get sid to featuretf mapping (on train set).
        """
        trainset_sid2featuretf = dict()
        for key, value in trainset_sentid2featuretf.items():
            assert isinstance(key, str)
            sentid = key
            sid = sent2sid[sentid]
            assert sid not in trainset_sid2featuretf
            trainset_sid2featuretf[sid] = value
        return trainset_sid2featuretf

    def get_sid2featuretf_eval(self, testset_sentid2featuretf, sent2sid, train_sent_num):
        """ Get sid to featuretf mapping (on valid/test set).
            During constructing the graph data, we load the valid/test sentences. Since the
            original sentid is seperated from train-set sentence sentid, we first add the
            sentid of valid/test-set with train_sent_num and then mapping the new sent_id
            to sid. Therefore, to simplify the mapping between sid and featureid (and also
            feature tf) we need to construct this mapping here.
        """
        testset_sid2featuretf = dict()
        for key, value in testset_sentid2featuretf.items():
            assert isinstance(key, str)
            sentid = int(key) + train_sent_num
            sentid = str(sentid)
            sid = sent2sid[sentid]
            assert sid not in testset_sid2featuretf
            testset_sid2featuretf[sid] = value
        return testset_sid2featuretf

    def majority_vote_popularity(self, user_id, item_id, cdd_sents, topk=3):
        # Get this user-item pair's candidate sentences union feature tf-value
        cdd_featuretf = dict()
        for sent_id in cdd_sents:
            cur_featuretf = self.d_trainset_sentid2featuretf[sent_id]
            for key, value in cur_featuretf.items():
                feature_id = key
                assert isinstance(feature_id, str)
                feature_tf = value
                assert isinstance(feature_tf, int)
                if feature_id not in cdd_featuretf:
                    cdd_featuretf[feature_id] = feature_tf
                else:
                    cdd_featuretf[feature_id] += feature_tf
        # cdd sentence selection based on the score function of cdd_featuretf
        hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores = self.mojority_vote_selection(
            cdd_sents=cdd_sents,
            feature_score=cdd_featuretf,
            topk=topk)
        return hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores

    def majority_vote_popularity_itemside(self, user_id, item_id, cdd_sents, item_features, topk=3):
        # Get this user-item pairs cdd sentences union item-side feature tf-value
        cdd_featuretf = dict()
        for sent_id in cdd_sents:
            cur_featuretf = self.d_trainset_sentid2featuretf[sent_id]
            for key, value in cur_featuretf.items():
                feature_id = key
                assert isinstance(feature_id, str)
                feature_tf = value
                assert isinstance(feature_tf, int)
                if feature_id in item_features:
                    if feature_id not in cdd_featuretf:
                        cdd_featuretf[feature_id] = feature_tf
                    else:
                        cdd_featuretf[feature_id] += feature_tf
        # cdd sentence selection based on score function of item-side cdd_featuretf
        hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores = self.mojority_vote_selection_itemside(
            cdd_sents=cdd_sents,
            feature_score=cdd_featuretf,
            item_features=item_features,
            topk=topk)
        return hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores

    def majority_vote_predicted_feature(self, user_id, item_id, cdd_sents, f_logits, featureids, topk=3):
        # Get each feature's predicted score
        feature_pred_score = dict()
        for idx, featureid in enumerate(featureids):
            assert featureid not in feature_pred_score
            feature_pred_score[featureid] = f_logits[idx].item()
        # cdd sentence selection based on the score function of feature_pred_score
        hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores = self.mojority_vote_selection(
            cdd_sents=cdd_sents,
            feature_score=feature_pred_score,
            topk=topk)
        return hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores

    def mojority_vote_selection(self, cdd_sents, feature_score, topk=3):
        # Compute the score for each cdd sentence
        cdd_scores = list()
        cdd_sentid_to_featureid_scores = dict()
        for sent_id in cdd_sents:
            # get the feature tf-value dict of this sent
            cur_featuretf = self.d_trainset_sentid2featuretf[sent_id]
            # count total number of features
            cur_num_features = 0
            cur_cdd_score = 0.0
            feature_score_dict = dict()
            for key, value in cur_featuretf.items():
                feature_id = key
                feature_tf = value
                cur_num_features += feature_tf
                feature_weighted_score = feature_score[feature_id] * feature_tf
                cur_cdd_score += feature_weighted_score
                feature_score_dict[feature_id] = feature_weighted_score
            assert cur_num_features > 0
            # normalize the cumu score by the number of features in this sentence
            cur_cdd_score = cur_cdd_score / cur_num_features
            cdd_scores.append(cur_cdd_score)
            cdd_sentid_to_featureid_scores[sent_id] = feature_score_dict
        # Get the topk cdd sentences based on the cdd_scores
        cdd_scores_th = torch.tensor(cdd_scores).cpu()
        topk_cdd_scores, topk_cdd_indices = torch.topk(cdd_scores_th, topk)
        # Construct the hypothesis based on the topk cdd sents
        hyps_sent_list = list()
        hyps_sent_feature_scores = list()
        for idx in topk_cdd_indices:
            sent_id = cdd_sents[idx]        # this is sentence id
            sent_content = self.d_trainset_id2sent[sent_id]
            hyps_sent_list.append(sent_content)
            hyps_sent_feature_scores.append(cdd_sentid_to_featureid_scores[sent_id])
        hyps = " ".join(hyps_sent_list)
        return hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores

    def mojority_vote_selection_itemside(self, cdd_sents, feature_score, item_features, topk=3):
        # Compute the score for each cdd sentence
        cdd_scores = list()
        cdd_sentid_to_featureid_scores = dict()
        for sent_id in cdd_sents:
            # get the feature tf-value dict of this sent
            cur_featuretf = self.d_trainset_sentid2featuretf[sent_id]
            # count total number of features
            cur_num_features = 0
            cur_cdd_score = 0.0
            feature_score_dict = dict()
            for key, value in cur_featuretf.items():
                feature_id = key
                feature_tf = value
                # only consider features from the item-side
                if feature_id in item_features:
                    cur_num_features += feature_tf
                    feature_weighted_score = feature_score[feature_id] * feature_tf
                    cur_cdd_score += feature_weighted_score
                    feature_score_dict[feature_id] = feature_weighted_score
            # This assertation may not be correct since some of the cdd sentences
            # may come from user-side and have no item features.
            # assert cur_num_features > 0
            # normalize the cumu score by the number of features in this sentence
            if cur_num_features > 0:
                cur_cdd_score = cur_cdd_score / cur_num_features
            else:
                cur_cdd_score = 0.0
            cdd_scores.append(cur_cdd_score)
            cdd_sentid_to_featureid_scores[sent_id] = feature_score_dict
        # Get the topk cdd sentences based on the cdd_scores
        cdd_scores_th = torch.tensor(cdd_scores).cpu()
        topk_cdd_scores, topk_cdd_indices = torch.topk(cdd_scores_th, topk)
        # Construct the hypothesis based on the topk cdd sents
        hyps_sent_list = list()
        hyps_sent_feature_scores = list()
        for idx in topk_cdd_indices:
            sent_id = cdd_sents[idx]        # this is sentence id
            sent_content = self.d_trainset_id2sent[sent_id]
            hyps_sent_list.append(sent_content)
            hyps_sent_feature_scores.append(cdd_sentid_to_featureid_scores[sent_id])
        hyps = " ".join(hyps_sent_list)
        return hyps, hyps_sent_list, topk_cdd_scores, hyps_sent_feature_scores

    def get_gt_review_featuretf(self, testset_sid2featuretf, gt_sids):
        """ Get the featureid list and featuretf dict for a list of ground-truth sids
        """
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sid in gt_sids:
            cur_sid_featuretf = testset_sid2featuretf[gt_sid.item()]
            for key, value in cur_sid_featuretf.items():
                assert isinstance(key, str)
                gt_featureid_set.add(int(key))
                if key not in gt_featuretf_dict:
                    gt_featuretf_dict[key] = value
                else:
                    gt_featuretf_dict[key] += value
        return list(gt_featureid_set), gt_featuretf_dict

    def get_sid2feature_train(self, trainset_sentid2featuretfidf, sent2sid):
        trainset_sid2feature = dict()
        for key, value in trainset_sentid2featuretfidf.items():
            assert isinstance(key, str)     # key is the sentid
            sid = sent2sid[key]
            assert sid not in trainset_sid2feature
            trainset_sid2feature[sid] = list(value.keys())
        return trainset_sid2feature