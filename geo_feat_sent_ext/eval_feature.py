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
from metric import compute_bleu, get_bleu, get_sentence_bleu, get_example_recall_precision
from metric import get_feature_recall_precision, get_recall_precision_f1, get_recall_precision_f1_random, get_recall_precision_f1_popular, get_recall_precision_f1_sent
from metric import get_recall_precision_f1_gt, get_recall_precision_f1_gt_random
from rouge import Rouge
from nltk.translate import bleu_score
import pickle
import random


dataset_name = 'medium_500_pure'
label_format = 'soft_label'

# how to select the top-predicted sentences
use_origin = True
use_trigram = False
use_bleu_filter = False

# select features randomly
random_features = False

# select features based on the popularity
popular_features = False
popular_features_vs_origin = False
popular_features_vs_trigram = False

# select features based on the feature prediction scores
predict_features = True
predict_features_vs_origin = False
predict_features_vs_trigram = False

save_sentence_selected = False
save_feature_selected = False
save_feature_logits = False

# True if the predicted features is compared with the ground-turth features.
# False if the predicted features is compared with the proxy's features.
use_ground_truth = True

avg_proxy_feature_num = 19
avg_gt_feature_num = 15
total_feature_num = 575
MAX_batch_output = 10000


class EVAL_FEATURE(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device
        self.m_mean_loss = 0
        self.m_batch_size = args.batch_size
        self.m_dataset_name = args.data_set
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path
        self.m_model_file_name = args.model_file.split('/')[-1].split('.pt')[0]
        self.m_feature_topk = args.select_topk_f    # default: 15
        self.m_sentence_topk = args.select_topk_s   # default: 3

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

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))
        if random_features:
            print("Use the random features.")
        elif popular_features:
            print("Use the popular features.")
        elif popular_features_vs_origin:
            print("Use the popular features vs. origin predict sentences.")
        elif popular_features_vs_trigram:
            print("Use the popular features vs. trigram predict sentences.")
        elif predict_features_vs_origin:
            print("Use the predict features vs. origin predict sentences.")
        elif predict_features_vs_trigram:
            print("Use the predict features vs. trigram predict sentences.")
        elif use_trigram:
            print("Use the features from sentence after trigram blocking.")
        elif use_origin:
            print("Use the features from sentence based on original scores.")
        else:
            pass
        if predict_features:
            print("Use the predicted features based on the feature prediction scores.")

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        trainset_id2sent_file = '../../Dataset/ratebeer/{}/train/sentence/id2sentence.json'.format(dataset_name)
        testset_id2sent_file = '../../Dataset/ratebeer/{}/test/sentence/id2sentence.json'.format(dataset_name)
        # trainset_useritem_pair_file = '../../Dataset/ratebeer/{}/train/useritem_pairs.json'.format(dataset_name)
        testset_useritem_cdd_withproxy_file = '../../Dataset/ratebeer/{}/test/useritem2sentids_withproxy.json'.format(dataset_name)
        trainset_user2featuretf_file = '../../Dataset/ratebeer/{}/train/user/user2featuretf.json'.format(dataset_name)
        trainset_item2featuretf_file = '../../Dataset/ratebeer/{}/train/item/item2featuretf.json'.format(dataset_name)
        trainset_sentid2featuretfidf_file = '../../Dataset/ratebeer/{}/train/sentence/sentence2feature.json'.format(dataset_name)
        testset_sentid2featuretf_file = '../../Dataset/ratebeer/{}/test/sentence/sentence2featuretf.json'.format(dataset_name)
        trainset_user2sentid_file = '../../Dataset/ratebeer/{}/train/user/user2sentids.json'.format(dataset_name)
        trainset_item2sentid_file = '../../Dataset/ratebeer/{}/train/item/item2sentids.json'.format(dataset_name)

        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        # Load train/test sentence_id to sentence content
        with open(trainset_id2sent_file, 'r') as f:
            print("Load file: {}".format(trainset_id2sent_file))
            self.d_trainset_id2sent = json.load(f)
        with open(testset_id2sent_file, 'r') as f:
            print("Load file: {}".format(testset_id2sent_file))
            self.d_testset_id2sent = json.load(f)
        # # Load trainset user-item pair
        # with open(trainset_useritem_pair_file, 'r') as f:
        #     print("Load file: {}".format(trainset_useritem_pair_file))
        #     self.d_trainset_useritempair = json.load(f)
        # Load testset user-item cdd sents with proxy
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
        # Load testset sentence id to feature tf-value dict
        with open(testset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(testset_sentid2featuretf_file))
            self.d_testset_sentid2featuretf = json.load(f)
        # Load trainset sentence id to feature tf-idf value dict
        with open(trainset_sentid2featuretfidf_file, 'r') as f:
            print("Load file: {}".format(trainset_sentid2featuretfidf_file))
            self.d_trainset_sentid2featuretfidf = json.load(f)
        # Load trainset user to sentid dict
        with open(trainset_user2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_user2sentid_file))
            self.d_trainset_user2sentid = json.load(f)
        # Load trainset item to sentid dict
        with open(trainset_item2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_item2sentid_file))
            self.d_trainset_item2sentid = json.load(f)

        print("Total number of feature: {}".format(len(self.d_id2feature)))
        global total_feature_num
        total_feature_num = len(self.d_id2feature)

        # Get the sid2featuretf dict (on Valid/Test Set)
        self.d_testset_sid2featuretf = self.get_sid2featuretf_eval(
            self.d_testset_sentid2featuretf, self.m_sent2sid, self.m_train_sent_num)
        # Get the sid2feature dict (on Train Set)
        self.d_trainset_sid2feature = self.get_sid2feature_train(
            self.d_trainset_sentid2featuretfidf, self.m_sent2sid)

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
        print("Start Eval ...")
        # Feature P/R/F/AUC
        f_recall_list = []
        f_precision_list = []
        f_F1_list = []
        f_auc_list = []
        # Popular feature P/R/F/AUC
        f_pop_recall_list = []
        f_pop_precision_list = []
        f_pop_F1_list = []
        f_pop_auc_list = []
        # Selected sentences' feature P/R/F/AUC
        f_sent_recall_list = []
        f_sent_precision_list = []
        f_sent_F1_list = []
        f_sent_auc_list = []

        # average features in proxy/ground-truth
        proxy_feature_num_cnt = []
        gt_feature_num_cnt = []

        s_topk = self.m_sentence_topk       # this is used for predict topk sentences
        s_topk_candidate = 20               # this is used for sanity check for the top/bottom topk sentneces

        cnt_useritem_batch = 0
        save_logging_cnt = 0
        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))

            for graph_batch in eval_data:
                if cnt_useritem_batch % 100 == 0:
                    print("... eval ... ", cnt_useritem_batch)

                graph_batch = graph_batch.to(self.m_device)

                # logits: batch_size*max_sen_num
                s_logits, sids, s_masks, target_sids, f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)

                """ Get the topk predicted sentences
                """
                if use_trigram or popular_features_vs_trigram or predict_features_vs_trigram:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate
                    )
                else:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.origin_blocking_sent_prediction(
                        s_logits, sids, s_masks, topk=s_topk, topk_cdd=s_topk_candidate
                    )

                # Get the user/item id of this current graph batch.
                # NOTE: They are not the 'real' user/item id in the dataset, still need to be mapped back.
                # userid = graph_batch.u_rawid
                # itemid = graph_batch.i_rawid

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
                    # pass
                    break

                # Loop through the batch
                for j in range(batch_size):
                    # # Get the user/item id of this graph
                    # userid_j = userid[j].item()
                    # itemid_j = itemid[j].item()
                    # # get the true user/item id
                    # true_userid_j = self.m_uid2user[userid_j]
                    # true_itemid_j = self.m_iid2item[itemid_j]

                    refs_j_list = []
                    hyps_j_list = []
                    hyps_featureid_j_list = []
                    for sid_k in target_sids[j]:
                        refs_j_list.append(self.m_sid2swords[sid_k.item()])
                    for sid_k in s_pred_sids[j]:
                        hyps_j_list.append(self.m_sid2swords[sid_k.item()])
                        hyps_featureid_j_list.extend(self.d_trainset_sid2feature[sid_k.item()])
                    hyps_num_unique_features = len(set(hyps_featureid_j_list))
                    
                    # # Get sid's user/item-side source
                    # user_item_side_source = self.get_sid_user_item_source(
                    #     s_pred_sids[j], true_userid_j, true_itemid_j)

                    # Get the hyps/refs/proxy sentences content
                    hyps_j = " ".join(hyps_j_list)
                    refs_j = " ".join(refs_j_list)
                    # proxy sids can be retrieved from the graph
                    # proxy_j_list = []
                    # for sid_k in self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][-1]:
                    #     proxy_j_list.append(self.d_trainset_id2sent[sid_k])
                    # proxy_j = " ".join(proxy_j_list)

                    # get feature prediction performance
                    # f_logits, fids, f_masks, target_f_labels
                    f_logits_j = f_logits[j]
                    fid_j = fids[j].cpu()
                    mask_f_j = f_masks[j].cpu()
                    target_f_labels_j = target_f_labels[j].cpu()

                    # get the user-item featuretf
                    useritem_to_featuretf = None
                    useritem_popular_features = None
                    ui_popular_features_freq = None
                    # user_to_featuretf = self.d_trainset_user2featuretf[true_userid_j]
                    # item_to_featuretf = self.d_trainset_item2featuretf[true_itemid_j]
                    # useritem_to_featuretf = self.combine_featuretf(user_to_featuretf, item_to_featuretf)

                    """ Get the popular features (and the corresponding tf-value) for this user-item pair
                        The number of popular features has severla options:
                        1. average number of ground-truth sentences' unique features across the test set
                        2. average number of proxy sentences' unique features across the test set
                        3. the top-predicted sentences' unique features of this ui-pair
                        4. the top-predicted (after 3-gram block) sentences' unique features of this ui-pair
                    """
                    if use_ground_truth:
                        if popular_features:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=self.m_feature_topk)
                        elif popular_features_vs_origin:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        elif popular_features_vs_trigram:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        else:
                            pass
                    else:
                        if popular_features:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=avg_proxy_feature_num)
                        elif popular_features_vs_origin:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        elif popular_features_vs_trigram:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        else:
                            pass

                    f_num_j = target_f_labels_j.size(0)
                    mask_f_logits_j = f_logits_j[:f_num_j].cpu()
                    mask_fid_j = fid_j[:f_num_j]
                    mask_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in mask_fid_j]

                    # if save_feature_logits and batch_save_flag:
                    #     self.feature_logits_save_file(true_userid_j, true_itemid_j, mask_f_logits_j)

                    # target is generated from the proxy. These features are unique features (duplications removed)
                    # get the index of the feature labels (feature labels are 1)
                    target_fid_index_j = (target_f_labels_j.squeeze() == 1).nonzero(as_tuple=True)[0]
                    # get the fid of the feature labels
                    target_fid_j = torch.gather(fid_j, dim=0, index=target_fid_index_j)
                    # get the featureid of the feature labels
                    target_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in target_fid_j]
                    # get the feature word of the feature labels
                    target_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in target_featureid_j]

                    # gt is generated from the ground-truth review. These features are unique features (duplications removed)
                    gt_featureid_j, _ = self.get_gt_review_featuretf(self.d_testset_sid2featuretf, target_sids[j])
                    # gt_featureid_j, _ = self.get_gt_review_featuretf_ui(true_userid_j, true_itemid_j)
                    # get the feature word of the gt feature labels
                    gt_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in gt_featureid_j]

                    # number of feaures in proxy/gt
                    proxy_feature_num_cnt.append(len(target_featureword_j))
                    gt_feature_num_cnt.append(len(gt_featureword_j))

                    # TODO: Check the percent of features that appear in the gt reivew but not the cdd reviews
                    # for fea_id in gt_featureid_j:
                    #     if fea_id not in useritem_to_featuretf.keys():
                    #         print("User: {0}\tItem: {1}\tFeature id: {2}".format(
                    #             true_userid_j, true_itemid_j, fea_id
                    #         ))

                    top_pred_featureid_j = None
                    topk_preds_features_logits = None
                    f_prec_j_pop, f_recall_j_pop, f_f1_j_pop, f_auc_j_pop = 0, 0, 0, 0
                    f_prec_j_sent, f_recall_j_sent, f_f1_j_sent, f_auc_j_sent = 0, 0, 0, 0
                    f_prec_j, f_recall_j, f_f1_j, f_auc_j = 0, 0, 0, 0

                    if use_ground_truth:
                        if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            # P/R/F1/AUC of the popular features vs. gt features
                            f_prec_j_pop, f_recall_j_pop, f_f1_j_pop, f_auc_j_pop = get_recall_precision_f1_popular(
                                useritem_popular_features, gt_featureid_j, useritem_to_featuretf, total_feature_num)
                        if use_origin or use_trigram:
                            # P/R/F1 of the selected sentences' features vs. gt features. AUC is meaningless here.
                            f_prec_j_sent, f_recall_j_sent, f_f1_j_sent, f_auc_j_sent = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, gt_featureid_j, total_feature_num)
                        if random_features:
                            # P/R/F1/AUC of the random features vs. gt features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j = get_recall_precision_f1_gt_random(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                self.m_feature_topk, total_feature_num)
                        elif predict_features_vs_origin or predict_features_vs_trigram:
                            # P/R/F1/AUC of the predicted features (dynamic topk) vs. gt features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j, topk_preds_features_logits = get_recall_precision_f1_gt(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                hyps_num_unique_features, total_feature_num)
                        else:
                            # P/R/F1/AUC of the predicted features vs. gt features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j, topk_preds_features_logits = get_recall_precision_f1_gt(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                self.m_feature_topk, total_feature_num)
                    else:
                        if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            f_prec_j_pop, f_recall_j_pop, f_f1_j_pop, f_auc_j_pop = get_recall_precision_f1_popular(
                                useritem_popular_features, target_featureid_j, useritem_to_featuretf, total_feature_num)
                        if use_origin or use_trigram:
                            # P/R/F1 of the selected sentences' features vs. proxy features. AUC is meaningless here.
                            f_prec_j_sent, f_recall_j_sent, f_f1_j_sent, f_auc_j_sent = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, target_featureid_j, total_feature_num)
                        if random_features:
                            # P/R/F1/AUC of the random features vs. proxy features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j = get_recall_precision_f1_random(
                                mask_f_logits_j, target_f_labels_j, avg_proxy_feature_num)
                        elif predict_features_vs_origin or predict_features_vs_trigram:
                            # P/R/F1/AUC of the predicted features (dynamic topk) vs. proxy features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j = get_recall_precision_f1(
                                mask_f_logits_j, target_f_labels_j, hyps_num_unique_features)
                        else:
                            # P/R/F1/AUC of the predicted features vs. proxy features
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j = get_recall_precision_f1(
                                mask_f_logits_j, target_f_labels_j, avg_proxy_feature_num)

                    # Add predicted (multi-task/random) features metrics
                    f_precision_list.append(f_prec_j)
                    f_recall_list.append(f_recall_j)
                    f_F1_list.append(f_f1_j)
                    f_auc_list.append(f_auc_j)
                    # Add popular features metrics
                    f_pop_precision_list.append(f_prec_j_pop)
                    f_pop_recall_list.append(f_recall_j_pop)
                    f_pop_F1_list.append(f_f1_j_pop)
                    f_pop_auc_list.append(f_auc_j_pop)
                    # Add selected sentence's feature metrics
                    f_sent_precision_list.append(f_prec_j_sent)
                    f_sent_recall_list.append(f_recall_j_sent)
                    f_sent_F1_list.append(f_f1_j_sent)
                    f_sent_auc_list.append(f_auc_j_sent)

                    # # Save feature results into file
                    # if save_feature_selected and batch_save_flag:
                    #     self.features_result_save_file(
                    #         proxy_featureids=target_featureid_j,
                    #         gt_featureids=gt_featureid_j,
                    #         hyps_featureids=hyps_featureid_j_list,
                    #         popular_featureids=useritem_popular_features,
                    #         top_predict_featureids=top_pred_featureid_j,
                    #         user_id=true_userid_j,
                    #         item_id=true_itemid_j,
                    #         ref_sents=refs_j,
                    #         hyps_sents=hyps_j,
                    #         proxy_sents=proxy_j,
                    #         hyps_sents_list=hyps_j_list,
                    #         sids_user_item_source=user_item_side_source,
                    #         f_precision=f_prec_j,
                    #         f_recall=f_recall_j,
                    #         f_f1=f_f1_j,
                    #         f_auc=f_auc_j,
                    #         f_precision_pop=f_prec_j_pop,
                    #         f_recall_pop=f_recall_j_pop,
                    #         f_f1_pop=f_f1_j_pop,
                    #         f_auc_pop=f_auc_j_pop,
                    #         f_precision_sent=f_prec_j_sent,
                    #         f_recall_sent=f_recall_j_sent,
                    #         f_f1_sent=f_f1_j_sent,
                    #         s_top_logits=s_topk_logits[j],
                    #         f_top_logits=topk_preds_features_logits
                    #     )
        # metrics of predicted features (sentences'/multi-task/random)
        self.m_mean_f_precision = np.mean(f_precision_list)
        self.m_mean_f_recall = np.mean(f_recall_list)
        self.m_mean_f_f1 = np.mean(f_F1_list)
        self.m_mean_f_auc = np.mean(f_auc_list)
        # metrics of popular features
        self.m_mean_f_precision_pop = np.mean(f_pop_precision_list)
        self.m_mean_f_recall_pop = np.mean(f_pop_recall_list)
        self.m_mean_f_f1_pop = np.mean(f_pop_F1_list)
        self.m_mean_f_auc_pop = np.mean(f_pop_auc_list)
        # metrics of the selected sentences' features
        self.m_mean_f_prec_hyps = np.mean(f_sent_precision_list)
        self.m_mean_f_recall_hyps = np.mean(f_sent_recall_list)
        self.m_mean_f_f1_hyps = np.mean(f_sent_F1_list)
        # number of features (proxy/gt)
        self.m_mean_proxy_feature = np.mean(proxy_feature_num_cnt)
        self.m_mean_gt_feature = np.mean(gt_feature_num_cnt)

        print("Totally {0} batches ({1} data instances).\nAmong them, {2} batches are saved into logging files.".format(
            len(eval_data), len(f_precision_list), save_logging_cnt
        ))
        print("Average features num in proxy: %.4f. Average features num in ground-truth: %.4f" % (
            self.m_mean_proxy_feature, self.m_mean_gt_feature
        ))
        print("feature prediction (predict features), precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
            self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc
        ))
        print("feature prediction (popular features), precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
            self.m_mean_f_precision_pop, self.m_mean_f_recall_pop, self.m_mean_f_f1_pop, self.m_mean_f_auc_pop
        ))
        print("feature prediction (select sents features), precision: %.4f, recall: %.4f, F1: %.4f" % (
            self.m_mean_f_prec_hyps, self.m_mean_f_recall_hyps, self.m_mean_f_f1_hyps
        ))

        output_dir = os.path.join(self.m_eval_output_path, self.m_model_file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if predict_features:
            metric_log_file = os.path.join(
                output_dir,
                'eval_metrics_{0}_{1}_f_topk{2}_s_topk{3}.txt'.format(
                    self.m_dataset_name, label_format, self.m_feature_topk, self.m_sentence_topk))
        else:
            metric_log_file = os.path.join(
                output_dir,
                'eval_metrics_{0}_{1}.txt'.format(
                    self.m_dataset_name, label_format))
        print("writing evaluation results to: {}".format(metric_log_file))
        with open(metric_log_file, 'w') as f:
            print("Totally {0} batches ({1} data instances).\nAmong them, {2} batches are saved into logging files.".format(
                len(eval_data), len(f_precision_list), save_logging_cnt
            ), file=f)
            print("Average features num in proxy: %.4f. Average features num in ground-truth: %.4f" % (
                self.m_mean_proxy_feature, self.m_mean_gt_feature
            ), file=f)
            print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
                self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc
            ), file=f)
            print("feature prediction (popular features), precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
                self.m_mean_f_precision_pop, self.m_mean_f_recall_pop, self.m_mean_f_f1_pop, self.m_mean_f_auc_pop
            ), file=f)
            print("feature prediction (select sents features), precision: %.4f, recall: %.4f, F1: %.4f" % (
                self.m_mean_f_prec_hyps, self.m_mean_f_recall_hyps, self.m_mean_f_f1_hyps
            ), file=f)

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

    def get_popular_features(self, useritem_featuretf, topk=26):
        """ Get the popular features (id) based on the feature tf value
        return: topk_popular_features: topk popular feature's featureid, list
                topk_popular_features_freq: topk popular feature's frequency (i.e. tf value), list
        """

        sorted_useritem_featuretf = dict(sorted(useritem_featuretf.items(), key=lambda item: item[1], reverse=True))

        topk_popular_features = []
        topk_popular_features_freq = []
        cnt_features = 0
        for key, value in sorted_useritem_featuretf.items():
            topk_popular_features.append(key)   # key is featureid
            topk_popular_features_freq.append(value)  # value is the frequency
            cnt_features += 1
            if cnt_features == topk:
                break
        assert len(topk_popular_features) <= topk

        return topk_popular_features, topk_popular_features_freq

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

    def get_sid2feature_train(self, trainset_sentid2featuretfidf, sent2sid):
        trainset_sid2feature = dict()
        for key, value in trainset_sentid2featuretfidf.items():
            assert isinstance(key, str)     # key is the sentid
            sid = sent2sid[key]
            assert sid not in trainset_sid2feature
            trainset_sid2feature[sid] = list(value.keys())
        return trainset_sid2feature

    def get_gt_review_featuretf(self, testset_sid2featuretf, gt_sids):
        """ Get the featureid list and featuretf dict for a list of ground-truth sids
        """
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sid in gt_sids:
            cur_sid_featuretf = testset_sid2featuretf[gt_sid.item()]
            for key, value in cur_sid_featuretf.items():
                gt_featureid_set.add(key)
                if key not in gt_featuretf_dict:
                    gt_featuretf_dict[key] = value
                else:
                    gt_featuretf_dict[key] += value
        return list(gt_featureid_set), gt_featuretf_dict

    def get_gt_review_featuretf_ui(self, true_userid, true_itemid):
        """ Get the featureid list and featuretf dict based on a query of userid and itemid
        """
        # Get the gt sentence ids
        gt_sentids = []
        for sentid in self.d_testset_useritem_cdd_withproxy[true_userid][true_itemid][-2]:
            gt_sentids.append(sentid)
        # Get the feature tf of the sentence ids
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sentid in gt_sentids:
            cur_sentid_featuretf = self.d_testset_sentid2featuretf[gt_sentid]
            for featureid, tf_value in cur_sentid_featuretf.items():
                gt_featureid_set.add(featureid)
                if featureid not in gt_featuretf_dict:
                    gt_featuretf_dict[featureid] = tf_value
                else:
                    gt_featuretf_dict[featureid] += tf_value
        return list(gt_featureid_set), gt_featuretf_dict

    def get_sid_user_item_source(self, pred_sids, user_id, item_id):
        """ Given the predicted/selected sids, find each sid's source, i.e. user-side or item-side sentence.
        :param: pred_sids:  predicted sids, tensor
        :param: user_id:    userid on the dataset, str
        :param: item_id:    itemid on the dataset, str
        return: user_item_source: the user/item side of the sids, a list
        """
        user_item_source = []
        for sid in pred_sids:
            sid_i = sid.item()
            ''' mapping back this sid to sentid (used in the dataset)
            since this is on trainset, we don't need minus the number of sentences
            in the trianset to get the true sentid
            '''
            sentid_i = self.m_sid2sentid[sid_i]
            # check whether this sentid occurs in the user-side or item-side
            if sentid_i in self.d_trainset_user2sentid[user_id]:
                if sentid_i in self.d_trainset_item2sentid[item_id]:
                    user_item_source.append("both user and item side")
                else:
                    user_item_source.append("user side")
            else:
                if sentid_i in self.d_trainset_item2sentid[item_id]:
                    user_item_source.append("item side")
                else:
                    raise Exception("Error: User:{0}\tItem:{1}\tSentid:{2} NOT ON USER AND ITEM SIDE!".format(
                        user_id, item_id, sentid_i
                    ))
        return user_item_source

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
        for i in range(len(sents)):
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
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
                except KeyError:
                    print("i: {0} \t idx: {1} has key error!".format(batch_idx, idx))
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
        batch_select_proba = torch.tensor(batch_select_proba)
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
            assert len(sids[i]) == len(sids[0])
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        sids = sids.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=3
        )
        pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
        topk_logits = ngram_block_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk_cdd
        )
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def bleu_filtering_sent_prediction(self):
        raise Exception("Bleu filtering sentence selection not implemented!")

    def feature_logits_save_file(self, true_userid_j, true_itemid_j, mask_f_logits_j):
        feature_logits_file = os.path.join(
            self.m_eval_output_path,
            'feature_logits_{0}_{1}.txt'.format(dataset_name, label_format)
        )
        with open(feature_logits_file, 'a') as f_l:
            f_l.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
            sorted_mask_f_logits_j, _ = torch.topk(mask_f_logits_j, mask_f_logits_j.size(0))
            for logit in sorted_mask_f_logits_j:
                f_l.write("%.4f" % logit.item())
                f_l.write(", ")
            f_l.write("\n")

    def features_result_save_file(
        self, proxy_featureids, gt_featureids, hyps_featureids, popular_featureids, top_predict_featureids,
            user_id, item_id, ref_sents, hyps_sents, proxy_sents, hyps_sents_list, sids_user_item_source,
            f_precision, f_recall, f_f1, f_auc, f_precision_pop, f_recall_pop, f_f1_pop, f_auc_pop,
            f_precision_sent, f_recall_sent, f_f1_sent, s_top_logits, f_top_logits):
        """ Write the results into file.
        :param: proxy_featureids: proxy's featureids, list
        :param: gt_featureids: ground-truth's featureids, list
        :param: hyps_featureids: hypothesis/predict sentences' featureids, list
        :param: popular_featureids: popular features' featureids, list
        :param: top_predict_featureids: top predicted featureids, list
        :param: user_id: true user id, str
        :param: item_id: true item id, str
        :param: ref_sents: reference sentences (text, not id), str
        :param: hyps_sents: hypothesis/predicted sentences (text, not id), str
        :param: proxy_sents: proxy sentences (text, not id), str
        :param: hyps_sents_list: hypothesis/predicted sentences (text, not id) split into sentences, list
        :param: sids_user_item_source: sid's source, i.e. user-side or item-side sentence, list
        :param: s_top_logits: the logits of the top-select sentences, tensor
        :param: f_top_logits: the features of the top-predicted features, tensor
        """

        hyps_sent_print_info = True
        predict_features_print_logits = True

        features_result_file_path = os.path.join(
            self.m_eval_output_path, 'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))

        proxy_featureids_set = set(proxy_featureids)
        gt_featureids_set = set(gt_featureids)
        hyps_featureids_set = set(hyps_featureids)
        if popular_featureids is not None:
            popular_featureids_set = set(popular_featureids)
            assert len(popular_featureids) == len(popular_featureids_set)
        else:
            popular_featureids_set = None
        if top_predict_featureids is not None:
            top_predict_featureids_set = set(top_predict_featureids)
            assert len(top_predict_featureids) == len(top_predict_featureids_set)
        else:
            top_predict_featureids_set = None

        assert s_top_logits.size(0) == len(hyps_sents_list)
        if f_top_logits is not None:
            assert f_top_logits.size(0) == len(top_predict_featureids)

        proxy_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in proxy_featureids_set]
        gt_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in gt_featureids_set]
        hyps_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in hyps_featureids_set]
        # NOTE: We need to reserve the rankings in the popular features and top-predict features, thus
        # we can't use the set of the corresponind featureids. These 2 original list of featureids should
        # not have duplications based on the way we compute them.
        if popular_featureids is not None:
            popular_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in popular_featureids]
        else:
            popular_featurewords = None
        if top_predict_featureids is not None:
            top_predict_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in top_predict_featureids]
        else:
            top_predict_featurewords = None

        overlap_featureids_set_pop = None
        overlap_featureids_set_sent = None
        overlap_featureids_set_pred = None
        if use_ground_truth:
            # 1. Use the popular features
            if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                overlap_featureids_set_pop = gt_featureids_set.intersection(popular_featureids_set)
            # 2. use the features from the predicted sentences.
            if use_origin or use_trigram:
                overlap_featureids_set_sent = gt_featureids_set.intersection(hyps_featureids_set)
            # 3. Use the predicted features by the multi-task model or random features
            if top_predict_featureids_set is not None:
                overlap_featureids_set_pred = gt_featureids_set.intersection(top_predict_featureids_set)
        else:
            # 1. Use the popular features
            if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                overlap_featureids_set_pop = proxy_featureids_set.intersection(popular_featureids_set)
            # 2. Use the features from the predicted sentences.
            # Options: use_origin/use_trigram
            if use_origin or use_trigram:
                overlap_featureids_set_sent = proxy_featureids_set.intersection(hyps_featureids_set)
            # 3. Use the predicted features by the multi-task model or random features
            if top_predict_featureids_set is not None:
                overlap_featureids_set_pred = proxy_featureids_set.intersection(top_predict_featureids_set)
        # Convert overlap featureid to feature words
        if overlap_featureids_set_pop is not None:
            overlap_featurewords_pop = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureids_set_pop]
        else:
            overlap_featurewords_pop = None
        if overlap_featureids_set_sent is not None:
            overlap_featurewords_sent = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureids_set_sent]
        else:
            overlap_featurewords_sent = None
        if overlap_featureids_set_pred is not None:
            overlap_featurewords_pred = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureids_set_pred]
        else:
            overlap_featurewords_pred = None

        # Combine sentence with its logits score and source (i.e. user/item side)
        if hyps_sent_print_info:
            hyps_sents_with_info = []
            for i in range(len(hyps_sents_list)):
                hyps_sent_logits = s_top_logits[i].item()
                hyps_sent_source = sids_user_item_source[i]
                hyps_sent_info = "({:.4}, {})".format(hyps_sent_logits, hyps_sent_source)
                hyps_sents_with_info.append(hyps_sents_list[i]+" "+hyps_sent_info)

        # Combine features with its logits score
        if predict_features_print_logits:
            top_pred_features_with_logits = []
            for i in range(len(top_predict_featurewords)):
                feature_logits = f_top_logits[i].item()
                feature_info = "({:.4})".format(feature_logits)
                top_pred_features_with_logits.append(top_predict_featurewords[i]+" "+feature_info)

        if use_ground_truth:
            # write file
            with open(features_result_file_path, 'a') as f:
                # write something into the file
                f.write("User: {0}\tItem: {1}\n".format(user_id, item_id))
                f.write("refs: {}\n".format(ref_sents))
                if hyps_sent_print_info:
                    f.write("hyps: {}\n".format(" ".join(hyps_sents_with_info)))
                else:
                    f.write("hyps: {}\n".format(hyps_sents))
                f.write("ground-truth features: {}\n".format("["+", ".join(gt_featurewords)+"]"))
                f.write("predict sentences' features: {}\n".format("["+", ".join(hyps_featurewords)+"]"))
                if use_origin or use_trigram:
                    f.write("overlappings (select sentences vs. gt features): {}\n".format(
                        "["+", ".join(overlap_featurewords_sent)+"]"))
                if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                    f.write("popular features: {}\n".format(
                        "["+", ".join(popular_featurewords)+"]"))
                    f.write("overlappings (popular vs. gt features): {}\n".format(
                        "["+", ".join(overlap_featurewords_pop)+"]"))
                if random_features:
                    f.write("random features: {}\n".format(
                        "["+", ".join(top_predict_featurewords)+"]"))
                    f.write("overlappings (random vs. gt features): {}\n".format(
                        "["+", ".join(overlap_featurewords_pred)+"]"))
                else:
                    if predict_features_print_logits:
                        f.write("top predict features: {}\n".format(
                            "["+", ".join(top_pred_features_with_logits)+"]"))
                    else:
                        f.write("top predict features: {}\n".format(
                            "["+", ".join(top_predict_featurewords)+"]"))
                    f.write("overlappings (top-predict vs. gt features): {}\n".format(
                        "["+", ".join(overlap_featurewords_pred)+"]"))
                f.write("Number of unique features in ground-truth: {}\n".format(len(gt_featurewords)))
                f.write("Number of unique features in predict sentences: {}\n".format(len(hyps_featurewords)))
                if use_origin or use_trigram:
                    f.write("Number of feature overlap (select sentences vs. gt features): {}\n".format(len(overlap_featurewords_sent)))
                if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                    f.write("Number of unique popular features: {}\n".format(len(popular_featurewords)))
                    f.write("Number of feature overlap (popular vs. gt features): {}\n".format(len(overlap_featurewords_pop)))
                if random_features:
                    f.write("Number of unique random features: {}\n".format(len(top_predict_featurewords)))
                    f.write("Number of feature overlap (random vs. gt features): {}\n".format(len(overlap_featurewords_pred)))
                else:
                    f.write("Number of top-predict features: {}\n".format(len(top_predict_featurewords)))
                    f.write("Number of feature overlap (top-predict vs. gt features): {}\n".format(len(overlap_featurewords_pred)))
                f.write("Top-predict features vs. gt features. Precision: %.4f\tRecall: %.4f\tF1: %.4f\tAUC: %.4f\n" % (
                    f_precision, f_recall, f_f1, f_auc))
                f.write("Popular features vs. gt features. Precision: %.4f\tRecall: %.4f\tF1: %.4f\tAUC: %.4f\n" % (
                    f_precision_pop, f_recall_pop, f_f1_pop, f_auc_pop))
                f.write("Features in the hyps vs. gt features. Precision: %.4f\tRecall: %.4f\tF1: %.4f\n" % (
                    f_precision_sent, f_recall_sent, f_f1_sent
                ))
                f.write("==------==------==------==------==------==------==\n")

        else:
            # write file
            with open(features_result_file_path, 'a') as f:
                # write something into the file
                f.write("User: {0}\tItem: {1}\n".format(user_id, item_id))
                f.write("refs: {}\n".format(ref_sents))
                f.write("hyps: {}\n".format(hyps_sents))
                f.write("proxy: {}\n".format(proxy_sents))
                f.write("proxy features: {}\n".format(", ".join(proxy_featurewords)))
                if use_origin or use_trigram:
                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_featurewords)))
                    f.write("overlappings (select sentences vs. gt features): {}\n".format(
                        ", ".join(overlap_featurewords_sent)))
                if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                    f.write("popular features: {}\n".format(", ".join(popular_featurewords)))
                    f.write("overlappings (popular vs. proxy features): {}\n".format(
                        ", ".join(overlap_featurewords_pop)))
                if top_predict_featurewords is not None:
                    f.write("top predict features: {}\n".format(", ".join(top_predict_featurewords)))
                    f.write("overlappings (top-predict vs. proxy features): {}\n".format(
                        ", ".join(overlap_featurewords_pred)))
                f.write("Number of unique features in proxy: {}\n".format(len(proxy_featurewords)))
                if use_origin or use_trigram:
                    f.write("Number of unique features in predict sentences: {}\n".format(len(hyps_featurewords)))
                    f.write("Number of feature overlap (select sentences vs. gt features): {}\n".format(len(overlap_featurewords_sent)))
                if popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                    f.write("Number of unique popular features: {}\n".format(len(popular_featurewords)))
                    f.write("Number of feature overlap (popular vs. proxy features): {}\n".format(len(overlap_featurewords_pop)))
                if top_predict_featurewords is not None:
                    f.write("Number of top-predict features: {}\n".format(len(top_predict_featurewords)))
                    f.write("Number of feature overlap (top-predict vs. proxy features): {}\n".format(len(overlap_featurewords_pred)))
                f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_precision, f_recall, f_f1, f_auc))
                f.write("==------==------==------==------==------==------==\n")
