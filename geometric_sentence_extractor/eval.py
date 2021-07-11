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
import pickle
import random

dataset_name = 'medium_500_pure'
label_format = 'soft_label'
use_blocking = False        # whether using 3-gram blocking or not
use_filtering = False       # whether using bleu score based filtering or not
save_predict = False
random_sampling = False
random_features = False
get_statistics = False
save_sentence_selected = False
save_feature_selected = False
bleu_filter_value = 0.25

# Baselines
use_majority_vote_popularity = False
use_majority_vote_feature_score = False

save_hyps_refs = True
compute_rouge_score = True
compute_bleu_score = True
# Save feature hidden embeddings (after forward through the GAT model)
save_train_feature_hidden = False
save_test_feature_hidden = False
percentage_train_data_saved = 0.1

MAX_batch_output = 2000
# S_TOPK = 5


class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
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
        if use_majority_vote_popularity:
            print("hypothesis selected based on feature popularity.")
        elif use_majority_vote_feature_score:
            print("hypothesis selected based on feature predicted scores.")
        else:
            print("hypothesis selected based on original score and filtering methods.")

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

        # Get the sid2featuretf dict (on Valid/Test Set)
        self.d_testset_sid2featuretf = self.get_sid2featuretf_eval(
            self.d_testset_sentid2featuretf, self.m_sent2sid, self.m_train_sent_num)
        # Get the sid2feature dict (on Train Set)
        self.d_trainset_sid2feature = self.get_sid2feature_train(
            self.d_trainset_sentid2featuretfidf, self.m_sent2sid)

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
        """ TODO:
        1. Save Predict/Selected sentences and Reference sentences to compute BLEU using the perl script.
        2. Add mojority vote based baselines.
        3. Seperate code chunks into functions.
        """
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
        s_topk_candidate = 20

        # already got feature2fid mapping, need the reverse
        self.m_fid2feature = {value: key for key, value in self.m_feature2fid.items()}
        # print(self.m_feture2fid)

        cnt_useritem_pair = 0
        cnt_useritem_batch = 0
        # train_test_overlap_cnt = 0
        # train_test_differ_cnt = 0
        save_logging_cnt = 0
        self.m_network.eval()
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
                s_logits, sids, s_masks, target_sids, f_logits, fids, f_masks, target_f_labels, hidden_f_batch = self.m_network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)

                # Save the predict logits and sids
                # if save_predict:
                #     userid_batch = graph_batch.u_rawid
                #     itemid_batch = graph_batch.i_rawid
                #     for i in range(batch_size):
                #         current_result_dict = {}
                #         current_result_dict['user_id'] = userid_batch[i].item()
                #         current_result_dict['item_id'] = itemid_batch[i].item()
                #         assert len(s_logits[i]) == len(sids[i])
                #         assert len(s_logits[i]) == len(s_masks[i])
                #         triple_data_list = []
                #         for pos in range(len(s_logits[i])):
                #             triple_data_list.append(
                #                 [s_logits[i][pos].item(), sids[i][pos].item(), s_masks[i][pos].item()])
                #         current_result_dict['predict_data'] = triple_data_list
                #         current_target_sent_sids = []
                #         for this_sid in target_sids[i]:
                #             current_target_sent_sids.append(this_sid.item())
                #         current_result_dict['target'] = current_target_sent_sids

                #         # save current_result_dict into json file
                #         model_ckpt_name = self.m_model_file.split('.')[0]
                #         this_json_file = os.path.join(self.this_DIR, 'result_{}.json'.format(model_ckpt_name))
                #         with open(this_json_file, 'a') as f:
                #             json.dump(current_result_dict, f)
                #             f.write("\n")
                #     continue

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
                        num_sent = int(sum(s_masks[i]).item())
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

                        if save_hyps_refs:
                            # Save refs and selected hyps into file for later ROUGE/BLEU computation
                            refs_file = os.path.join(self.m_eval_output_path, 'reference.txt')
                            hyps_file = os.path.join(self.m_eval_output_path, 'hypothesis.txt')
                            with open(refs_file, 'a') as f_ref:
                                f_ref.write(refs_j)
                                f_ref.write("\n")
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_j)
                                f_hyp.write("\n")

                        if compute_rouge_score:
                            scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)
                            # ROUGE-1
                            rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                            rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                            rouge_1_p_list.append(scores_j["rouge-1"]["p"])
                            # ROUGE-2
                            rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                            rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                            rouge_2_p_list.append(scores_j["rouge-2"]["p"])
                            # ROUGE-L
                            rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                            rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                            rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                        if compute_bleu_score:
                            bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                            bleu_list.append(bleu_scores_j)
                            # NLTK BLEU
                            bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())
                            bleu_1_list.append(bleu_1_scores_j)
                            bleu_2_list.append(bleu_2_scores_j)
                            bleu_3_list.append(bleu_3_scores_j)
                            bleu_4_list.append(bleu_4_scores_j)

                    cnt_useritem_batch += 1
                    continue

                # elif get_statistics:
                #     for i in range(batch_size):
                #         this_g = graph_batch[i]
                #         labels_feature = this_g.f_label
                #         print("shape of feature labels: {}".format(labels_feature.shape))
                #         num_features_per_target_review.append(torch.sum(labels_feature).item())
                #     continue

                elif use_blocking:
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate
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

                    if save_train_feature_hidden:
                        # Only save 10% of the data from train set
                        if random.random() <= percentage_train_data_saved:
                            # Form the feature hidden f tensor with label
                            f_hidden_train_file = os.path.join(self.m_eval_output_path, 'train_f_hidden.json')
                            with open(f_hidden_train_file, 'a') as f_h:
                                for f_idx in range(f_num_j):
                                    cur_dict = dict()
                                    cur_hidden_f = mask_hidden_f_j[f_idx].detach().numpy()
                                    cur_f_label = target_f_labels_j[f_idx].detach().numpy()
                                    cur_hidden_f_data = np.append(cur_hidden_f, cur_f_label)
                                    cur_hidden_f_data = cur_hidden_f_data.tolist()
                                    cur_dict['ui_pair_index'] = cnt_useritem_pair
                                    cur_dict['f_hidden'] = cur_hidden_f_data
                                    # Save this dict into json file
                                    json.dump(cur_dict, f_h)
                                    f_h.write('\n')
                            train_ui_pair_saved_cnt += 1

                    if save_test_feature_hidden:
                        # Form the feature hidden f tensor with label
                        f_hidden_test_file = os.path.join(self.m_eval_output_path, 'test_f_hidden.json')
                        with open(f_hidden_test_file, 'a') as f_h:
                            # Need the gt-feature id of this user-item pair
                            gt_featureid_j, _ = self.get_gt_review_featuretf(
                                self.d_testset_sid2featuretf, target_sids[j])
                            cur_test_user_item_f_hidden = dict()
                            cur_test_user_item_f_hidden['ui_pair_index'] = cnt_useritem_pair
                            f_hidden_np = []
                            for f_idx in range(f_num_j):
                                cur_f_hidden_np = []
                                cur_f_hidden_np.append(int(mask_featureid_j[f_idx]))
                                cur_f_hidden_np.extend(mask_hidden_f_j[f_idx].detach().numpy().tolist())
                                f_hidden_np.append(cur_f_hidden_np)
                            cur_test_user_item_f_hidden['feature'] = f_hidden_np
                            cur_test_user_item_f_hidden['gt'] = gt_featureid_j
                            cur_test_user_item_f_hidden['topk'] = hyps_num_unique_features
                            # Save this dict into json file
                            json.dump(cur_test_user_item_f_hidden, f_h)
                            f_h.write('\n')
                        test_ui_pair_saved_cnt += 1

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
                        with open(refs_file, 'a') as f_ref:
                            f_ref.write(refs_j)
                            f_ref.write("\n")
                        if use_majority_vote_popularity:
                            cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                            hyps_pop, _, _ = self.majority_vote_popularity(
                                true_userid_j, true_itemid_j, cur_cdd_sents, topk=s_topk)
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_pop)
                                f_hyp.write("\n")
                        elif use_majority_vote_feature_score:
                            cur_cdd_sents = self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][0]
                            hyps_f_score, _, _, _ = self.majority_vote_predicted_feature(
                                true_userid_j, true_itemid_j, cur_cdd_sents, mask_f_logits_j, mask_featureid_j, topk=s_topk)
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_f_score)
                                f_hyp.write("\n")
                        else:
                            with open(hyps_file, 'a') as f_hyp:
                                f_hyp.write(hyps_j)
                                f_hyp.write("\n")

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

        # if len(num_sents_per_target_review) != 0:
        #     self.m_mean_num_sents_per_target_review = np.mean(num_sents_per_target_review)
        #     print("Number of sentences for each target review (on average): {}".format(
        #         self.m_mean_num_sents_per_target_review))

        print("Totally {0} batches ({1} data instances).\nAmong them, {2} batches are saved into logging files.".format(
            len(eval_data), cnt_useritem_pair, save_logging_cnt
        ))
        print("Totally {0} train ui-pairs and the corresponding feature hidden embeddings are saved.".format(
            train_ui_pair_saved_cnt
        ))
        print("Totally {0} test ui-pairs and the corresponding feature hidden embeddings are saved.".format(
            test_ui_pair_saved_cnt
        ))

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
                # print("Total number of user-item on testset (not appear in trainset): {}\n".format(train_test_differ_cnt), file=f)
                # print("Total number of user-item on testset (appear in trainset): {}\n".format(train_test_overlap_cnt), file=f)
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
            assert len(sids[i]) == len(sids[0])
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
