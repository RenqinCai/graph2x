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
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision, get_recall_precision_f1, get_sentence_bleu
from rouge import Rouge
import dgl
import pickle

dataset_name = 'medium_500'


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
        # id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        # feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        # testset_sent2id_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2id.json'.format(dataset_name)
        # testset_sentid2feature_file = '../../Dataset/ratebeer/{}/valid/sentence/sentence2feature.json'.format(dataset_name)
        # with open(id2feature_file, 'r') as f:
        #     print("Load file: {}".format(id2feature_file))
        #     self.d_id2feature = json.load(f)
        # with open(feature2id_file, 'r') as f:
        #     print("Load file: {}".format(feature2id_file))
        #     self.d_feature2id = json.load(f)
        # with open(testset_sent2id_file, 'r') as f:
        #     print("Load file: {}".format(testset_sent2id_file))
        #     self.d_testsetsent2id = json.load(f)
        # with open(testset_sentid2feature_file, 'r') as f:
        #     print("Load file: {}".format(testset_sentid2feature_file))
        #     self.d_testsetsentid2feature = json.load(f)

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

    def f_get_statistics(self, train_data, eval_data):
        f_num = []
        s_num = []
        g_num = 0
        node_num = []

        # for graph_batch in eval_data:
        #     batch_size = graph_batch.num_graphs
        #     g_num += batch_size
        #     for j in range(batch_size):
        #         g = graph_batch[j]
        #         f_num.append(g.f_num)
        #         s_num.append(g.s_num)
        #         node_num.append(g.num_nodes)

        # print("test data graph num", g_num)
        # print("test data graph node num", np.mean(node_num))
        # print("test data feature node num", np.mean(f_num))
        # print("test data sentence node num", np.mean(s_num))

        f_num = []
        s_num = []
        node_num = 0
        g_num = 0

        index = 0

        for graph_batch in train_data:
            if index % 1e2 == 0:
                print(index)
            index += 1
            batch_size = graph_batch.num_graphs
            # print("batch_size", batch_size)
            g_num += batch_size
            batch_fnum = graph_batch.f_num
            f_num.extend(list(batch_fnum.cpu().numpy()))

            batch_snum = graph_batch.s_num
            s_num.extend(list(batch_snum.cpu().numpy()))

            batch_node_num = graph_batch.num_nodes
            node_num += batch_node_num
            # print("batch_node_num", batch_node_num)
            # node_num.extend(list(batch_node_num.cpu().numpy()))

            # for j in range(batch_size):
                # g = graph_batch[j]
                # f_num.append(g.f_num)
                # s_num.append(g.s_num)
                # node_num.append(g.num_nodes)

        print("train data graph num", g_num)
        print("train data graph node num", node_num/g_num)
        # print("train data graph node num", np.mean(node_num))
        print("train data feature node num", np.mean(f_num))
        print("train data sentence node num", np.mean(s_num))



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

        print('--'*10)

        debug_index = 0
        topk = 3
        topk_candidate = 20

        # already got feature2fid mapping, need the reverse
        self.m_fid2feature = {value: key for key, value in self.m_feature2fid.items()}
        # print(self.m_feture2fid)

        i = 0
        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))

            for graph_batch in eval_data:
                
                if i % 100 == 0:
                    print("... eval ... ", i)
                i += 1

                graph_batch = graph_batch.to(self.m_device)

                #### logits: batch_size*max_sen_num
                s_logits, sids, s_masks, target_sids, f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)

                topk_logits, topk_pred_snids = torch.topk(s_logits, topk, dim=1)
                
                #### topk sentence index
                #### pred_sids: batch_size*topk_sent
                pred_sids = sids.gather(dim=1, index=topk_pred_snids)

                batch_size = s_logits.size(0)

                top_cdd_logits, top_cdd_pred_snids = torch.topk(s_logits, topk_candidate, dim=1)
                top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)

                reverse_s_logits = (1-s_logits)*s_masks
                bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_candidate, dim=1)
                bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                for j in range(batch_size):
                    refs_j = []
                    hyps_j = []

                    for sid_k in target_sids[j]:
                        refs_j.append(self.m_sid2swords[sid_k.item()])

                    for sid_k in pred_sids[j]:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j)
                    refs_j = " ".join(refs_j)

                    userid_j = userid[j]
                    itemid_j = itemid[j]

                    with open('../result/eval_logging_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(userid_j))
                        f.write("item id: {}\n".format(itemid_j))
                        f.write("hyps_j: {}\n".format(hyps_j))
                        f.write("refs_j: {}\n".format(refs_j))
                        f.write("========================================\n")

                    top_cdd_hyps_j = []
                    top_cdd_probs_j = top_cdd_logits[j]
                    for sid_k in top_cdd_pred_sids[j]:
                        top_cdd_hyps_j.append(self.m_sid2swords[sid_k.item()])

                    with open('../result/eval_logging_top_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(userid_j))
                        f.write("item id: {}\n".format(userid_j))
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

                    with open('../result/eval_logging_bottom_{}.txt'.format(dataset_name), 'a') as f:
                        f.write("user id: {}\n".format(userid_j))
                        f.write("item id: {}\n".format(userid_j))
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
                    # f_logits, fids, f_masks, target_f_labels
                    f_logits_j = f_logits[j]
                    fid_j = fids[j]
                    mask_f_j = f_masks[j]
                    target_f_labels_j = target_f_labels[j]

                    f_num_j = target_f_labels_j.size(0)
                    mask_f_logits_j = f_logits_j[:f_num_j]
                    
                    f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1(mask_f_logits_j, target_f_labels_j)
                    f_precision_list.append(f_prec_j)
                    f_recall_list.append(f_recall_j)
                    f_F1_list.append(f_f1_j)
                    f_auc_list.append(f_auc_j)


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

        self.m_mean_f_precision = np.mean(f_precision_list)
        self.m_mean_f_recall = np.mean(f_recall_list)
        self.m_mean_f_f1 = np.mean(f_F1_list)
        self.m_mean_f_auc = np.mean(f_auc_list)

        print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f"%(self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc))

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
            # print("feature recall: {}\n".format(self.m_recall_feature), file=f)
            # print("feature precision: {}\n".format(self.m_precision_feature), file=f)
