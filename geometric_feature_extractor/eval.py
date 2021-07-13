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
from metric import get_recall_precision_f1_gt_valid
from rouge import Rouge
import dgl
import pickle


class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_dataset_name = args.data_set
        self.m_mean_loss = 0
        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file_name = args.model_file.split('/')[-1].split('.pt')[0]
        self.m_eval_output_path = args.eval_output_path

        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid

        self.m_feature_topk = args.select_topk_f    # default: 15
        print("Number of predicted features selected: {}".format(self.m_feature_topk))

        # self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Load data mappings
        self.f_load_dicts(vocab_obj, args)

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

        f_recall_list, f_precision_list, f_F1_list, f_AUC_list = [], [], [], []
        num_features_per_graph_train = []
        num_features_per_graph_eval = []
        print('--'*10)

        i = 0
        self.m_network.eval()
        with torch.no_grad():
            for graph_batch in train_data:
                if i % 100 == 0:
                    print("... eval(train data) ...", i)
                i += 1
                # Get batch data
                graph_batch = graph_batch.to(self.m_device)
                #### logits: batch_size*max_sen_num
                f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)
                batch_size_f = f_logits.size(0)
                batch_size = graph_batch.num_graphs
                assert batch_size == batch_size_f
                for j in range(batch_size):
                    # get feature prediction performance
                    # f_logits, fids, f_masks, target_f_labels
                    target_f_labels_j = target_f_labels[j].cpu()
                    # get the predicted feature ids and feature logits
                    f_num_j = target_f_labels_j.size(0)
                    num_features_per_graph_train.append(f_num_j)

            i = 0
            print("Number of evaluation data: {}".format(len(eval_data)))
            for graph_batch in eval_data:
                if i % 100 == 0:
                    print("... eval ... ", i)
                i += 1
                # Get batch data
                graph_batch = graph_batch.to(self.m_device)

                #### logits: batch_size*max_sen_num
                f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)

                batch_size_f = f_logits.size(0)
                batch_size = graph_batch.num_graphs
                assert batch_size == batch_size_f

                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                # Loop through this batch
                for j in range(batch_size):
                    # Get the user/item id of this graph
                    userid_j = userid[j].item()
                    itemid_j = itemid[j].item()
                    # get the true user/item id
                    true_userid_j = self.m_uid2user[userid_j]
                    true_itemid_j = self.m_iid2item[itemid_j]
                    # get feature prediction performance
                    # f_logits, fids, f_masks, target_f_labels
                    f_logits_j = f_logits[j].cpu()
                    fid_j = fids[j].cpu()
                    mask_f_j = f_masks[j].cpu()
                    target_f_labels_j = target_f_labels[j].cpu()
                    # get the predicted feature ids and feature logits
                    f_num_j = target_f_labels_j.size(0)
                    mask_f_logits_j = f_logits_j[:f_num_j]
                    mask_fid_j = fid_j[:f_num_j]
                    mask_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in mask_fid_j]
                    num_features_per_graph_eval.append(f_num_j)
                    # get the gt feature ids
                    gt_featureid_j, _ = self.get_gt_review_featuretf_ui(true_userid_j, true_itemid_j)
                    # compute P/R/F1/AUC of the predicted features vs. gt features. topk=15
                    f_prec_j, f_recall_j, f_f1_j, f_auc_j, _, _ = get_recall_precision_f1_gt_valid(
                        mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                        self.m_feature_topk, self.total_feature_num)
                    # Add predicted (multi-task/random) features metrics
                    f_precision_list.append(f_prec_j)
                    f_recall_list.append(f_recall_j)
                    f_F1_list.append(f_f1_j)
                    f_AUC_list.append(f_auc_j)

        self.m_mean_f_precision = np.mean(f_precision_list)
        self.m_mean_f_recall = np.mean(f_recall_list)
        self.m_mean_f_f1 = np.mean(f_F1_list)
        self.m_mean_f_auc = np.mean(f_AUC_list)
        self.m_mean_f_node_num_train = np.mean(num_features_per_graph_train)
        self.m_mean_f_node_num_test = np.mean(num_features_per_graph_eval)

        print(
            "feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
                self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc))
        print("Average number of features, train-set: {0}\ttest-set: {1}".format(
            self.m_mean_f_node_num_train, self.m_mean_f_node_num_test
        ))

        output_dir = os.path.join(self.m_eval_output_path, self.m_model_file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        metric_log_file = os.path.join(
            output_dir,
            'eval_metrics_{0}_f_topk{1}.txt'.format(
                self.m_dataset_name,
                self.m_feature_topk)
        )
        print("writing evaluation result to: {}".format(metric_log_file))
        with open(metric_log_file, 'w') as f:
            f.write(
                "feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f\n" % (
                    self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc)
            )

    def f_load_dicts(self, vocab_obj, args):
        """ Load some useful dicts of data
        """
        self.dataset_name = args.data_set
        self.dataset_dir = args.data_dir
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid

        self.m_train_sent_num = vocab_obj.m_train_sent_num
        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}

        # need to load some mappings
        id2feature_file = os.path.join(self.dataset_dir, 'train/feature/id2feature.json')
        feature2id_file = os.path.join(self.dataset_dir, 'train/feature/feature2id.json')
        # trainset_id2sent_file = os.path.join(self.dataset_dir, 'train/sentence/id2sentence.json')
        # validset_id2sent_file = os.path.join(self.dataset_dir, 'valid/sentence/id2sentence.json')
        testset_useritem_cdd_withproxy_file = os.path.join(self.dataset_dir, 'test/useritem2sentids_withproxy.json')
        # trainset_user2featuretf_file = os.path.join(self.dataset_dir, 'train/user/user2featuretf.json')
        # trainset_item2featuretf_file = os.path.join(self.dataset_dir, 'train/item/item2featuretf.json')
        # trainset_sentid2featuretfidf_file = os.path.join(self.dataset_dir, 'train/sentence/sentence2feature.json')
        testset_sentid2featuretf_file = os.path.join(self.dataset_dir, 'test/sentence/sentence2featuretf.json')
        # trainset_user2sentid_file = os.path.join(self.dataset_dir, 'train/user/user2sentids.json')
        # trainset_item2sentid_file = os.path.join(self.dataset_dir, 'train/item/item2sentids.json')

        # Load dicts
        # Load features
        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)

        # # Load train/valid sentence_id to sentence content
        # with open(trainset_id2sent_file, 'r') as f:
        #     print("Load file: {}".format(trainset_id2sent_file))
        #     self.d_trainset_id2sent = json.load(f)
        # with open(validset_id2sent_file, 'r') as f:
        #     print("Load file: {}".format(validset_id2sent_file))
        #     self.d_validset_id2sent = json.load(f)

        # Load validset user-item cdd sents with proxy
        with open(testset_useritem_cdd_withproxy_file, 'r') as f:
            print("Load file: {}".format(testset_useritem_cdd_withproxy_file))
            self.d_testset_useritem_cdd_withproxy = json.load(f)

        # # Load trainset user to feature tf-value dict
        # with open(trainset_user2featuretf_file, 'r') as f:
        #     print("Load file: {}".format(trainset_user2featuretf_file))
        #     self.d_trainset_user2featuretf = json.load(f)
        # # Load trainset item to feature tf-value dict
        # with open(trainset_item2featuretf_file, 'r') as f:
        #     print("Load file: {}".format(trainset_item2featuretf_file))
        #     self.d_trainset_item2featuretf = json.load(f)

        # Load validset sentence id to feature tf-value dict
        with open(testset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(testset_sentid2featuretf_file))
            self.d_testset_sentid2featuretf = json.load(f)
        # # Load trainset sentence id to feature tf-idf value dict
        # with open(trainset_sentid2featuretfidf_file, 'r') as f:
        #     print("Load file: {}".format(trainset_sentid2featuretfidf_file))
        #     self.d_trainset_sentid2featuretfidf = json.load(f)

        # # Load trainset user to sentid dict
        # with open(trainset_user2sentid_file, 'r') as f:
        #     print("Load file: {}".format(trainset_user2sentid_file))
        #     self.d_trainset_user2sentid = json.load(f)
        # # Load trainset item to sentid dict
        # with open(trainset_item2sentid_file, 'r') as f:
        #     print("Load file: {}".format(trainset_item2sentid_file))
        #     self.d_trainset_item2sentid = json.load(f)

        print("Total number of feature: {}".format(len(self.d_id2feature)))
        self.total_feature_num = len(self.d_id2feature)

        # # Get the sid2featuretf dict (on Valid Set)
        # self.d_testset_sid2featuretf = self.get_sid2featuretf_eval(
        #     self.d_validset_sentid2featuretf, self.m_sent2sid, self.m_train_sent_num)
        # # Get the sid2feature dict (on Train Set)
        # self.d_trainset_sid2feature = self.get_sid2feature_train(
        #     self.d_trainset_sentid2featuretfidf, self.m_sent2sid)

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
