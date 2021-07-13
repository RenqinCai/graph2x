import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
import torch.nn as nn
from tensorboardX import SummaryWriter
from loss import XE_LOSS, BPR_LOSS, SIG_LOSS
from metric import get_example_recall_precision, get_recall_precision_f1_gt_valid
from model import GraphX
import random
import torch.nn.functional as F
from rouge import Rouge


class TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device

        self.m_sid2swords = vocab_obj.m_sid2swords

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_train_precision = 0
        self.m_mean_train_recall = 0

        self.m_mean_val_loss = 0
        self.m_mean_eval_precision = 0
        self.m_mean_eval_recall = 0

        self.m_mean_eval_bleu = 0
        self.m_feature_topk = 15

        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        # self.m_rec_loss = XE_LOSS(vocab_obj.item_num, self.m_device)
        # self.m_rec_loss = BPR_LOSS(self.m_device)
        self.m_rec_loss = SIG_LOSS(self.m_device)
        self.m_rec_soft_loss = BPR_LOSS(self.m_device)
        # self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_grad_clip = args.grad_clip
        self.m_weight_decay = args.weight_decay
        # self.m_l2_reg = args.l2_reg

        self.m_soft_train = args.soft_label

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval

        # Load data mappings
        self.f_load_dicts(vocab_obj, args)

        print("print_interval", self.m_print_interval)
        self.m_overfit_epoch_threshold = 3

    def f_save_model(self, checkpoint):
        # checkpoint = {'model':network.state_dict(),
        #     'epoch': epoch,
        #     'en_optimizer': en_optimizer,
        #     'de_optimizer': de_optimizer
        # }
        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, valid_data, network, optimizer, logger_obj):
        last_train_loss = 0
        # last_eval_loss = 0
        last_eval_f1 = 0
        # self.m_mean_eval_loss = 0
        self.m_mean_eval_f1 = 0

        overfit_indicator = 0

        # best_eval_precision = 0
        # best_eval_recall = 0
        best_eval_f1 = 0
        # best_eval_bleu = 0
        # self.f_init_word_embed(pretrain_word_embed, network)
        try:
            for epoch in range(self.m_epochs):
                print("++"*10, epoch, "++"*10)

                s_time = datetime.datetime.now()
                self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()
                print("validation epoch duration", e_time-s_time)

                # if last_eval_loss == 0:
                #     last_eval_loss = self.m_mean_eval_loss
                # elif last_eval_loss < self.m_mean_eval_loss:
                #     print(
                #         "!"*10, "error val loss increase", "!"*10,
                #         "last val loss %.4f" % last_eval_loss,
                #         "cur val loss %.4f" % self.m_mean_eval_loss
                #     )
                #     overfit_indicator += 1
                #     # if overfit_indicator > self.m_overfit_epoch_threshold:
                #     # 	break
                # else:
                #     print(
                #         "last val loss %.4f" % last_eval_loss,
                #         "cur val loss %.4f" % self.m_mean_eval_loss)
                #     last_eval_loss = self.m_mean_eval_loss

                if last_eval_f1 == 0:
                    last_eval_f1 = self.m_mean_eval_f1
                elif last_eval_f1 > self.m_mean_eval_f1:
                    print(
                        "!"*10, "error val f1 decrease", "!"*10,
                        "last val f1 %.4f" % last_eval_f1,
                        "cur val f1 %.4f" % self.m_mean_eval_f1
                    )
                    overfit_indicator += 1
                else:
                    print(
                        "last val f1 %.4f" % last_eval_f1,
                        "cur val f1 %.4f" % self.m_mean_eval_f1)
                    last_eval_f1 = self.m_mean_eval_f1

                print("--"*10, epoch, "--"*10)

                s_time = datetime.datetime.now()
                # train_data.sampler.set_epoch(epoch)
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
                # self.f_eval_train_epoch(train_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()

                print("epoch duration", e_time-s_time)

                if last_train_loss == 0:
                    last_train_loss = self.m_mean_train_loss

                elif last_train_loss < self.m_mean_train_loss:
                    print(
                        "!"*10, "error training loss increase",
                        "!"*10, "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    # break
                else:
                    print(
                        "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    last_train_loss = self.m_mean_train_loss

                # if best_eval_bleu < self.m_mean_eval_bleu:
                #     print("... saving model ...")
                #     checkpoint = {'model':network.state_dict()}
                #     self.f_save_model(checkpoint)
                #     best_eval_bleu = self.m_mean_eval_bleu

                if best_eval_f1 < self.m_mean_eval_f1:
                    print("... saving model ...")
                    checkpoint = {'model': network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_f1 = self.m_mean_eval_f1

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")

            # if best_eval_bleu < self.m_mean_eval_bleu:
            #     print("... final save ...")
            #     checkpoint = {'model':network.state_dict()}
            #     self.f_save_model(checkpoint)
            #     best_eval_bleu = self.m_mean_eval_bleu
            if best_eval_f1 < self.m_mean_eval_f1:
                print("... final save ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_f1 = self.m_mean_eval_f1

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        # loss_s_list = []
        # loss_f_list = []
        loss_list = []

        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)

        # tmp_loss_s_list = []
        # tmp_loss_f_list = []
        tmp_loss_list = []

        start_time = time.time()

        network.train()
        # feat_loss_weight = 1.0

        for g_batch in train_data:
            # print("graph_batch", g_batch)
            # if i % self.m_print_interval == 0:
            #     print("... eval ... ", i)
            # feed batch data into graph model
            graph_batch = g_batch.to(self.m_device)
            logits_f = network(graph_batch)

            labels_f = graph_batch.f_label
            loss_f = self.m_rec_loss(logits_f, labels_f.float())

            loss = loss_f

            loss_list.append(loss.item())
            tmp_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # if self.m_grad_clip:
            #     max_norm = 5.0
            #     torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)

            optimizer.step()

            self.m_train_iteration += 1

            iteration += 1
            # Logging loss (i.e. feature loss) during epochs
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%d, loss:%.4f" % (
                    iteration, np.mean(tmp_loss_list)
                    ))

                tmp_loss_list = []
                # tmp_loss_s_list = []
                # tmp_loss_f_list = []

        logger_obj.f_add_output2IO("%d, loss:%.4f" % (self.m_train_iteration, np.mean(loss_list)))
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)

        end_time = time.time()
        print("+++ duration +++", end_time-start_time)
        self.m_mean_train_loss = np.mean(loss_list)

    def f_eval_train_epoch(self, eval_data, network, optimizer, logger_obj):
        loss_list = []
        recall_list = []
        precision_list = []
        F1_list = []

        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval for train data"+" "*10)

        rouge = Rouge()

        network.eval()
        topk = 3

        start_time = time.time()

        with torch.no_grad():
            for i, (G, index) in enumerate(eval_data):
                eval_flag = random.randint(1, 100)
                if eval_flag != 2:
                    continue

                G = G.to(self.m_device)

                logits = network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)

                loss = self.m_rec_loss(glist)

                for j in range(len(glist)):
                    hyps_j = []
                    refs_j = []

                    idx = index[j]
                    example_j = eval_data.dataset.get_example(idx)
                    
                    label_sid_list_j = example_j["label_sid"]
                    gt_sent_num = len(label_sid_list_j)
                    # print("gt_sent_num", gt_sent_num)

                    g_j = glist[j]
                    snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
                    N = len(snode_id_j)
                    p_sent_j = g_j.ndata["p"][snode_id_j]
                    p_sent_j = p_sent_j.view(-1)
                    # p_sent_j = p_sent_j.view(-1, 2)

                    # topk_j, pred_idx_j = torch.topk(p_sent_j[:, 1], min(topk, N))
                    # topk_j, topk_pred_idx_j = torch.topk(p_sent_j, min(topk, N))
                    topk_j, topk_pred_idx_j = torch.topk(p_sent_j, gt_sent_num)
                    topk_pred_snode_id_j = snode_id_j[topk_pred_idx_j]

                    topk_pred_sid_list_j = g_j.nodes[topk_pred_snode_id_j].data["raw_id"]
                    topk_pred_logits_list_j = g_j.nodes[topk_pred_snode_id_j].data["p"]

                    # recall_j, precision_j = get_example_recall_precision(pred_sid_list_j.cpu(), label_sid_list_j, min(topk, N))

                    print("topk_j", topk_j)
                    print("label_sid_list_j", label_sid_list_j)
                    print("topk_pred_idx_j", topk_pred_sid_list_j)

                    recall_j, precision_j = get_example_recall_precision(topk_pred_sid_list_j.cpu(), label_sid_list_j, gt_sent_num)

                    recall_list.append(recall_j)
                    precision_list.append(precision_j)

                    for sid_k in label_sid_list_j:
                        refs_j.append(self.m_sid2swords[sid_k])

                    for sid_k in topk_pred_sid_list_j:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j)
                    refs_j = " ".join(refs_j)

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

                    # bleu_scores_j = compute_bleu([hyps_j], [refs_j])
                    bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                    bleu_list.append(bleu_scores_j)

                    # bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_bleu([refs_j], [hyps_j])
                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())

                    bleu_1_list.append(bleu_1_scores_j)

                    bleu_2_list.append(bleu_2_scores_j)

                    bleu_3_list.append(bleu_3_scores_j)

                    bleu_4_list.append(bleu_4_scores_j)

                loss_list.append(loss.item())

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            # logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)
                
        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        self.m_mean_eval_precision = np.mean(precision_list)

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

        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(self.m_eval_iteration, self.m_mean_eval_loss))
        logger_obj.f_add_output2IO("recall@%d:%.4f"%(topk, self.m_mean_eval_recall))
        logger_obj.f_add_output2IO("precision@%d:%.4f"%(topk, self.m_mean_eval_precision))

        logger_obj.f_add_output2IO("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f"%(self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r, self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r, self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        logger_obj.f_add_output2IO("bleu:%.4f"%(self.m_mean_eval_bleu))
        logger_obj.f_add_output2IO("bleu-1:%.4f"%(self.m_mean_eval_bleu_1))
        logger_obj.f_add_output2IO("bleu-2:%.4f"%(self.m_mean_eval_bleu_2))
        logger_obj.f_add_output2IO("bleu-3:%.4f"%(self.m_mean_eval_bleu_3))
        logger_obj.f_add_output2IO("bleu-4:%.4f"%(self.m_mean_eval_bleu_4))

        network.train()

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
        f_recall_list, f_precision_list, f_F1_list, f_AUC_list = [], [], [], []

        # rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        # rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        # rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        # bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        network.eval()
        # topk = 3
        start_time = time.time()

        i = 0
        with torch.no_grad():
            for graph_batch in eval_data:
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                # start_time = time.time()
                # print("... eval ", i)

                if i % 100 == 0:
                    print("... eval ... ", i)
                i += 1

                graph_batch = graph_batch.to(self.m_device)
                # model eval forward
                #### logits: batch_size*max_sen_num
                f_logits, fids, f_masks, target_f_labels = network.eval_forward(graph_batch)

                batch_size_f = f_logits.size(0)
                batch_size = graph_batch.num_graphs
                assert batch_size == batch_size_f

                # Get the user/item raw id of this current graph batch
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

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)
            self.m_mean_eval_precision = np.mean(f_precision_list)
            self.m_mean_eval_recall = np.mean(f_recall_list)
            self.m_mean_eval_f1 = np.mean(f_F1_list)
            self.m_mean_eval_AUC = np.mean(f_AUC_list)
            print(
                "P@15: %.4f" % self.m_mean_eval_precision,
                "R@15: %.4f" % self.m_mean_eval_recall,
                "F1@15: %.4f" % self.m_mean_eval_f1,
                "AUC: %.4f" % self.m_mean_eval_AUC,
            )

            logger_obj.f_add_output2IO(
                "P@15: %.4f\tR@15: %.4f\tF1@15: %.4f\tAUC: %.4f\n" % (
                    self.m_mean_eval_precision,
                    self.m_mean_eval_recall,
                    self.m_mean_eval_f1,
                    self.m_mean_eval_AUC
                )
            )

            # logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            # logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)

        network.train()

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
        validset_useritem_cdd_withproxy_file = os.path.join(self.dataset_dir, 'valid/useritem2sentids_withproxy.json')
        # trainset_user2featuretf_file = os.path.join(self.dataset_dir, 'train/user/user2featuretf.json')
        # trainset_item2featuretf_file = os.path.join(self.dataset_dir, 'train/item/item2featuretf.json')
        # trainset_sentid2featuretfidf_file = os.path.join(self.dataset_dir, 'train/sentence/sentence2feature.json')
        validset_sentid2featuretf_file = os.path.join(self.dataset_dir, 'valid/sentence/sentence2featuretf.json')
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
        with open(validset_useritem_cdd_withproxy_file, 'r') as f:
            print("Load file: {}".format(validset_useritem_cdd_withproxy_file))
            self.d_validset_useritem_cdd_withproxy = json.load(f)

        # # Load trainset user to feature tf-value dict
        # with open(trainset_user2featuretf_file, 'r') as f:
        #     print("Load file: {}".format(trainset_user2featuretf_file))
        #     self.d_trainset_user2featuretf = json.load(f)
        # # Load trainset item to feature tf-value dict
        # with open(trainset_item2featuretf_file, 'r') as f:
        #     print("Load file: {}".format(trainset_item2featuretf_file))
        #     self.d_trainset_item2featuretf = json.load(f)

        # Load validset sentence id to feature tf-value dict
        with open(validset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(validset_sentid2featuretf_file))
            self.d_validset_sentid2featuretf = json.load(f)
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

    def get_sid2featuretf_eval(self, validset_sentid2featuretf, sent2sid, train_sent_num):
        """ Get sid to featuretf mapping (on valid/test set).
            During constructing the graph data, we load the valid/test sentences. Since the
            original sentid is seperated from train-set sentence sentid, we first add the
            sentid of valid/test-set with train_sent_num and then mapping the new sent_id
            to sid. Therefore, to simplify the mapping between sid and featureid (and also
            feature tf) we need to construct this mapping here.
        """
        validset_sid2featuretf = dict()
        for key, value in validset_sentid2featuretf.items():
            assert isinstance(key, str)
            sentid = int(key) + train_sent_num
            sentid = str(sentid)
            sid = sent2sid[sentid]
            assert sid not in validset_sid2featuretf
            validset_sid2featuretf[sid] = value
        return validset_sid2featuretf

    def get_sid2feature_train(self, trainset_sentid2featuretfidf, sent2sid):
        trainset_sid2feature = dict()
        for key, value in trainset_sentid2featuretfidf.items():
            assert isinstance(key, str)     # key is the sentid
            sid = sent2sid[key]
            assert sid not in trainset_sid2feature
            trainset_sid2feature[sid] = list(value.keys())
        return trainset_sid2feature

    def get_gt_review_featuretf(self, validset_sid2featuretf, gt_sids):
        """ Get the featureid list and featuretf dict for a list of ground-truth sids
        """
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sid in gt_sids:
            cur_sid_featuretf = validset_sid2featuretf[gt_sid.item()]
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
        for sentid in self.d_validset_useritem_cdd_withproxy[true_userid][true_itemid][-2]:
            gt_sentids.append(sentid)
        # Get the feature tf of the sentence ids
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sentid in gt_sentids:
            cur_sentid_featuretf = self.d_validset_sentid2featuretf[gt_sentid]
            for featureid, tf_value in cur_sentid_featuretf.items():
                gt_featureid_set.add(featureid)
                if featureid not in gt_featuretf_dict:
                    gt_featuretf_dict[featureid] = tf_value
                else:
                    gt_featuretf_dict[featureid] += tf_value
        return list(gt_featureid_set), gt_featuretf_dict
