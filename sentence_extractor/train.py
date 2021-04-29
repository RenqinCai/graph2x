import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loss import XE_LOSS, BPR_LOSS, SIG_LOSS
from metric import get_example_recall_precision, compute_bleu
from model import GraphX
# from infer_new import _INFER
import random
import dgl
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
        
        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        # self.m_rec_loss = XE_LOSS(vocab_obj.item_num, self.m_device)
        # self.m_rec_loss = BPR_LOSS(self.m_device)
        # self.m_rec_loss = SIG_LOSS(self.m_device)
        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_grad_clip = args.grad_clip
        self.m_weight_decay = args.weight_decay
        # self.m_l2_reg = args.l2_reg

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval
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
        last_eval_loss = 0
        self.m_mean_eval_loss = 0

        overfit_indicator = 0

        # best_eval_precision = 0
        best_eval_recall = 0
        best_eval_bleu = 0
        # self.f_init_word_embed(pretrain_word_embed, network)
        try: 
            for epoch in range(self.m_epochs):
                
                print("++"*10, epoch, "++"*10)

                s_time = datetime.datetime.now()
                self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()
                print("validation epoch duration", e_time-s_time)
                    
                if last_eval_loss == 0:
                    last_eval_loss = self.m_mean_eval_loss

                elif last_eval_loss < self.m_mean_eval_loss:
                    print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    
                    overfit_indicator += 1

                    # if overfit_indicator > self.m_overfit_epoch_threshold:
                    # 	break
                else:
                    print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    last_eval_loss = self.m_mean_eval_loss

                print("--"*10, epoch, "--"*10)

                s_time = datetime.datetime.now()
                # train_data.sampler.set_epoch(epoch)
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()

                print("epoch duration", e_time-s_time)

                if last_train_loss == 0:
                    last_train_loss = self.m_mean_train_loss

                elif last_train_loss < self.m_mean_train_loss:
                    print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                    # break
                else:
                    print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                    last_train_loss = self.m_mean_train_loss

                if best_eval_bleu < self.m_mean_eval_bleu:
                    print("... saving model ...")
                    checkpoint = {'model':network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_bleu = self.m_mean_eval_bleu

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
           
            if best_eval_recall < self.m_mean_eval_recall:
                    print("... final save ...")
                    checkpoint = {'model':network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_recall = self.m_mean_eval_recall

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_list = []

        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)

        tmp_loss_list = []

        start_time = time.time()

        network.train()

        for i, (G, index) in enumerate(train_data):
            G = G.to(self.m_device)

            logits = network(G)

            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
            labels = G.ndata["label"][snode_id]

            labels = labels.float()
            node_loss = self.m_criterion(logits, labels)
            
            G.nodes[snode_id].data["loss"] = node_loss
            loss = dgl.sum_nodes(G, "loss")
            loss = loss.mean()

            loss_list.append(loss.item()) 
            
            tmp_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            
            if self.m_grad_clip:
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
            
            optimizer.step()

            self.m_train_iteration += 1
            
            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(iteration, np.mean(tmp_loss_list)))

                tmp_loss_list = []

        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(self.m_train_iteration, np.mean(loss_list)))
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)

        end_time = time.time()
        print("+++ duration +++", end_time-start_time)
        self.m_mean_train_loss = np.mean(loss_list)
      
    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
        loss_list = []
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

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        rouge = Rouge()

        network.eval()
        topk = 3

        start_time = time.time()

        with torch.no_grad():
            for i, (G, index) in enumerate(eval_data):
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                # start_time = time.time()
                # print("... eval ", i)
                if i % 100 == 0:
                    print("... eval ... ", i)

                G = G.to(self.m_device)

                logits = network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                labels = G.ndata["label"][snode_id]

                # one_hot_labels = F.one_hot(labels.squeeze(-1), num_classes=2).float()

                # node_loss = self.m_criterion(logits, one_hot_labels)

                labels = labels.float()
                node_loss = self.m_criterion(logits, labels)

                G.nodes[snode_id].data["loss"] = node_loss
                loss = dgl.sum_nodes(G, "loss")
                loss = loss.mean()

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)

                for j in range(len(glist)):
                    hyps_j = []
                    refs_j = []

                    idx = index[j]
                    example_j = eval_data.dataset.get_example(idx)
                    
                    label_sid_list_j = example_j["label_sid"]

                    g_j = glist[j]
                    snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
                    N = len(snode_id_j)
                    p_sent_j = g_j.ndata["p"][snode_id_j]
                    p_sent_j = p_sent_j.view(-1)
                    # p_sent_j = p_sent_j.view(-1, 2)

                    # topk_j, pred_idx_j = torch.topk(p_sent_j[:, 1], min(topk, N))
                    topk_j, pred_idx_j = torch.topk(p_sent_j, min(topk, N))
                    pred_snode_id_j = snode_id_j[pred_idx_j]

                    pred_sid_list_j = g_j.nodes[pred_snode_id_j].data["raw_id"]
                    pred_logits_list_j =  g_j.nodes[pred_snode_id_j].data["p"]

                    # recall_j, precision_j = get_example_recall_precision(pred_idx_j, label_sid_list_j, min(topk, N))

                    # recall_list.append(recall_j)
                    # precision_list.append(precision_j)

                    for sid_k in label_sid_list_j:
                        refs_j.append(self.m_sid2swords[sid_k])

                    for sid_k in pred_sid_list_j:
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

                    bleu_scores_j = compute_bleu([hyps_j], [refs_j])
                    bleu_list.append(bleu_scores_j)

                loss_list.append(loss.item())

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            # logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)
                
        self.m_mean_eval_loss = np.mean(loss_list)
        # self.m_mean_eval_recall = np.mean(recall_list)
        # self.m_mean_eval_precision = np.mean(precision_list)

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

        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(self.m_eval_iteration, self.m_mean_eval_loss))
        logger_obj.f_add_output2IO("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f"%(self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r, self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r, self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        logger_obj.f_add_output2IO("bleu:%.4f"%(self.m_mean_eval_bleu))

        network.train()

