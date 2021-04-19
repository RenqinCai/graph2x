import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loss import XE_LOSS, BPR_LOSS, SIG_LOSS
from metric import get_example_recall_precision
from model import GraphX
# from infer_new import _INFER
import random
import dgl

class TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_train_precision = 0
        self.m_mean_train_recall = 0

        self.m_mean_val_loss = 0
        self.m_mean_eval_precision = 0
        self.m_mean_eval_recall = 0
        
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

        overfit_indicator = 0

        # best_eval_precision = 0
        best_eval_recall = 0
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

                if best_eval_recall < self.m_mean_eval_recall:
                        print("... saving model ...")
                        checkpoint = {'model':network.state_dict()}
                        self.f_save_model(checkpoint)
                        best_eval_recall = self.m_mean_eval_recall

            s_time = datetime.datetime.now()
            self.f_eval_epoch(test_data, network, optimizer, logger_obj)
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
            self.f_eval_epoch(test_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_list = []

        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)

        tmp_loss_list = []

        network.train()
        # train_data.dataset.f_neg_sample()

        for i, (G, index) in train_data:
            iter_start_time = time.time()

            logits = network(G)

            snode_id = G.filter_nodes(lambda nodes, nodes.data["dtype"]==1)
            labels = G.ndata["label"][snode_id]
            G.nodes[snode_id].data["loss"] = self.m_criterion(logits, labels).unsqueeze(-1)
            loss = dgl.sum_nodes(G, "loss")
            loss = loss.mean()

            loss_list.append(loss.item()) 
            
            tmp_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            
            if self.m_grad_clip:
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()

            self.m_train_iteration += 1
            
            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(iteration, np.mean(tmp_loss_list)))

                tmp_loss_list = []
                           
        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(self.m_train_iteration, np.mean(loss_list)))
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)
       
        self.m_mean_train_loss = np.mean(loss_list)
      
    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
        loss_list = []
        recall_list = []
        precision_list = []

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        network.eval()
        topk = 3
        with torch.no_grad():
            for i, (G, index) in eval_data:
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                
                logits = network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                labels = G.ndata["label"][snode_id]

                G.nodes[snode_id].data["loss"] = self.m_criterion(logits, labels).unsqueeze(-1)
                loss = dgl.sum_nodes(G, "loss")
                loss = loss.mean()

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)
                for j in range(len(glist)):
                    idx = index[j]
                    example = eval_data.dataset.get_example(idx)
                    
                    label_sid_list = example["label_sid"]

                    g = glist[j]
                    snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
                    N = len(snode_id)
                    p_sent = g.ndata["p"][snode_id]
                    p_sent = p_sent.view(-1, 2)

                    label = g.ndata["label"][snode_id].cpu()

                    topk, pred_idx = torch.topk(p_sent[:, 1], min(topk, N))
                    pred_idx = pred_idx.cpu()

                    recall, precision = get_example_recall_precision(pred_idx, label, min(topk, N))

                    recall_list.append(recall)
                    precision_list.append(precision)

                loss_list.append(loss)
                
            logger_obj.f_add_output2IO("%d, NLL_loss:%.4f, recall@%d:%.4f"%(self.m_eval_iteration, np.mean(loss_list), topk, np.mean(recall_list)))

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)
                
        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        self.m_mean_eval_precision = np.mean(precision_list)

        network.train()

