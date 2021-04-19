import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from loss import XE_LOSS, BPR_LOSS
from metric import get_recall
from model import MF
# from infer_new import _INFER
import random

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
        self.m_rec_loss = BPR_LOSS(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

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

    def f_train(self, train_data, valid_data, test_data, network, optimizer, logger_obj):
        last_train_loss = 0
        last_eval_loss = 0

        overfit_indicator = 0

        best_eval_precision = 0
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

                
                if best_eval_precision < self.m_mean_eval_precision:
                        checkpoint = {'model':network.state_dict()}
                        self.f_save_model(checkpoint)
                        best_eval_precision = self.m_mean_eval_precision

            s_time = datetime.datetime.now()
            self.f_eval_epoch(test_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
           
            if best_eval_precision < self.m_mean_eval_precision:
                    print("... final save ...")
                    checkpoint = {'model':network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_precision = self.m_mean_eval_precision

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

        for user_batch, pos_item_batch, neg_item_batch in train_data:

            user_batch_gpu = user_batch.to(self.m_device)
            pos_item_batch_gpu = pos_item_batch.to(self.m_device)
            neg_item_batch_gpu = neg_item_batch.to(self.m_device)

            # logits, u, i, j = network(user_batch_gpu, pos_item_batch_gpu, neg_item_batch_gpu)

            logits = network(user_batch_gpu, pos_item_batch_gpu, neg_item_batch_gpu)

            NLL_loss = self.m_rec_loss(logits)
            
            # regularization = self.m_weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())

            # loss = NLL_loss+regularization

            loss = NLL_loss
            
            loss_list.append(loss.item()) 
            
            tmp_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
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

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        network.eval()
        topk = 10
        with torch.no_grad():
            for user_batch, item_batch, mask_item_batch, itemnum_batch in eval_data:

                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue

                user_batch_gpu = user_batch.to(self.m_device)
                # item_batch_gpu = item_batch.to(self.m_device)
                # mask_item_batch_gpu = mask_item_batch.to(self.m_device)

                loss = 0.0

                logits = network.f_eval_forward(user_batch_gpu)

                recall = get_recall(logits.cpu(), item_batch, mask_item_batch, itemnum_batch, k=topk)

                loss_list.append(loss)
                
                recall_list.append(recall)

            logger_obj.f_add_output2IO("%d, NLL_loss:%.4f, recall@%d:%.4f"%(self.m_eval_iteration, np.mean(loss_list), topk, np.mean(recall_list)))

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)
                
        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        network.train()

