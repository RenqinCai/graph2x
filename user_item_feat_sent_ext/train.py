import os
import json
import time
import argparse
import numpy as np
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from loss import XE_LOSS, BPR_LOSS, SIG_LOSS
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_sentence_bleu
from model import FeatSentExt


class TRAINER(object):
    def __init__(self, args, device, vocab_obj):
        super().__init__()
        self.m_device = device
        self.m_vocab = vocab_obj

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_val_loss = 0

        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        self.m_criterion = nn.BCEWithLogitsLoss()

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_grad_clip = args.grad_clip
        self.m_weight_decay = args.weight_decay

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval

        print("print_interval", self.m_print_interval)
        self.m_overfit_epoch_threshold = 3

    def f_save_model(self, checkpoint):
        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, valid_data, test_data, network, optimizer, logger_obj):
        last_train_loss = 0
        last_eval_loss = 0
        self.m_mean_eval_loss = 0

        overfit_indicator = 0
        best_eval_bleu = 0
        best_train_loss = 0
        best_eval_loss = 0

        try:
            for epoch in range(self.m_epochs):
                print("++"*10, epoch, "++"*10)

                # s_time = datetime.datetime.now()
                # # using valid/test to perform validation
                # self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
                # # self.f_eval_epoch(test_data, network, optimizer, logger_obj)
                # e_time = datetime.datetime.now()
                # print("valid epoch duration", e_time-s_time)

                # if last_eval_loss == 0:
                #     last_eval_loss = self.m_mean_eval_loss
                # elif last_eval_loss < self.m_mean_eval_loss:
                #     print(
                #         "!"*10, "error val loss increase",
                #         "!"*10, "last val loss %.4f" % last_eval_loss,
                #         "cur val loss %.4f" % self.m_mean_eval_loss
                #     )
                #     overfit_indicator += 1
                # else:
                #     print(
                #         "last val loss %.4f" % last_eval_loss,
                #         "cur val loss %.4f" % self.m_mean_eval_loss
                #     )
                #     last_eval_loss = self.m_mean_eval_loss

                print("--"*10, epoch, "--"*10)

                # train epoch
                s_time = datetime.datetime.now()
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()

                print("epoch duration", e_time-s_time)

                if last_train_loss == 0:
                    # For Epoch 0
                    last_train_loss = self.m_mean_train_loss
                    print("Epoch {} ... saving model ...".format(epoch))
                    logger_obj.f_add_output2IO(
                        "Epoch {} ... saving model ...".format(epoch)
                    )
                    checkpoint = {'model': network.state_dict()}
                    self.f_save_model(checkpoint)
                elif last_train_loss < self.m_mean_train_loss:
                    print(
                        "!"*10, "error training loss increase",
                        "!"*10, "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                else:
                    print(
                        "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    last_train_loss = self.m_mean_train_loss
                    print("Epoch {} ... saving model ...".format(epoch))
                    logger_obj.f_add_output2IO(
                        "Epoch {} ... saving model ...".format(epoch)
                    )
                    checkpoint = {'model': network.state_dict()}
                    self.f_save_model(checkpoint)

            # # Test on the valid-set
            # s_time = datetime.datetime.now()
            # self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            # e_time = datetime.datetime.now()
            # print("valid epoch duration", e_time-s_time)
            # whether should we save this model
            if last_train_loss > self.m_mean_train_loss:
                print("Epoch {} ... saving model ...".format(epoch))
                logger_obj.f_add_output2IO(
                    "Epoch {} ... saving model ...".format(epoch)
                )
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
            # # Test on the test-set
            # s_time = datetime.datetime.now()
            # self.f_eval_epoch(test_data, network, optimizer, logger_obj)
            # e_time = datetime.datetime.now()
            # print("test epoch duration", e_time-s_time)

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
            if last_train_loss > self.m_mean_train_loss:
                print("... final save ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)

            # s_time = datetime.datetime.now()
            # self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            # e_time = datetime.datetime.now()
            # print("valid epoch duration", e_time-s_time)

            # s_time = datetime.datetime.now()
            # self.f_eval_epoch(test_data, network, optimizer, logger_obj)
            # e_time = datetime.datetime.now()
            # print("test epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_list = []
        tmp_loss_list = []
        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)

        start_time = time.time()
        network.train()

        for train_batch in train_data:
            user_batch = train_batch.user
            item_batch = train_batch.item
            sentence_batch = train_batch.sentence
            feature_batch, feature_length_batch = train_batch.feature
            label_batch = train_batch.label
            # forward
            output = network(
                user_batch, item_batch, sentence_batch, feature_batch, feature_length_batch
            )
            output = output.squeeze(dim=-1)
            # compute loss (Cross Entropy)
            loss = self.m_criterion(output, label_batch.float())
            # add current loss
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
                logger_obj.f_add_output2IO(
                    "%d, loss:%.4f" % (
                        iteration, np.mean(tmp_loss_list)
                    )
                )

                tmp_loss_list = []

        logger_obj.f_add_output2IO(
            "%d, loss:%.4f" % (
                self.m_train_iteration, np.mean(loss_list)
            )
        )
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)
        end_time = time.time()
        print("+++ duration +++", end_time-start_time)
        self.m_mean_train_loss = np.mean(loss_list)

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
        """ This can work on both valid and test
        """
        loss_list = []

        self.m_eval_iteration = self.m_train_iteration
        logger_obj.f_add_output2IO("---"*10 + " eval " + "---"*10)

        network.eval()
        start_time = time.time()
        i = 0
        with torch.no_grad():
            for eval_batch in eval_data:
                if i % 100 == 0:
                    print("... eval ... ", i)
                i += 1
                user_batch = eval_batch.user
                item_batch = eval_batch.item
                sent_batch = eval_batch.sentence
                feat_batch, feat_length = eval_batch.feature
                batch_size = user_batch.shape[0]

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)

            # logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)

        network.train()
