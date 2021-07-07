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
from model import Att2Seq

from rouge import Rouge


class TRAINER(object):
    def __init__(self, args, device, vocab_obj):
        super().__init__()
        self.m_device = device
        self.m_vocab = vocab_obj

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_val_loss = 0
        self.m_mean_eval_bleu = 0

        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        self.m_criterion = nn.CrossEntropyLoss(ignore_index=self.m_vocab.pad_token_id)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_tf_ratio = args.teacher_forcing_ratio

        self.m_grad_clip = args.grad_clip
        self.m_weight_decay = args.weight_decay

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval

        print("print_interval", self.m_print_interval)
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
        self.m_mean_eval_loss = 0

        overfit_indicator = 0
        best_eval_bleu = 0

        # TODO: Fix the order of train/valid epochs and evaluation
        # i.e. the evaluation metric is for the previous training epoch

        try:
            for epoch in range(self.m_epochs):
                print("++"*10, epoch, "++"*10)

                # valid epoch, this should be applied before and 1st training epoch
                # so that we can eval the untrained model
                s_time = datetime.datetime.now()
                self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()
                print("valid epoch duration", e_time-s_time)

                if last_eval_loss == 0:
                    last_eval_loss = self.m_mean_eval_loss
                elif last_eval_loss < self.m_mean_eval_loss:
                    print(
                        "!"*10, "error val loss increase",
                        "!"*10, "last val loss %.4f" % last_eval_loss,
                        "cur val loss %.4f" % self.m_mean_eval_loss
                    )
                    overfit_indicator += 1
                else:
                    print(
                        "last val loss %.4f" % last_eval_loss,
                        "cur val loss %.4f" % self.m_mean_eval_loss
                    )
                    last_eval_loss = self.m_mean_eval_loss

                print("--"*10, epoch, "--"*10)

                # train epoch
                s_time = datetime.datetime.now()
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
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
                else:
                    print(
                        "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    last_train_loss = self.m_mean_train_loss

                # whether should we save this model based on the bleu score
                if best_eval_bleu < self.m_mean_eval_bleu:
                    print("... saving model ...")
                    checkpoint = {'model': network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_bleu = self.m_mean_eval_bleu

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("valid epoch duration", e_time-s_time)
            # whether should we save this model based on the bleu score
            if best_eval_bleu < self.m_mean_eval_bleu:
                print("... saving model ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_bleu = self.m_mean_eval_bleu
            # Test on the test-set
            s_time = datetime.datetime.now()
            self.f_eval_epoch(test_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
            if best_eval_bleu < self.m_mean_eval_bleu:
                print("... final save ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_bleu = self.m_mean_eval_bleu

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("valid epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_list = []
        tmp_loss_list = []
        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)
        # set the teacher forcing ratio
        network.set_tf_ratio(self.m_tf_ratio)
        start_time = time.time()
        network.train()

        for train_batch in train_data:
            user_batch = train_batch.user
            item_batch = train_batch.item
            rating_batch = train_batch.rating
            text_batch = train_batch.text
            start_time = time.time()
            output = network(user_batch, item_batch, rating_batch, text_batch)
            end_time = time.time()
            forward_time = end_time - start_time
            output_dim = output.shape[-1]
            pred_output = output[1:].view(-1, output_dim)
            gt_text = text_batch[1:].view(-1)
            # compute loss (Cross Entropy)
            start_time = time.time()
            loss = self.m_criterion(pred_output, gt_text)
            end_time = time.time()
            loss_time = end_time - start_time
            # add current loss
            start_time = time.time()
            loss_list.append(loss.item())
            tmp_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            if self.m_grad_clip:
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)

            optimizer.step()
            end_time = time.time()
            optimizer_time = end_time - start_time

            self.m_train_iteration += 1

            print("Forward compute time: {0} \t Loss compute time: {1} \t Optim compute time: {2}".format(
                    forward_time, loss_time, optimizer_time
                )
            )

            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO(
                    "%d, loss:%.4f" % (
                        iteration, np.mean(tmp_loss_list)
                    )
                )

                tmp_loss_list = []
                # shorten the training procedure
                break

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
        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []
        loss_list = []

        self.m_eval_iteration = self.m_train_iteration
        rouge = Rouge()

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        # set the teacher forcing ratio
        network.set_tf_ratio(0.0)
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
                rating_batch = eval_batch.rating
                text_batch = eval_batch.text
                batch_size = user_batch.shape[0]

                output = network.eval_forward(user_batch, item_batch, rating_batch, text_batch)
                output_dim = output.shape[-1]
                seq_length_output = output.shape[0]
                seq_length_gt = text_batch.shape[0]
                seq_length_this_batch = min(seq_length_output, seq_length_gt)
                pred_output = output[1:seq_length_this_batch].view(-1, output_dim)
                gt_text = text_batch[1:seq_length_this_batch].view(-1)
                # compute loss (Cross Entropy)
                loss = self.m_criterion(pred_output, gt_text)
                # add current loss
                loss_list.append(loss.item())

                # convert tensor to text
                gt_sentences, pred_sentences = self.convert_tensor_to_text(text_batch, output[1:])
                assert len(gt_sentences) == batch_size
                assert len(pred_sentences) == batch_size

                for j in range(batch_size):
                    hyps_j = pred_sentences[j]
                    refs_j = gt_sentences[j]
                    if hyps_j == '':
                        hyps_j = self.m_vocab.unk_token

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

                    bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                    bleu_list.append(bleu_scores_j)

                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())

                    bleu_1_list.append(bleu_1_scores_j)
                    bleu_2_list.append(bleu_2_scores_j)
                    bleu_3_list.append(bleu_3_scores_j)
                    bleu_4_list.append(bleu_4_scores_j)

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

        # self.m_mean_eval_bleu = 0.0
        self.m_mean_eval_bleu = np.mean(bleu_list)
        self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
        self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
        self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
        self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        logger_obj.f_add_output2IO("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f" % (
            self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r,
            self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r,
            self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        logger_obj.f_add_output2IO("bleu:%.4f" % (self.m_mean_eval_bleu))
        logger_obj.f_add_output2IO("bleu-1:%.4f" % (self.m_mean_eval_bleu_1))
        logger_obj.f_add_output2IO("bleu-2:%.4f" % (self.m_mean_eval_bleu_2))
        logger_obj.f_add_output2IO("bleu-3:%.4f" % (self.m_mean_eval_bleu_3))
        logger_obj.f_add_output2IO("bleu-4:%.4f" % (self.m_mean_eval_bleu_4))

        network.train()

    def convert_tensor_to_text(self, review_data, pred_data):
        """ Convert tensor type review and prediction into raw text
        :param: review_data:
        :param: pred_data:
        :return:
            gt_sentences
            pred_sentences
        """
        # Convert tensor into batch-first format
        review_idx = torch.transpose(review_data, 0, 1)
        # Get the predict token from the predicted logits and then convert to batch-first
        _, pred_idx = pred_data.max(2)
        pred_idx = torch.transpose(pred_idx, 0, 1)

        gt_sentences = []
        pred_sentences = []
        # mapping the idx to word tokens and then forms the sentences
        for token_ids in review_idx:
            current = []
            for id in token_ids.detach().cpu().numpy():
                if id == self.m_vocab.pad_token_id or id == self.m_vocab.sos_token_id:
                    pass
                elif id == self.m_vocab.eos_token_id:
                    break
                else:
                    current.append(self.m_vocab.m_textvocab.itos[id])
            gt_sentences.append(" ".join(current))
        for token_ids in pred_idx:
            current = []
            for id in token_ids.detach().cpu().numpy():
                if id == self.m_vocab.pad_token_id or id == self.m_vocab.sos_token_id:
                    pass
                elif id == self.m_vocab.eos_token_id:
                    break
                else:
                    current.append(self.m_vocab.m_textvocab.itos[id])
            pred_sentences.append(" ".join(current))

        return gt_sentences, pred_sentences
