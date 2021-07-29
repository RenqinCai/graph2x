import numpy as np
import torch
import os
import json
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import random

dataset_name = 'medium_500_pure'


class EVAL(object):
    def __init__(self, args, device, vocab_obj, userid2uid_voc, itemid2iid_voc):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_batch_size_eval = args.batch_size_eval
        self.m_mean_loss = 0

        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_data_dir = args.data_dir
        self.m_eval_output_path = args.eval_output_path

        self.m_vocab = vocab_obj
        self.m_userid2uid = userid2uid_voc
        self.m_itemid2iid = itemid2iid_voc
        self.m_uid2userid = {v: u for u, v in self.m_userid2uid.items()}
        self.m_iid2itemid = {v: u for u, v in self.m_itemid2iid.items()}

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {}".format(dataset_name))
        testset_uicddsent_file = os.path.join(
            self.m_data_dir, "test/useritem2sentids_test.json")
        with open(testset_uicddsent_file, 'r') as f:
            print("Load file: {}".format(testset_uicddsent_file))
            self.d_testset_ui2cddsent = json.load(f)

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

    def f_eval(self, train_data, valid_data, test_data):
        print("start eval ...")
        # self.f_cluster_embedding()
        self.f_eval_new(train_data, valid_data, test_data)

    def f_eval_new(self, train_data, valid_data, test_data):
        """ Get the predicted score for each (u,i,s) and save it into file
        """
        cnt_useritem_batch = 0
        cnt_useritemsent_tuple = 0
        pred_file = os.path.join(self.m_eval_output_path, 'pred_scores.json')
        print("Predict scores json file is stored at: {}".format(pred_file))
        self.m_network.eval()
        with torch.no_grad():
            # print("Number of train data (batch): {}".format(len(train_data)))
            # print("Number of valid data (batch): {}".format(len(valid_data)))
            print("Number of test data (batch): {}".format(len(test_data)))
            print("Batch size: {}".format(self.m_batch_size_eval))
            # Perform Evaluation on eval_data / train_data
            with open(pred_file, 'w') as f_pred:
                for test_batch in test_data:
                    if cnt_useritem_batch % 100 == 0:
                        print("... eval ... ", cnt_useritem_batch)
                    user_batch = test_batch.user
                    item_batch = test_batch.item
                    sentence_batch = test_batch.sentence
                    feature_batch, feature_length_batch = test_batch.feature
                    # label_batch = test_batch.label
                    batch_size = user_batch.shape[0]
                    # forward
                    output = self.m_network.eval_forward(
                        user_batch, item_batch, sentence_batch, feature_batch, feature_length_batch
                    )
                    output = output.squeeze(dim=-1)     # shape: (batch_size,)

                    # get true user/item/sent id
                    true_user_ids = []
                    for user_id in user_batch:
                        true_user_ids.append(self.m_uid2userid[user_id.item()])
                    true_item_ids = []
                    for item_id in item_batch:
                        true_item_ids.append(self.m_iid2itemid[item_id.item()])
                    true_sent_ids = []
                    for sent_id in sentence_batch:
                        true_sent_ids.append(str(sent_id.item()))
                    # get the predicted scores
                    pred_scores = []
                    for pred_out in output:
                        pred_scores.append(pred_out.item())

                    cnt_useritem_batch += 1

                    for j in range(batch_size):
                        true_user_id_j = true_user_ids[j]
                        true_item_id_j = true_item_ids[j]
                        true_sent_id_j = true_sent_ids[j]
                        pred_score_j = pred_scores[j]
                        assert true_sent_id_j in self.d_testset_ui2cddsent[true_user_id_j][true_item_id_j][0]
                        # save this into file
                        cur_pred_data = {
                            'user': true_user_id_j,
                            'item': true_item_id_j,
                            'sentid': true_sent_id_j,
                            'score': pred_score_j
                        }
                        json.dump(cur_pred_data, f_pred)
                        f_pred.write("\n")
                        cnt_useritemsent_tuple += 1

        print("Totally {0} batches ({1} data instances).".format(
            cnt_useritem_batch, cnt_useritemsent_tuple
        ))

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
