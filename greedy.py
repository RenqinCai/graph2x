import json
import os
import sys
import time
import ray
from rouge import Rouge
import numpy as np
import copy
import pickle
import random


# Start Ray for multi-processing
cpu_nums = 80
ray.init(num_cpus=cpu_nums)
# ray.init()


# Compute the rouge score for a hyps and a ground truth
def rouge_eval(hyps, ref):
    rouge = Rouge()
    try:
        score = rouge.get_scores(hyps, ref)[0]
        mean_score = np.mean([score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']])
    except:
        mean_score = 0.0
    return mean_score


def calLabel(article, abstract):
    """
    :param article: list of candidate sentences
    :param abstract: true review
    :return selected: a list of idx of the selected the sentences which can maximize the rouge score
    :return best_rouge: the best rouge score that this greedy algorithm can reach
    """
    hyps_list = article
    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1

    best_rouge = np.max(scores)

    # if the true review is empty, the best rouge score can only be 0.0
    if best_rouge == 0.0:
        return selected, best_rouge

    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = " ".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break
    # print(selected, best_rouge)
    return selected, best_rouge


@ray.remote
def ray_best_rouge(index, test_reviews):
    """
    :param id_to_sent: dict which mapping id to sentence
    :param user_sent_ids: dict which mapping user to its relevant sents' ids
    :param item_sent_ids: dict which mapping item to its relevant sents' ids
    :param test_reviews: list of test reviews in a format of [[user_id, item_id, rating, review],...]
    :return : a list of best rouge score for each review
    """
    # Load some dictionaries
    # with open('sent_to_id.pickle', 'rb') as handle:
    #     sent_to_id = pickle.load(handle)
    with open('id_to_sent.pickle', 'rb') as handle:
        id_to_sent = pickle.load(handle)
    with open('user_sentences_ids.pickle', 'rb') as handle:
        user_sent_ids = pickle.load(handle)
    with open('item_sentences_ids.pickle', 'rb') as handle:
        item_sent_ids = pickle.load(handle)
    best_rouge_scores = []
    cnt = 0
    for test_review_chunk in test_reviews:
        user_id = int(test_review_chunk[0])
        item_id = int(test_review_chunk[1])
        # rating = test_review_chunk[2]
        review = test_review_chunk[3]
        # Get the user's candidate set
        user_candidates = user_sent_ids[user_id]
        # Get the item's candidate set
        item_candidates = item_sent_ids[item_id]
        # Get the total candidate set
        this_candidates = user_candidates.union(item_candidates)
        # Get the corresponding sentences
        this_candidate_sents = []
        for this_cand in this_candidates:
            this_candidate_sents.append(id_to_sent[this_cand])
        # Compute the label and the best rouge score
        selected_sents, best_rouge = calLabel(this_candidate_sents, review)
        best_rouge_scores.append(best_rouge)
        cnt += 1
        if cnt % 10 == 0:
            print("[Thread {0}] Finished {1} review instances".format(index, cnt))
    return best_rouge_scores


if __name__ == "__main__":
    # The main function
    # # Load some dictionaries
    # with open('sent_to_id.pickle', 'rb') as handle:
    #     sent_to_id = pickle.load(handle)
    # with open('id_to_sent.pickle', 'rb') as handle:
    #     id_to_sent = pickle.load(handle)
    # with open('user_sentences_ids.pickle', 'rb') as handle:
    #     user_sentences_ids = pickle.load(handle)
    # with open('item_sentences_ids.pickle', 'rb') as handle:
    #     item_sentences_ids = pickle.load(handle)
    # Load the test dataset
    test_review = []
    file_path = './test_example_short.json'
    cnt = 0
    with open(file_path) as f:
        for line in f:
            line_data = json.loads(line)
            user_id = line_data['user']
            item_id = line_data['item']
            rating = line_data['rating']
            review = line_data['review']
            test_review.append([user_id, item_id, rating, review])
            cnt += 1
    print("{} lines of test data loaded!".format(cnt))
    # Start parallel greedy label assigning and rouge score computing
    # random.shuffle(test_review)
    results_rouge = []
    window_size = 100
    num_threads = 50
    print("We select {} lines".format(window_size*num_threads))
    for i in range(num_threads):
        results_rouge.append(ray_best_rouge.remote(i, test_review[i*window_size:(i+1)*window_size]))
    res_best_rouge = ray.get(results_rouge)
    res_score = []
    for score in res_best_rouge:
        res_score.extend(score)
    print("Average best rouge score: {}".format(np.mean(res_score)))
    ray.shutdown()
