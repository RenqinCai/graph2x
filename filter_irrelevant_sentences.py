from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import json
# import ray
import multiprocessing

from transformers import (AutoConfig, AutoTokenizer, AutoModel)
import torch
import torch.nn.functional as F
import datetime

print("cuda", torch.cuda.is_available())

# cpu_nums = 20
# ray.init(num_cpus=cpu_nums)

### load the pretrained model
output_dir = "../checkpoint/test-mlm-wwm"

config_file = os.path.join(output_dir, "tokenizer_config.json")
model_path = os.path.join(output_dir)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device)


### load the data

### load sentences
dataset_path = "/p/reviewde/data/ratebeer/sentences"
id2sent_file = "id_to_sent.pickle"

max_corpus_size = 100
embedding_cache_path = 'ratebeer-embeddings-size-{}.pkl'.format(max_corpus_size)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

id2sent_abs_file = os.path.join(dataset_path, id2sent_file)
id2sent_abs_f = open(id2sent_abs_file, "rb")

id2sent_dict = pickle.load(id2sent_abs_f)
sent_num = len(id2sent_dict)
print("sentences num", sent_num)

### load user, item, candidate sent id, target sent id
input_pair_file = "test_example_100.json"
pair_abs_file = os.path.join(dataset_path, input_pair_file)
print("pair file", pair_abs_file)

data = []
line_num = 0
with open(pair_abs_file) as f:
    for line in f:
        line_data = json.loads(line)
        data.append(line_data)

data_num = len(data)
print("data num", data_num)

print("xxx"*3, " Start ", "xxx"*3)
start_time = time.time()
print("start_time", datetime.datetime.now())

def get_sentence_embed(sent):
    encoded_input = tokenizer(sent, padding=True, truncation=True, max_length=30, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    corpus_embeddings = corpus_embeddings.cpu()

    return corpus_embeddings

sim_threshold = 0.9
candidate_num_list = []

def f_process_candidate(thread_input_data):
    thread_data_num = len(thread_input_data)

    for data_idx in range(thread_data_num):
        data_i = thread_input_data[data_idx]
        user_i = data_i["user"]
        item_i = data_i["item"]
        candidate_i = data_i["candidate"]
        review_i = data_i["review"]

        candidate_num_i = len(candidate_i)
        review_num_i = len(review_i)

        target_i = []

        candidate_sentence_i = []

        candidate_num_list.append(candidate_num_i)

        # print("before candidate_num_i", candidate_num_i)
        if candidate_num_i > 12000:
            candidate_num_i = int(12000)
        # print("after candidate_num_i", candidate_num_i)

        batch_size = 6000
        batch_num = candidate_num_i//batch_size

        candidate_embed_i = []

        for batch_idx in range(batch_num):
            candidate_sentence_batch_i = []
            for j in range(batch_size):
                candidate_idx = batch_idx*batch_size+j
                candidate_ij = candidate_i[candidate_idx]
                sent_ij = id2sent_dict[candidate_ij]
                candidate_sentence_batch_i.append(sent_ij)
            candidate_embed_batch_i = get_sentence_embed(candidate_sentence_batch_i)
            candidate_embed_i.append(candidate_embed_batch_i)

        candidate_embed_i = torch.cat(candidate_embed_i, dim=0)
        # print("candidate_embed_i", candidate_embed_i.size())
        review_sentence_i = []

        for k in range(review_num_i):
            review_ik = review_i[k] 
            sent_ik = id2sent_dict[review_ik]
            review_sentence_i.append(sent_ik)

        review_embed_i = get_sentence_embed(review_sentence_i)

        cos_scores = util.pytorch_cos_sim(candidate_embed_i, review_embed_i)

        mask = cos_scores > sim_threshold
        mask_sum = torch.sum(mask, dim=1)

        nonzero_index = torch.nonzero(mask_sum)
        nonzero_index = nonzero_index.squeeze(1)

        nonzero_index = nonzero_index.cpu().numpy()
        target_i = list(np.array(candidate_i)[nonzero_index])

        # target_i = list()
        data_i["target"] = target_i
    
    return thread_input_data

num_threads = 4
input_thread_data_list = [[] for i in range(num_threads)]

data_num = 8
for data_idx in range(data_num):
    data_i = data[data_idx]

    thread_idx = data_idx%num_threads
    input_thread_data_list[thread_idx].append(data_i)

# torch.multiprocessing.set_start_method("spawn", force=True)
# with multiprocessing.Pool(processes=num_threads) as pool:
#     results = list(pool.map(f_process_candidate, input_thread_data_list))

results = list(map(f_process_candidate, input_thread_data_list))

thread_output_data = []
for thread_idx in range(num_threads):

    thread_output_data.extend(results[thread_idx])

output_data = thread_output_data

end_time = time.time()

duration = end_time-start_time
print("duration", duration)
print("end time", datetime.datetime.now())

outputput_pair_file = "new_test_example_100.json"
output_pair_abs_file = os.path.join(dataset_path, outputput_pair_file)
print("output pair file", output_pair_abs_file)

with open(output_pair_abs_file, "w") as f:
    # for data_idx in range(data_num):
    for data_i in output_data:
        # data_i = data[data_idx]
        user_i = data_i["user"]
        item_i = data_i["item"]
        candidate_i = [i.item() for i in data_i["target"]]
        review_i = [i for i in list(data_i["review"])]
        
        # print(candidate_i)
        # print(review_i)

        line = {"user": user_i, "item": item_i, "candidate": candidate_i, "review": review_i}

        json.dump(line, f)
        f.write("\n")
