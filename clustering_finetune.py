from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import torch

from transformers import (AutoConfig, AutoTokenizer, AutoModel)

print("cuda", torch.cuda.is_available())

def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):

    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    extracted_communities = []

    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)

            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)
            
            extracted_communities.append(new_cluster)

    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break
        
        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities

# model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')

# model = SentenceTransformer('stsb-roberta-base')

# url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
# dataset_path = "quora_duplicate_questions.tsv"
# max_corpus_size = 50000 # We limit our corpus to only the first 50k questions
# embedding_cache_path = 'quora-embeddings-size-{}.pkl'.format(max_corpus_size)

dataset_path = "/p/reviewde/data/ratebeer/sentences/test_sentences.txt"
max_corpus_size = 50000
embedding_cache_path = 'ratebeer-embeddings-size-{}.pkl'.format(max_corpus_size)

output_dir = "/tmp/test-mlm-wwm"

config_file = os.path.join(output_dir, "tokenizer_config.json")
model_path = os.path.join(output_dir)

# config = AutoConfig(config_file)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained(output_dir)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()

    with open(dataset_path, encoding="utf8") as fIn:
        # reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in fIn:
            corpus_sentences.add(row)
            if len(corpus_sentences) >= max_corpus_size:
                break

    # with open(dataset_path, encoding='utf8') as fIn:
    #     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    #     for row in reader:
    #         corpus_sentences.add(row['question1'])
    #         if len(corpus_sentences) >= max_corpus_size:
    #             break

    #         corpus_sentences.add(row['question2'])
    #         if len(corpus_sentences) >= max_corpus_size:
    #             break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    
    print("corpus size", len(corpus_sentences))
    # exit()
    encoded_input = tokenizer(corpus_sentences, padding=True, truncation=True, max_length=50, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    corpus_embeddings = corpus_embeddings.cpu().numpy()
    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

print("Start clustering")
start_time = time.time()

#Two parameter to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = community_detection(corpus_embeddings, min_community_size=25, threshold=0.95)

print("cluster num", len(clusters))


output_cluster_file = "cluster.txt"
output_cluster_f = open(output_cluster_file, "w")

#Print all cluster / communities
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))

    output_cluster_f.write("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))

    for sentence_id in cluster:
        print("\t", corpus_sentences[sentence_id])

        output_cluster_f.write("\t"+corpus_sentences[sentence_id])

output_cluster_f.close()

print("Clustering done after {:.2f} sec".format(time.time() - start_time))
