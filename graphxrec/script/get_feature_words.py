"""
This script is to obtain featuer words 
"""

import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def calTFidf(text):
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(text)

    print("wordcount", wordcount.shape)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_data_path', type=str, default='data/ratebeer/graph/', help='input data path')
    parser.add_argument('--input_data_file', type=str, default='user_sentences.json', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='ratebeer', help='dataset name')
    parser.add_argument('--output_data_path', type=str, default='data/ratebeer/graph')
    parser.add_argument('--vocab_file', type=str, default='vocab')

    args = parser.parse_args()

    input_data_path = args.input_data_path
    output_data_path = args.output_data_path

    if output_data_path == "":
        output_data_path = input_data_path
    
    input_file = os.path.join(input_data_path, args.input_data_file)

    # save_dir = os.path.join(input_data_path, args.dataset)
    save_dir = output_data_path
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "filter_word.txt")
    print("Save low tfidf words in dataset %s to %s" % (args.dataset, saveFile))


    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            text = e["review"]
            documents.append(" ".join(text))
            
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is %d, and the TFIDF vocabulary size is %d" % (len(documents), len(vectorizer.vocabulary_)))
    word_tfidf = np.array(tfidf_matrix.mean(0))
    del tfidf_matrix
    word_order = np.argsort(-word_tfidf[0])

    vocab = {}

    # feature_word_list = []
    id2word = vectorizer.get_feature_names()

    feature_word_num = 5000
    feature_word_index = 0

    with open(saveFile, "w") as fout:
        for idx in word_order:
            if feature_word_index > feature_word_num:
                break
            w = id2word[idx]
            fout.write(w)

            vocab[w] = len(vocab)
            # feature_word_index += 1
            # feature_word_list.append(w)
        
    # print("feature word num", len(feature_word_list))

    vocab_file = args.vocab_file
    vocab_abs_file = os.path.join(output_data_path, vocab_file)
    with open(vocab_abs_file, "w") as fout:
        json.dump(vocab, fout)
      
if __name__ == '__main__':
    main()
    