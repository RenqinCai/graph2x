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

    vocab_freq_dict = {}
    ### load vocab and transform the doc into vectors
    vocab_file = args.vocab_file
    vocab_abs_file = os.path.join(input_data_path, vocab_file)

    with open(vocab_abs_file, "r") as f:
        for line in f:
            word_freq_i = line.split("\t")
            word_i = word_freq_i[0]
            freq_i = word_freq_i[1]

            vocab_freq_dict[word_i] = freq_i

    print("before filtering, vocab size", len(vocab_freq_dict))

    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            # if isinstance(e["text"], list) and isinstance(e["text"][0], list):
            #     text = catDoc(e["text"])
            # else:
            #     text = e["text"]

            text = e["review"]
            documents.append(" ".join(text))
            
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is %d, and the TFIDF vocabulary size is %d" % (len(documents), len(vectorizer.vocabulary_)))
    word_tfidf = np.array(tfidf_matrix.mean(0))
    del tfidf_matrix
    word_order = np.argsort(word_tfidf[0])

    filter_word_list = []

    id2word = vectorizer.get_feature_names()
    with open(saveFile, "w") as fout:
        for idx in word_order:
            w = id2word[idx]
            string = w + "\n"
            filter_word_list.append(w)
            try:
                fout.write(string)
            except:
                pass
                # print(string.encode("utf-8"))
    print("filter word num", len(filter_word_list))

    new_vocab_list = []
    new_vocab_file = "new_vocab"
    new_vocab_abs_file = os.path.join(output_data_path, new_vocab_file)
    with open(new_vocab_abs_file, "w") as fout:
        vocab_list = list(vocab_freq_dict.keys())
        print("vocab size", len(vocab_list))
        for word in vocab_list:
            if word not in filter_word_list:
                new_vocab_list.append(word)
            
                # print("%s\t%s\n" % (word, vocab_freq_dict[word]))
                fout.write("%s\t%s\n" % (word, vocab_freq_dict[word]))
                
    
    print("new vocab len", len(new_vocab_list))

if __name__ == '__main__':
    main()
    