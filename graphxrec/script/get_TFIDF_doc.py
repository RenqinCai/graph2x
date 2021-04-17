import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def get_tfidf_embedding(text, feature_word_list):
    """
    :param text: list, doc_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    # print("text", text)

    vectorizer = CountVectorizer(lowercase=True, vocabulary=feature_word_list)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight

def compress_array(a, id2word, vocab):
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                wid_voc = vocab[id2word[j]]
                d[i][wid_voc] = a[i][j]
    return d

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_path', type=str, default='data/ratebeer/graph/', help='input data path')
    parser.add_argument('--input_data_file', type=str, default='user_sentences.json', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='ratebeer', help='dataset name')
    parser.add_argument('--output_data_path', type=str, default='data/ratebeer/graph/')
    parser.add_argument('--vocab_file', type=str, default='vocab')
    parser.add_argument('--output_data_file', type=str, default='tfidf_sent.json')

    args = parser.parse_args()

    input_data_path = args.input_data_path
    input_data_file = args.input_data_file
    input_data_abs_file = os.path.join(input_data_path, input_data_file)

    vocab_file = args.vocab_file
    vocab_abs_file = os.path.join(input_data_path, vocab_file)

    # feature_word_list = []
    # with open(vocab_abs_file, "r") as vocab_f:
    #     for line in vocab_f:
    #         str_list = line.strip().split("\t")
    #         word = str_list[0]
    #         feature_word_list.append(word)
    with open(vocab_abs_file, "r") as fin:
        vocab = json.load(fin)

    feature_word_list = list(vocab.keys())
    print("feature word num", len(feature_word_list))

    save_dir = args.output_data_path
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    output_data_file = args.output_data_file
    saveFile = os.path.join(save_dir, output_data_file)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    fout = open(saveFile, "w")
    line_num = 0

    invalid_review_num = 0
    valid_review_num = 0

    with open(input_data_abs_file) as f:
        for line in f:
            e = json.loads(line)
            sents = e["review"]

            if len(sents) == 0:
                invalid_review_num += 1
                continue
            docs = [" ".join(sents)]

            line_num += 1
            # if line_num > 1:
            #     break

            if line_num % 100 == 0:
                print("line num", line_num)
            valid_review_num += 1

            cntvector, tfidf_weight = get_tfidf_embedding(docs, feature_word_list, vocab)
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():
                id2word[tfidf_id] = w
            tfidfvector = compress_array(tfidf_weight, id2word)
            fout.write(json.dumps(tfidfvector) + "\n")

    print("valid_review_num", valid_review_num)
    print("invalid_review_num", invalid_review_num)

if __name__ == '__main__':
    main()