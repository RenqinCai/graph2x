import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def get_tfidf_embedding(text):
    """
    :param text: list, sent_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight

def compress_array(a, id2word):
    """
    :param a: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                d[i][id2word[j]] = a[i][j]
    return d


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    # parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    parser.add_argument('--input_data_path', type=str, default='data/ratebeer/graph/', help='input data path')
    parser.add_argument('--input_data_file', type=str, default='user_sentences.json', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='ratebeer', help='dataset name')
    parser.add_argument('--output_data_path', type=str, default='data/ratebeer/graph/')
    parser.add_argument('--vocab_file', type=str, default='vocab')

    args = parser.parse_args()

    input_data_path = args.input_data_path
    input_data_file = args.input_data_file
    input_data_abs_file = os.path.join(input_data_path, input_data_file)

    ### load vocab and transform the doc into vectors
    vocab_file = args.vocab_file
    vocab_abs_file = os.path.join(input_data_path, vocab_file)

    # save_dir = os.path.join("cache", args.dataset)
    save_dir = args.output_data_path
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = GetType(save_dir) + ".w2s.tfidf.json"
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    fout = open(saveFile, "w")
    with open(input_data_abs_file) as f:
        for line in f:
            e = json.loads(line)

            sents = e["review"]

            # if isinstance(e["text"], list) and isinstance(e["text"][0], list):
            #     sents = catDoc(e["text"])
            # else:
            #     sents = e["text"]

            cntvector, tfidf_weight = get_tfidf_embedding(sents)
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():   # word -> tfidf matrix row number
                id2word[tfidf_id] = w
            tfidfvector = compress_array(tfidf_weight, id2word)
            fout.write(json.dumps(tfidfvector) + "\n")


if __name__ == '__main__':
    main()
        
