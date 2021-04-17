import argparse
import datetime
import os
import shutil
import time

import dgl
import numpy as np
import torch
from rouge import Rouge

from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *

_DEBUG_FLAG_ = False

def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    
    ### load word embedding
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    hps = args
    logger.info(hps)

    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,collate_fn=graph_collate_fn)
        del dataset
        valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)
    elif hps.model == "HDSG":
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
        train_w2d_path = os.path.join(args.cache_dir, "train.w2d.tfidf.jsonl")
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,collate_fn=graph_collate_fn)
        del dataset
        val_w2d_path = os.path.join(args.cache_dir, "val.w2d.tfidf.jsonl")
        valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,collate_fn=graph_collate_fn, num_workers=32)  # Shuffle Must be False for ROUGE evaluation
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")


    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    setup_training(model, train_loader, valid_loader, valid_dataset, hps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM',help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM',help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100,help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()


    main(args)