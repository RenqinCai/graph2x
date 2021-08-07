from datetime import datetime
import numpy as np
import torch
import random
import torch.nn as nn

import pickle
import argparse
from data import DATA
import json
import os
from optimizer import OPTIM
from logger import LOGGER

import time
from train import TRAINER
from model import GraphX
from eval import EVAL
from eval_feature import EVAL_FEATURE
from eval_embed import EVAL_EMBED


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    seed = 1234
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    data_obj = DATA()

    s_time = datetime.now()
    if "beer" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_graph_ratebeer(args)
        # train_data, valid_data, vocab_obj = data_obj.f_load_ratebeer(args)
    elif "yelp" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_graph_ratebeer(args)

    e_time = datetime.now()
    print("... save data duration ... ", e_time-s_time)

    if args.train:
        now_time = datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.data_name+"_"+args.model_name)

        if not os.path.isdir(model_file):
            print("create a directory", model_file)
            os.mkdir(model_file)

        args.model_file = model_file+"/model_best_"+time_name+".pt"
        print("model_file", model_file)

    # print("vocab_size", vocab_obj.vocab_size)
    print("user num", vocab_obj.user_num)
    print("item num", vocab_obj.item_num)

    # Load sentid2wordids mapping
    sentid2wordids_file = os.path.join(args.data_dir, "train/sentence/sentid2wordid.json")
    with open(sentid2wordids_file, 'r') as f:
        print("Load file: {}".format(sentid2wordids_file))
        sentid2wordids_dict = json.load(f)
    # Construct sid2wordids mapping (convert sentid to sid)
    sid2wordids_dict = dict()
    for sentid, wordids in sentid2wordids_dict.items():
        sid = vocab_obj.m_sent2sid[sentid]
        assert sid not in sid2wordids_dict
        sid2wordids_dict[sid] = wordids
    # Load vocabulary, id2vocab
    id2vocab_file = os.path.join(args.data_dir, "id2vocab.json")
    with open(id2vocab_file, 'r') as f:
        print("Load file: {}".format(id2vocab_file))
        id2vocab_dict = json.load(f)
    # convert str keys to int
    id2vocab_dict = {int(k): v for k, v in id2vocab_dict.items()}
    # Load vocabulary, vocab2id
    vocab2id_file = os.path.join(args.data_dir, "vocab2id.json")
    with open(vocab2id_file, 'r') as f:
        print("Load file: {}".format(vocab2id_file))
        vocab2id_dict = json.load(f)
    # Load vocab word2vec
    vocab2embed_file = os.path.join(args.data_dir, "vocab2embed.json")
    with open(vocab2embed_file, 'r') as f:
        print("Load file: {}".format(vocab2embed_file))
        vocab2embed_dict = json.load(f)
    # convert str keys to int
    vocab2embed_dict = {int(k): v for k, v in vocab2embed_dict.items()}
    assert len(vocab2embed_dict) == len(id2vocab_dict)
    assert len(sid2wordids_dict) == vocab_obj.train_sent_num
    print("vocab size: {}".format(len(id2vocab_dict)))
    print("cdd sentences on train: {}".format(len(sid2wordids_dict)))

    network = GraphX(args, vocab_obj, device, vocab2id_dict, vocab2embed_dict)

    total_param_num = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_param_num += param_num
            print(name, "\t", param_num)

    print("total parameters num", total_param_num)

    if args.train:
        logger_obj = LOGGER()
        logger_obj.f_add_writer(args)

        optimizer = OPTIM(filter(lambda p: p.requires_grad, network.parameters()), args)
        trainer = TRAINER(vocab_obj, args, device, vocab2id_dict, id2vocab_dict, sid2wordids_dict)
        trainer.f_train(train_data, valid_data, network, optimizer, logger_obj)

        logger_obj.f_close_writer()

    if args.eval:
        print("="*10, "eval", "="*10)

        if args.eval_feature:
            print("Start feature prediction evaluation ...")
            eval_obj = EVAL_FEATURE(vocab_obj, args, device, vocab2id_dict, id2vocab_dict, sid2wordids_dict)
            network = network.to(device)
            eval_obj.f_init_eval(network, args.model_file, reload_model=True)
            eval_obj.f_eval(train_data, valid_data)

        elif args.eval_embed:
            print("Start feature & sentence embedding evaluation ...")
            eval_obj = EVAL_EMBED(vocab_obj, args, device, vocab2id_dict, id2vocab_dict, sid2wordids_dict)
            network = network.to(device)
            eval_obj.f_init_eval(network, args.model_file, reload_model=True)
            eval_obj.f_eval(train_data, valid_data)

        else:
            print("Start sentence prediction evaluation ...")
            eval_obj = EVAL(vocab_obj, args, device, vocab2id_dict, id2vocab_dict, sid2wordids_dict)
            network = network.to(device)
            eval_obj.f_init_eval(network, args.model_file, reload_model=True)
            eval_obj.f_eval(train_data, valid_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ratebeer')
    parser.add_argument('--data_file', type=str, default='data.pickle')
    parser.add_argument('--graph_dir', type=str, default='../output_graph/')
    parser.add_argument('--data_set', type=str, default='medium_500_pure')

    parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="graph_sentence_avg_word_extractor")
    parser.add_argument('--model_path', type=str, default="../checkpoint/")
    parser.add_argument('--eval_output_path', type=str, default="../result/")

    ### model
    parser.add_argument('--user_embed_size', type=int, default=256)
    parser.add_argument('--item_embed_size', type=int, default=256)
    parser.add_argument('--feature_embed_size', type=int, default=256)
    parser.add_argument('--sent_embed_size', type=int, default=256)
    parser.add_argument('--word_embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    # parser.add_argument('--output_hidden_size', type=int, default=256)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=256)

    ### train
    parser.add_argument('--soft_label', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--attn_dropout_rate', type=float, default=0.02)
    parser.add_argument('--ffn_dropout_rate', type=float, default=0.02)
    parser.add_argument('--grad_clip', action="store_true", default=True)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--feature_lambda', type=float, default=1.0)

    parser.add_argument('--feat_finetune', action='store_true', default=False)
    parser.add_argument('--sent_finetune', action='store_true', default=False)
    parser.add_argument('--word_finetune', action='store_true', default=False)
    parser.add_argument('--multi_task', action='store_true', default=False)

    ### hyper-param
    # parser.add_argument('--init_mult', type=float, default=1.0)
    # parser.add_argument('--variance', type=float, default=0.995)
    # parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--select_topk_s', type=int, default=5)
    parser.add_argument('--select_topk_f', type=int, default=15)

    ### others
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_feature', action='store_true', default=False)
    parser.add_argument('--eval_embed', action='store_true', default=False)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    main(args)
