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
from model import Encoder, Decoder, Att2Seq
from eval import EVAL


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

    # local_rank = None
    # if args.parallel:
    #     local_rank = args.local_rank
    #     torch.distributed.init_process_group(backend="nccl")
    #     device = torch.device('cuda:{}'.format(local_rank))

    # Load data
    data_obj = DATA(device=device)
    s_time = datetime.now()
    train_data, valid_data, test_data, vocab_obj = data_obj.f_load_rnn_ratebeer(args)
    e_time = datetime.now()
    print("... load data duration ... ", e_time-s_time)

    if args.train:
        now_time = datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.data_name+"_"+args.model_name)

        if not os.path.isdir(model_file):
            print("create a directory", model_file)
            os.mkdir(model_file)

        args.model_file = model_file+"/model_best_"+time_name+".pt"
        print("model_file: ", model_file)

    print("vocab_size", vocab_obj.vocab_size)
    print("user num", vocab_obj.user_num)
    print("item num", vocab_obj.item_num)

    # Load model
    enc = Encoder(
        user_num=vocab_obj.user_num,
        item_num=vocab_obj.item_num,
        enc_hid_dim=args.enc_hidden_dim,
        dec_hid_dim=args.dec_hidden_dim,
        rnn_layers=args.rnn_layers,
        rating_range=args.rating_range
    )
    dec = Decoder(
        output_dim=vocab_obj.vocab_size,
        emb_dim=args.word_embed_dim,
        enc_hid_dim=args.enc_hidden_dim,
        dec_hid_dim=args.dec_hidden_dim,
        num_layers=args.rnn_layers,
        dropout=args.rnn_dropout_rate
    )
    network = Att2Seq(
        encoder=enc,
        decoder=dec,
        device=device,
        tf_ratio=args.teacher_forcing_ratio,
        max_seq_length=args.max_seq_length
    )
    network.to(device)

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
        # NOTE: The default optimizer for Att2Seq is RMSprop
        optimizer = OPTIM(filter(lambda p: p.requires_grad, network.parameters()), args)
        trainer = TRAINER(args, device, vocab_obj)
        trainer.f_train(train_data, valid_data, test_data, network, optimizer, logger_obj)

        logger_obj.f_close_writer()

    if args.eval:
        # TODO: Implement Evaluation
        print("="*10, "eval", "="*10)
        print("Start evaluation...")
        network.set_tf_ratio(0.0)       # teacher forcing ratio should be assigned as 0
        eval_obj = EVAL(vocab_obj, args, device)
        network = network.to(device)
        eval_obj.f_init_eval(network, args.model_file, reload_model=True)
        eval_obj.f_eval(train_data, valid_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ratebeer')
    # parser.add_argument('--data_file', type=str, default='data.pickle')
    # parser.add_argument('--graph_dir', type=str, default='../output_graph/')

    # parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="att2seq")
    parser.add_argument('--model_path', type=str, default="../checkpoint/")
    parser.add_argument('--eval_output_path', type=str, default="../result/")

    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--max_vocab', type=int, default=15000)
    parser.add_argument('--rating_range', type=int, default=20)

    ### model
    # parser.add_argument('--user_embed_size', type=int, default=256)
    # parser.add_argument('--item_embed_size', type=int, default=256)
    # parser.add_argument('--feature_embed_size', type=int, default=256)
    # parser.add_argument('--sent_embed_size', type=int, default=256)
    # parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--enc_hidden_dim', type=int, default=64)
    parser.add_argument('--dec_hidden_dim', type=int, default=512)
    parser.add_argument('--word_embed_dim', type=int, default=512)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout_rate', type=float, default=0.2)

    ### train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    # parser.add_argument('--attn_dropout_rate', type=float, default=0.02)
    # parser.add_argument('--ffn_dropout_rate', type=float, default=0.02)
    parser.add_argument('--grad_clip', action="store_true", default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--print_interval', type=int, default=200)
    # parser.add_argument('--feature_lambda', type=float, default=1.0)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)

    # parser.add_argument('--feat_finetune', action='store_true', default=False)
    # parser.add_argument('--sent_finetune', action='store_true', default=False)
    # parser.add_argument('--multi_task', action='store_true', default=False)

    ### hyper-param
    # parser.add_argument('--init_mult', type=float, default=1.0)
    # parser.add_argument('--variance', type=float, default=0.995)
    parser.add_argument('--max_seq_length', type=int, default=100)

    ### others
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    # parser.add_argument('--eval_feature', action='store_true', default=False)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--verbose', action="store_true", default=False)

    args = parser.parse_args()

    main(args)
