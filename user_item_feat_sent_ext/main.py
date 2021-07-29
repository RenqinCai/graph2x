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
from model import FeatSentExt
from eval import EVAL


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def readJson(fname):
    data = []
    line_num = 0
    with open(fname, encoding="utf-8") as f:
        for line in f:
            # print("line", line)
            line_num += 1
            try:
                data.append(json.loads(line))
            except:
                print("error", line_num)
    return data


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    seed = 1234
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # Load data
    data_obj = DATA(device=device)
    s_time = datetime.now()
    train_data, valid_data, test_data, vocab_obj = data_obj.f_load_featsent(args)
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

    # load userid2uid
    userid2uid_file = os.path.join(args.data_dir, "userid2uid.json")
    with open(userid2uid_file, 'r') as f:
        userid2uid_voc = json.load(f)
    # load itemid2iid
    itemid2iid_file = os.path.join(args.data_dir, "itemid2iid.json")
    with open(itemid2iid_file, 'r') as f:
        itemid2iid_voc = json.load(f)
    # set user_num and item_num
    vocab_obj.set_usernum(len(userid2uid_voc))
    vocab_obj.set_itemnum(len(itemid2iid_voc))

    print("feature words vocab size", vocab_obj.vocab_size)
    print("user num", vocab_obj.user_num)
    print("item num", vocab_obj.item_num)

    feat_voc = vocab_obj.m_textvocab

    # Load Pre-trained Feature Embedding
    feature2id_file = os.path.join(args.data_dir, "train/feature/feature2id.json")
    id2feature_file = os.path.join(args.data_dir, "train/feature/id2feature.json")
    feat_embed_file = os.path.join(args.data_dir, "train/feature/featureid2embedding.json")
    with open(feature2id_file, 'r') as f:
        feature2id_voc = json.load(f)
    with open(id2feature_file, 'r') as f:
        id2feature_voc = json.load(f)
    with open(feat_embed_file, 'r') as f:
        feature_emb = json.load(f)
    # Construct model's feature fid to emb mapping
    m_feature_emb = dict()
    for feat_word in feat_voc.stoi.keys():
        if feat_word == vocab_obj.unk_token or feat_word == vocab_obj.pad_token:
            assert feat_word not in feature2id_voc
            feat_word_fid = feat_voc.stoi[feat_word]
            assert args.feature_embed_size == len(feature_emb['0'])
            m_feature_emb[feat_word_fid] = [0.0] * args.feature_embed_size
        else:
            assert feat_word in feature2id_voc
            feat_word_fid = feat_voc.stoi[feat_word]
            feat_word_rawid = feature2id_voc[feat_word]
            m_feature_emb[feat_word_fid] = feature_emb[feat_word_rawid]
    assert len(m_feature_emb) == vocab_obj.vocab_size
    print("Finish read {} feature embedding.".format(len(m_feature_emb)))

    # Load Pre-trained Sentence Embedding
    id2sentence_file = os.path.join(args.data_dir, "train/sentence/id2sentence.json")
    sent_embed_file = os.path.join(args.data_dir, "train/sentence/sid2sentembed.json")
    m_id2sentence = dict()
    m_sent_emb = dict()
    with open(id2sentence_file, 'r') as f:
        m_id2sentence = json.load(f)
    with open(sent_embed_file, 'r') as f:
        for line in f:
            line_data = json.loads(line)
            sentid = list(line_data.keys())[0]
            sentembed_i = line_data[sentid]
            sentid_int = int(sentid)
            assert sentid_int not in m_sent_emb
            assert sentid in m_id2sentence
            m_sent_emb[sentid_int] = sentembed_i
    m_sent_num = len(m_sent_emb)
    assert len(m_id2sentence) == m_sent_num
    print("Finish read {} sentence embedding.".format(m_sent_num))

    # Load model
    network = FeatSentExt(
        args=args,
        vocab_obj=vocab_obj,
        device=device,
        sent_num=m_sent_num,
        feature_emb=m_feature_emb,
        sentence_emb=m_sent_emb
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
        print("="*10, "eval", "="*10)
        print("Start evaluation...")
        eval_obj = EVAL(args, device, vocab_obj, userid2uid_voc, itemid2iid_voc)
        network = network.to(device)
        eval_obj.f_init_eval(network, args.model_file, reload_model=True)
        eval_obj.f_eval(train_data, valid_data, test_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ratebeer')

    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="uifeatsentext")
    parser.add_argument('--model_path', type=str, default="../checkpoint/")
    parser.add_argument('--eval_output_path', type=str, default="../result/")

    # parser.add_argument('--min_freq', type=int, default=5)
    # parser.add_argument('--max_length', type=int, default=100)
    # parser.add_argument('--max_vocab', type=int, default=20000)

    ### model
    parser.add_argument('--user_embed_size', type=int, default=256)
    parser.add_argument('--item_embed_size', type=int, default=256)
    parser.add_argument('--feature_embed_size', type=int, default=256)
    parser.add_argument('--sent_embed_size', type=int, default=768)

    ### train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--grad_clip', action="store_true", default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--print_interval', type=int, default=200)

    parser.add_argument('--feat_finetune', action='store_true', default=False)
    parser.add_argument('--sent_finetune', action='store_true', default=False)

    ### hyper-param
    # parser.add_argument('--init_mult', type=float, default=1.0)
    # parser.add_argument('--variance', type=float, default=0.995)
    # parser.add_argument('--max_seq_length', type=int, default=100)

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
