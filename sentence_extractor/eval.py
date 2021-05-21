import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %matplotlib inline
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import get_example_recall_precision, compute_bleu, get_bleu
from rouge import Rouge
import dgl

class EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_item2iid = vocab_obj.m_item2iid

        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_device = device
        self.m_model_path = args.model_path

    def f_init_eval(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        print("eval new")
        # self.f_cluster_embedding()
        self.f_eval_new(train_data, eval_data)

    def f_cluster_embedding(self):
        self.m_iid2item = {self.m_item2iid[k]:k for k in self.m_item2iid}

        embeds = self.m_network.m_item_embed.weight.data.cpu().numpy()
        item_num = len(embeds)
        labels = [self.m_iid2item[i] for i in range(item_num)]

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(embeds)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            
        plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.savefig("item_embed_tsne.png")
        

    def f_eval_new(self, train_data, eval_data):

        recall_list = []
        precision_list = []
        F1_list = []

        rouge_1_f_list = []
        rouge_1_p_list = []
        rouge_1_r_list = []

        rouge_2_f_list = []
        rouge_2_p_list = []
        rouge_2_r_list = []

        rouge_l_f_list = []
        rouge_l_p_list = []
        rouge_l_r_list = []

        bleu_list = []
        bleu_1_list = []
        bleu_2_list = []
        bleu_3_list = []
        bleu_4_list = []

        rouge = Rouge()

        print('--'*10)

        debug_index = 0

        topk = 3
        self.m_network.eval()
        with torch.no_grad():
            for i, (G, index) in enumerate(eval_data):
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                if i % 100 == 0:
                    print("... eval ", i)

                # debug_index += 1
                # if debug_index > 1:
                #     break
                    
                G = G.to(self.m_device)

                logits = self.m_network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                labels = G.ndata["label"][snode_id]

                labels = labels.float()
                node_loss = self.m_criterion(logits, labels)
                # print("node_loss", node_loss.size())

                G.nodes[snode_id].data["loss"] = node_loss
                loss = dgl.sum_nodes(G, "loss")
                loss = loss.mean()

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)

                for j in range(len(glist)):
                    hyps_j = []
                    refs_j = []

                    idx = index[j]
                    example_j = eval_data.dataset.get_example(idx)
                    
                    label_sid_list_j = example_j["label_sid"]

                    g_j = glist[j]
                    snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==1)

                    """
                    get feature attn weight
                    """

                    for k in snode_id_j:
                        predecessors = list(g_j.predecessors(k))
                        edges_id = g_j.edge_ids(predecessors, k)
                        

                    N = len(snode_id_j)
                    p_sent_j = g_j.ndata["p"][snode_id_j]
                    
                    p_sent_j = p_sent_j.view(-1)
                    p_sent_j = F.sigmoid(p_sent_j)

                    topk_j, pred_idx_j = torch.topk(p_sent_j, min(topk, N))
                    pred_snode_id_j = snode_id_j[pred_idx_j]
                    # print("topk_j", topk_j)

                    # pred_idx_j = pred_idx_j.cpu().numpy()

                    pred_sid_list_j = g_j.nodes[pred_snode_id_j].data["raw_id"]
                    pred_logits_list_j =  g_j.nodes[pred_snode_id_j].data["p"]

                    # print("pred_logits_list_j", pred_logits_list_j)
                    # exit()
                    unode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==2)
                    uid_j = g_j.nodes[unode_id_j].data["raw_id"]

                    inode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==3)
                    iid_j = g_j.nodes[inode_id_j].data["raw_id"]
                    
                    recall_j, precision_j = get_example_recall_precision(pred_sid_list_j.cpu(), label_sid_list_j, min(topk, N))

                    # recall_list.append(recall_j)
                    # precision_list.append(precision_j)

                    # for sid_k in label_sid_list_j:
                    #     hyps_j.append(self.m_sid2swords[sid_k])

                    # for sid_k in pred_sid_list_j:
                    #     refs_j.append(self.m_sid2swords[sid_k.item()])

                    for sid_k in label_sid_list_j:
                        refs_j.append(self.m_sid2swords[sid_k])

                    for sid_k in pred_sid_list_j:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j)
                    refs_j = " ".join(refs_j)

                    if uid_j.item() == 0:
                        continue
                    print("==="*10)
                    print("user id", uid_j.item())
                    print("item id", iid_j.item())
                    print("hyps_j", hyps_j)
                    print("refs_j", refs_j)

                    scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)

                    rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                    rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                    rouge_1_p_list.append(scores_j["rouge-1"]["p"])

                    rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                    rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                    rouge_2_p_list.append(scores_j["rouge-2"]["p"])

                    rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                    rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                    rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                    bleu_scores_j = compute_bleu([refs_j], [hyps_j])
                    bleu_list.append(bleu_scores_j)


                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_bleu([refs_j], [hyps_j])

                    # bleu_1_scores_j = compute_bleu_order([refs_j], [hyps_j], order=1)
                    bleu_1_list.append(bleu_1_scores_j)

                    # bleu_2_scores_j = compute_bleu_order([refs_j], [hyps_j], order=2)
                    bleu_2_list.append(bleu_2_scores_j)

                    # bleu_3_scores_j = compute_bleu_order([refs_j], [hyps_j], order=3)
                    bleu_3_list.append(bleu_3_scores_j)

                    # bleu_4_scores_j = compute_bleu_order([refs_j], [hyps_j], order=4)
                    bleu_4_list.append(bleu_4_scores_j)
                
                exit()
        self.m_mean_eval_rouge_1_f = np.mean(rouge_1_f_list)
        self.m_mean_eval_rouge_1_r = np.mean(rouge_1_r_list)
        self.m_mean_eval_rouge_1_p = np.mean(rouge_1_p_list)

        self.m_mean_eval_rouge_2_f = np.mean(rouge_2_f_list)
        self.m_mean_eval_rouge_2_r = np.mean(rouge_2_r_list)
        self.m_mean_eval_rouge_2_p = np.mean(rouge_2_p_list)

        self.m_mean_eval_rouge_l_f = np.mean(rouge_l_f_list)
        self.m_mean_eval_rouge_l_r = np.mean(rouge_l_r_list)
        self.m_mean_eval_rouge_l_p = np.mean(rouge_l_p_list)

        self.m_mean_eval_bleu = np.mean(bleu_list)
        self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
        self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
        self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
        self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        # print("NLL_loss:%.4f"%(self.m_mean_eval_loss))
        print("rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f"%(self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r, self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r, self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        print("bleu:%.4f"%(self.m_mean_eval_bleu))
        print("bleu-1:%.4f"%(self.m_mean_eval_bleu_1))
        print("bleu-2:%.4f"%(self.m_mean_eval_bleu_2))
        print("bleu-3:%.4f"%(self.m_mean_eval_bleu_3))
        print("bleu-4:%.4f"%(self.m_mean_eval_bleu_4))

