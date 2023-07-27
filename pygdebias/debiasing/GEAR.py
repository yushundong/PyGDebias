import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling, GATConv, GINConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score
#from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm
import time
import argparse
import numpy as np
import random
import math
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE as tsn
import os
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.data import Data
import torch_geometric.utils as gm_utils



EPS = 1e-15


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        #self.gc1 = spectral_norm(GCNConv(nfeat, nhid).lin)
        
        self.gc1 = GCNConv(nfeat, nhid)
        
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class JK(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx= spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption)

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z




class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())



class CFDA(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFDA, self).__init__()
        self.type = 'GAE'
        self.h_dim = h_dim
        self.s_num = 4
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim+1, adj.shape[1]), nn.Sigmoid())
        # X
        self.base_gcn_x = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)

        # reconst_X
        self.reconst_X = nn.Sequential(nn.Linear(h_dim+1, input_dim))
        # pred_S
        self.pred_s = nn.Sequential(nn.Linear(h_dim + h_dim, self.s_num), nn.Softmax())

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.h_dim)
        if self.training and self.type == 'VGAE':
            sampled_z = gaussian_noise * torch.exp(logstd) + mean
        else:
            sampled_z = mean
        return sampled_z

    def encode_X(self, X):
        hidden = self.base_gcn_x(X)
        mean = self.gcn_mean_x(hidden)
        logstd = self.gcn_logstddev_x(hidden)
        gaussian_noise = torch.randn(X.size(0), self.h_dim)
        if self.training and self.type == 'VGAE':
            sampled_z = gaussian_noise * torch.exp(logstd) + mean
        else:
            sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        ZS = torch.cat([Z, S], dim=1)
        A_pred = self.pred_a(ZS)
        # A_pred = F.sigmoid(self.pred_a(ZS, ZS))
        # A_pred = torch.sigmoid(torch.matmul(ZS, ZS.t()))
        return A_pred

    def pred_features(self, Z, S):
        ZS = torch.cat([Z, S], dim=1)
        X_pred = self.reconst_X(ZS)
        return X_pred

    def pred_S_agg(self, Z):
        S_pred = self.pred_s(Z)
        return S_pred

    def encode(self, X):
        Z_a = self.encode_A(X)
        Z_x = self.encode_X(X)
        return Z_a, Z_x

    def pred_graph(self, Z_a, Z_x, S):
        A_pred = self.pred_adj(Z_a, S)
        X_pred = self.pred_features(Z_x, S)
        return A_pred, X_pred

    def forward(self, X, sen_idx):
        # encoder: X\S, adj -> Z
        # decoder: Z + S' -> X', A'
        S = X[:, sen_idx].view(-1, 1)
        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this dim
        Z_a, Z_x = self.encode(X_ns)
        A_pred, X_pred = self.pred_graph(Z_a, Z_x, S)
        S_agg_pred = self.pred_S_agg(torch.cat([Z_a, Z_x], dim=1))
        return A_pred, X_pred, S_agg_pred

    def loss_function(self, adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred):
        # loss_reconst
        weighted = True
        if weighted:
            weights_0 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
            weights_1 = 1 - weights_0
            assert (weights_0 > 0 and weights_1 > 0)
            weight = torch.ones_like(A_pred).reshape(-1) * weights_0  # (n x n), weight 0
            idx_1 = adj.to_dense().reshape(-1) == 1
            weight[idx_1] = weights_1

            loss_bce = nn.BCELoss(weight=weight, reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))
        else:
            loss_bce = nn.BCELoss(reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this sensitive dim
        loss_mse = nn.MSELoss(reduction='mean')
        loss_reconst_x = loss_mse(X_pred, X_ns)

        loss_ce = nn.CrossEntropyLoss()
        loss_s = loss_ce(S_agg_pred, S_agg_cat.view(-1))  # S_agg_pred: n x K, S_agg: n
        loss_result = {'loss_reconst_a': loss_reconst_a, 'loss_reconst_x': loss_reconst_x, 'loss_s': loss_s}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        par_s = list(self.pred_s.parameters())
        par_other = list(self.base_gcn.parameters()) + list(self.gcn_mean.parameters()) + list(self.gcn_logstddev.parameters()) + list(self.pred_a.parameters()) + \
                    list(self.base_gcn_x.parameters()) + list(self.gcn_mean_x.parameters()) + list(self.gcn_logstddev_x.parameters()) + list(self.reconst_X.parameters())
        optimizer_1 = optim.Adam([{'params': par_s, 'lr': lr}], weight_decay=weight_decay)  #
        optimizer_2 = optim.Adam([{'params': par_other, 'lr': lr}], weight_decay=weight_decay)  #

        self.train()
        n = X.shape[0]

        S = X[:, sen_idx].view(-1, 1)  # n x 1
        S_agg = torch.mm(adj, S) / n  # n x 1
        S_agg_max = S_agg.max()
        S_agg_min = S_agg.min()
        S_agg_cat = torch.floor(S_agg / ((S_agg_max + 0.000001 - S_agg_min) / self.s_num)).long()  # n x 1

        print("start training counterfactual augmentation module!")
        for epoch in range(500): #2000
            for i in range(3):
                optimizer_1.zero_grad()

                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_s.backward()
                optimizer_1.step()

            for i in range(5):
                optimizer_2.zero_grad()

                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_reconst_x = loss_result['loss_reconst_x']
                loss_reconst_a = loss_result['loss_reconst_a']
                #loss_reconst_a.backward()
                (-loss_s + loss_reconst_a + loss_reconst_x).backward()
                optimizer_2.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(adj, X, sen_idx, S_agg_cat)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'loss_reconst_x: {:.4f}'.format(loss_reconst_x.item()),
                      'loss_s: {:.4f}'.format(loss_s.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    save_model_path = model_path + f'weights_CFDA_{dataset}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, adj, X, sen_idx, S_agg_cat):
        self.eval()
        A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)
        eval_result = loss_result

        A_pred_binary = (A_pred > 0.5).float()  # binary
        adj_size = A_pred_binary.shape[0] * A_pred_binary.shape[1]

        sum_1 = torch.sparse.sum(adj)
        correct_num_1 = torch.sparse.sum(sparse_dense_mul(adj, A_pred_binary))  # 1
        correct_num_0 = (adj_size - (A_pred_binary + adj).sum() + correct_num_1)
        acc_a_pred = (correct_num_1 + correct_num_0) / adj_size
        acc_a_pred_0 = correct_num_0 / (adj_size - sum_1)
        acc_a_pred_1 = correct_num_1 / sum_1

        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1

        eval_result = loss_result
        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1
        return eval_result



# def get_all_node_emb(model, mask, subgraph, num_node):
#     # Obtain central node embs from subgraphs
#     node_list = np.arange(0, num_node, 1)[mask]
#     list_size = node_list.size
#     z = torch.Tensor(list_size, args.hidden_size).cuda()
#     group_nb = math.ceil(list_size / args.batch_size)  # num of batches
#     for i in range(group_nb):
#         maxx = min(list_size, (i + 1) * args.batch_size)
#         minn = i * args.batch_size
#         batch, index = subgraph.search(node_list[minn:maxx])
#         node = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
#         z[minn:maxx] = node
#     return z


# the model generates counterfactual data as augmentation(don't have to be true)
def generate_cf_data(data, sens_idx, mode=1, sens_cf=None, adj_raw=None, model_path='', train='non-test', dataset = 'running-1', device='cuda'):
    h_dim = 32
    input_dim = data.x.shape[1]
    adj = sp.coo_matrix(adj_raw.to_dense().numpy())
    indices_adj = torch.LongTensor([adj.row, adj.col])
    adj = torch.sparse_coo_tensor(indices_adj, adj.data, size=(adj.shape[0], adj.shape[1])).float()

    model_DA = CFDA(h_dim, input_dim, adj.cuda()).to(device)
    if train == 'test':
        model_DA.load_state_dict(torch.load(model_path + f'weights_CFDA_{dataset}' + '.pt'))
        # test?
        test_model = True
        if test_model:
            S = data.x[:, sens_idx].view(-1, 1)  # n x 1
            S_agg = torch.mm(adj, S) / adj.shape[0]  # n x 1
            S_agg_max = S_agg.max()
            S_agg_min = S_agg.min()
            s_num = 4
            S_agg_cat = torch.floor(S_agg / ((S_agg_max + 0.000001 - S_agg_min) / s_num)).long()  # n x 1

            eval_result = model_DA.test(adj.cuda(), data.x.cuda(), sens_idx, S_agg_cat.cuda())
            print(
                'loss_reconst_a: {:.4f}'.format(eval_result['loss_reconst_a'].item()),
                'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
            )
    else:
        model_DA.train_model(data.x.cuda(), adj.cuda(), sens_idx, dataset, model_path=model_path, lr=0.0001, weight_decay=1e-5)

    # generate cf for whole graph to achieve better efficiency
    Z_a, Z_x = model_DA.encode(data.x.cuda())
    adj_update, x_update = model_DA.pred_graph(Z_a, Z_x, sens_cf.view(-1,1).cuda())

    # hybrid
    w_hd_x = 0.99
    thresh_a = 0.9

    data_cf = data.clone()
    data_cf.x = (w_hd_x * data.x + (1 - w_hd_x) * x_update.cpu())
    data_cf.x[:, sens_idx] = sens_cf

    adj_cf = adj.to_dense().clone()
    adj_cf[adj_update > thresh_a] = 1  # to binary
    adj_cf[adj_update < 1 - thresh_a] = 0
    rate_na = (adj.to_dense() != adj_cf).sum() / (adj.shape[0] * adj.shape[0])
    print('rate of A change: ', rate_na)

    edge_index_cf = adj_cf.to_sparse().indices().cpu()
    data_cf.edge_index = edge_index_cf

    return data_cf


# def show_rep_distri(adj, model, data, subgraph, labels, sens, idx_select):
#     model.eval()
#     n = len(labels)
#     idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)  # size = n, bool
#
#     adj_norm = normalize(adj, norm='l1', axis=1)
#     nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors
#     nb_sens_ave[np.where(nb_sens_ave < 0.5)] = 0
#     nb_sens_ave[np.where(nb_sens_ave >= 0.5)] = 1
#     nb_sens_ave = nb_sens_ave.reshape(-1)
#
#     emb = get_all_node_emb(model, idx_select_mask, subgraph, n).cpu().detach().numpy()
#     idx_emb_0 = np.where(nb_sens_ave[idx_select] == 0)
#     idx_emb_1 = np.where(nb_sens_ave[idx_select] == 1)
#
#     fig, ax = plt.subplots()
#     point_size = 8
#     Zt_tsn = tsn(n_components=2).fit_transform(emb)  # m x d => m x 2
#     ax.scatter(Zt_tsn[idx_emb_0, 0], Zt_tsn[idx_emb_0, 1], point_size, marker='o', color='r')  # cluster k
#     ax.scatter(Zt_tsn[idx_emb_1, 0], Zt_tsn[idx_emb_1, 1], point_size, marker='o', color='b')  # cluster k
#     # ax.scatter(Zt_tsn[k - num_cluster, 0], Zt_tsn[k - num_cluster, 1], centroid_size, marker='D',
#       #             color=cluster_color[k])  # centroid
#         # plt.xlim(-100, 100)
#
#     # plt.show()
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#
#     #plt.savefig('./' + args.dataset + '_zt_cluster.tsne.pdf', bbox_inches='tight')
#     plt.show()
#     return





class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                base_model='sage', k: int = 2):
        super(Encoder, self).__init__()
        self.conv = None
        self.base_model = base_model
        if self.base_model == 'gcn':
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == 'gin':
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == 'sage':
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == 'infomax':
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == 'jk':
            self.conv = JK(in_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        return x


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


class PPR:
    # Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()

    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)

        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])

        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else:
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]

        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0

        return neighbor

    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else:
            print('File of node {} exists.'.format(seed))

    def search_all(self, node_num, path):
        neighbor = {}
        if os.path.isfile(path + '_neighbor') and os.stat(path + '_neighbor').st_size != 0:
            print("Exists neighbor file")
            neighbor = torch.load(path + '_neighbor')
        else:
            print("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))

            print("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path + '_neighbor')
            os.system('rm -r {}'.format(path))
            print("Finish Writing")
        return neighbor



class Subgraph:
    # Class for subgraph extraction

    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x



        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize

        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])),
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def adjust_edge(self, idx):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def build(self, postfix=""):



        # Extract subgraphs for all nodes
        #if os.path.isfile(self.path + '_subgraph' + postfix) and os.stat(
        #        self.path + '_subgraph' + postfix).st_size != 0:
        #    print("loading subgraphs from" + self.path + '_subgraph' + postfix)
        #    self.subgraph = torch.load(self.path + '_subgraph' + postfix)
        #    return


        self.neighbor = self.ppr.search_all(self.node_num, self.path)

        self.process_adj_list()


        for i in range(self.node_num):  # for every node in the graph
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)


            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path + '_subgraph' + postfix)  # JM: subgraph[i]: center node i's subgraph
        print('save subgraphs in ' + self.path + '_subgraph' + postfix)

    def search(self, node_list):
        # Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)  # JM: index of the first node of each subgraph in batch
        batch = Batch().from_data_list(batch)  # JM: All the nodes in the [node_list] subgraphs
        return batch, index





def generate_cf_true_synthetic(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, raw_data_info=None):
    # generate graphs in sens_rate_list
    embedding = raw_data_info['z']
    v = raw_data_info['v']
    feat_idxs = raw_data_info['feat_idxs']
    w = raw_data_info['w']
    w_s = raw_data_info['w_s']
    n = data.x.shape[0]
    adj_orin = raw_data_info['adj']
    alpha = raw_data_info['alpha']
    oa = 0.9

    for i in range(len(sens_rate_list)):
        sens_rate = sens_rate_list[i]
        sampled_idx = random.sample(range(n), int(sens_rate * n))
        data_cf = data.clone()
        data_cf.x[:, sens_idx] = 0
        data_cf.x[sampled_idx, sens_idx] = 1

        sens = data_cf.x[:, sens_idx].numpy()

        # x\s
        features_cf = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1, 1), v))  # (n x dim) + (1 x dim) -> n x dim
        features_cf = torch.FloatTensor(features_cf)
        data_cf.x[:, 1:] = oa * data_cf.x[:, 1:] + (1 - oa) * features_cf

        # adj
        sens_sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):  # i<=j
                if i == j:
                    sens_sim[i][j] = 1
                    continue
                sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])

        similarities = cosine_similarity(embedding)  # n x n
        adj = similarities + alpha * sens_sim

        adj = oa * adj_orin + (1 - oa) * adj

        adj[np.where(adj >= 0.4)] = 1
        adj[np.where(adj < 0.4)] = 0
        adj = sp.csr_matrix(adj)

        data_cf.edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

        # skip y
        # adj_norm = normalize(adj, norm='l1', axis=1)
        # nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors
        #
        # labels = np.matmul(embedding, w) + w_s * nb_sens_ave.reshape(-1, 1)  # n x 1
        # labels = labels.reshape(-1)
        #
        # labels = oa * data_cf.y.numpy() + (1-oa)* labels
        # labels_mean = np.mean(labels)
        # labels_binary = np.zeros_like(labels)
        # labels_binary[np.where(labels > labels_mean)] = 1.0
        # data_cf.y = torch.FloatTensor(labels_binary)

        data_results = {'data': data_cf}

        # save in files
        if save_file:
            with open(save_path + '/' + dataset + '_cf_' + str(sens_rate) + '.pkl', 'wb') as f:
                pickle.dump(data_results, f)
                print('saved counterfactual data: ', dataset + '_cf_' + str(sens_rate) + '.pkl')
    return


class CFGT(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFGT, self).__init__()
        self.h_dim = h_dim
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim, adj.shape[1]))

        # S
        self.sf = nn.Sequential(nn.Linear(1, 1))  # n x 1, parameter: 1

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        A_pred = self.pred_a(Z)  # n x n
        S_rep_f = self.sf(S)
        S_rep_cf = self.sf(1 - S)

        s_match = (torch.matmul(S_rep_f, S_rep_f.t()) + torch.matmul(S_rep_cf, S_rep_cf.t())) / 2
        A_pred = F.sigmoid(A_pred + s_match)
        return A_pred

    def encode(self, X):
        Z_a = self.encode_A(X)
        return Z_a

    def pred_graph(self, Z_a, S):
        A_pred = self.pred_adj(Z_a, S)
        return A_pred

    def forward(self, X, sen_idx):
        # encoder: X\S, adj -> Z
        # decoder: Z + S' -> A'
        S = X[:, sen_idx].view(-1, 1)
        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this dim

        Z_a = self.encode(X_ns)
        A_pred = self.pred_graph(Z_a, S)
        return A_pred

    def loss_function(self, adj, A_pred):
        # loss_reconst
        weighted = True
        if weighted:
            weights_0 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
            weights_1 = 1 - weights_0
            assert (weights_0 > 0 and weights_1 > 0)
            weight = torch.ones_like(A_pred).reshape(-1) * weights_0  # (n x n), weight 0
            idx_1 = adj.to_dense().reshape(-1) == 1
            weight[idx_1] = weights_1

            loss_bce = nn.BCELoss(weight=weight, reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))
        else:
            loss_bce = nn.BCELoss(reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        loss_result = {'loss_reconst_a': loss_reconst_a}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        optimizer = optim.Adam([{'params': self.parameters(), 'lr': lr}], weight_decay=weight_decay)

        self.train()
        n = X.shape[0]

        print("start training counterfactual augmentation module!")
        for epoch in range(2000):
            optimizer.zero_grad()

            A_pred = self.forward(X, sen_idx)
            loss_result = self.loss_function(adj, A_pred)

            # backward propagation
            loss_reconst_a = loss_result['loss_reconst_a']
            loss_reconst_a.backward()
            optimizer.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(X, adj, sen_idx)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    save_model_path = model_path + f'weights_CFGT_{dataset}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, X, adj, sen_idx):
        self.eval()
        A_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, A_pred)
        eval_result = loss_result

        A_pred_binary = (A_pred > 0.5).float()  # binary
        adj_size = A_pred_binary.shape[0] * A_pred_binary.shape[1]

        sum_1 = torch.sparse.sum(adj)
        correct_num_1 = torch.sparse.sum(sparse_dense_mul(adj, A_pred_binary))  # 1
        correct_num_0 = (adj_size - (A_pred_binary + adj).sum() + correct_num_1)
        acc_a_pred = (correct_num_1 + correct_num_0) / adj_size
        acc_a_pred_0 = correct_num_0 / (adj_size - sum_1)
        acc_a_pred_1 = correct_num_1 / sum_1

        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1
        return eval_result


# get true cf for real-world datasets
def generate_cf_true_rw(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, train='test', raw_data_info=None):
    n = data.x.shape[0]
    input_dim = data.x.shape[1]
    h_dim = 32
    w_hd_x = 0.95
    thresh_a = 0.9
    adj_orin = raw_data_info['adj']
    adj = adj_orin.tocoo()
    indices_adj = torch.LongTensor([adj.row, adj.col])
    adj = torch.sparse_coo_tensor(indices_adj, adj.data, size=(adj.shape[0], adj.shape[1])).float()

    for i in range(len(sens_rate_list)):
        sens_rate = sens_rate_list[i]
        sampled_idx = random.sample(range(n), int(sens_rate * n))
        data_cf = data.clone()

        sens_new = torch.zeros_like(data_cf.x[:, sens_idx])
        sens_new[sampled_idx] = 1

        # X
        sens = data.x[:, sens_idx]
        idx_1 = (sens == 1)
        idx_0 = (sens == 0)
        x_mean_1 = data.x[idx_1, :]  # n1 x d
        x_mean_0 = data.x[idx_0, :]
        x_mean_diff = x_mean_1.mean(dim=0) - x_mean_0.mean(dim=0)  # d
        x_update = ((sens_new - sens).view(-1, 1).tile(1, x_mean_1.shape[1]) * x_mean_diff.view(1, -1).tile(n, 1))
        data_cf.x = w_hd_x * data.x + (1-w_hd_x) * x_update

        # S
        data_cf.x[:, sens_idx] = sens_new

        # adj
        model_GT = CFGT(h_dim, input_dim, adj.cuda()).cuda()
        if train == 'test':  # train or load existing model
            model_GT.load_state_dict(torch.load(save_path + f'weights_CFGT_{dataset}' + '.pt'))
            # test?
            test_model = False
            if test_model:
                eval_result = model_GT.test(data.x.cuda(), adj.cuda(), sens_idx)
                print(
                    'loss_reconst_a: {:.4f}'.format(eval_result['loss_reconst_a'].item()),
                    'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                    'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                    'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                )
        else:
            model_GT.train_model(data.x.cuda(), adj.cuda(), sens_idx, dataset, model_path=save_path, lr=0.0001, weight_decay=1e-5)

        # generate cf for whole graph to achieve better efficiency
        Z_a = model_GT.encode(data_cf.x.cuda())
        adj_update = model_GT.pred_graph(Z_a, sens_new.view(-1, 1).cuda())
        adj_cf = adj.to_dense().clone()
        adj_cf[adj_update > thresh_a] = 1  # to binary
        adj_cf[adj_update < 1 - thresh_a] = 0
        rate_na = (adj.to_dense() != adj_cf).sum() / (n * n)
        print('rate of A change: ', rate_na)

        edge_index_cf = adj_cf.to_sparse().indices().cpu()
        data_cf.edge_index = edge_index_cf

        # skip y, as y is not used

        # data_cf
        data_results = {'data': data_cf}
        # save in files
        if save_file:
            with open(save_path + '/' + dataset + '_cf_' + str(sens_rate) + '.pkl', 'wb') as f:
                pickle.dump(data_results, f)
                print('saved counterfactual data: ', dataset + '_cf_' + str(sens_rate) + '.pkl')

    return

def generate_cf_true(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, raw_data_info=None, mode=1):
    n = data.x.shape[0]
    if dataset == 'synthetic':
        generate_cf_true_synthetic(data, dataset, sens_rate_list, sens_idx, save_path, save_file=save_file, raw_data_info=raw_data_info)
        return
    else:
        generate_cf_true_rw(data, dataset, sens_rate_list, sens_idx, save_path, save_file=save_file, raw_data_info=raw_data_info)

    return



def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()



def get_all_node_emb(model, mask, subgraph, num_node, hidden_size, batch_size):
    # Obtain central node embs from subgraphs
    node_list = np.arange(0, num_node, 1)[mask]
    list_size = node_list.size
    z = torch.Tensor(list_size, hidden_size).cuda()
    group_nb = math.ceil(list_size / batch_size)  # num of batches
    for i in range(group_nb):
        maxx = min(list_size, (i + 1) * batch_size)
        minn = i * batch_size
        batch, index = subgraph.search(node_list[minn:maxx])
        node = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        z[minn:maxx] = node
    return z



def compute_loss_sim(model, subgraph, cf_subgraph, idx_select, n, sim_coeff, hidden_size, batch_size, z1=None, z2=None):
    idx_select_mask = ((torch.zeros(n)).scatter_(0, idx_select, 1) > 0)
    if z1 is None:
        z1 = get_all_node_emb(model, idx_select_mask, subgraph, n, hidden_size, batch_size)
    if z2 is None:
        z2 = get_all_node_emb(model, idx_select_mask, cf_subgraph, n, hidden_size, batch_size)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1, p2)/2
    l2 = model.D(h2, p1)/2
    sim_loss = sim_coeff*(l1+l2)

    return sim_loss



def compute_loss(model, subgraph, cf_subgraph_list, labels, idx_select, n, sim_coeff, hidden_size, batch_size, device):
    idx_select_mask = ((torch.zeros(n)).scatter_(0, idx_select, 1) > 0)
    z1 = get_all_node_emb(model, idx_select_mask, subgraph, n, hidden_size, batch_size)
    # classifier
    c1 = model.classifier(z1)

    # Binary Cross-Entropy
    l1 = F.binary_cross_entropy_with_logits(c1, labels[idx_select].unsqueeze(1).float().to(device)) / 2
    loss_c = (1 - sim_coeff) * l1

    loss_sim = 0.0
    for si in range(len(cf_subgraph_list)):
        cf_subgraph = cf_subgraph_list[si]
        z2 = get_all_node_emb(model, idx_select_mask, cf_subgraph, n, hidden_size, batch_size)
        loss_sim_si = compute_loss_sim(model, subgraph, cf_subgraph, idx_select, n, sim_coeff, hidden_size, batch_size, z1, z2)
        loss_sim += loss_sim_si
    loss_sim /= len(cf_subgraph_list)

    loss_result = {'loss_c': loss_c, 'loss_s': loss_sim, 'loss': loss_sim+loss_c}

    return loss_result



def evaluate(model, data, subgraph, cf_subgraph_list, labels, sens, idx_select, n, sim_coeff, hidden_size, batch_size, device='cuda', type='all'):
    loss_result = compute_loss(model, subgraph, cf_subgraph_list, labels, idx_select, n, sim_coeff, hidden_size, batch_size, device)
    if type == 'easy':
        eval_results = {'loss': loss_result['loss'], 'loss_c': loss_result['loss_c'], 'loss_s': loss_result['loss_s']}

    elif type == 'all':
        n = len(labels)
        idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)  # size = n, bool

        # performance
        emb = get_all_node_emb(model, idx_select_mask, subgraph, n, hidden_size, batch_size)
        output = model.forwarding_predict(emb)
        output_preds = (output.squeeze() > 0).type_as(labels)

        auc_roc = roc_auc_score(labels.cpu().numpy()[idx_select], output.detach().cpu().numpy())
        f1_s = f1_score(labels[idx_select].cpu().numpy(), output_preds.cpu().numpy())
        acc = accuracy_score(labels[idx_select].cpu().numpy(), output_preds.cpu().numpy())

        # fairness
        parity, equality = fair_metric(output_preds.cpu().numpy(), labels[idx_select].cpu().numpy(),
                                       sens[idx_select].numpy())
        # counterfactual fairness
        cf = 0.0
        for si in range(len(cf_subgraph_list)):
            cf_subgraph = cf_subgraph_list[si]
            emb_cf = get_all_node_emb(model, idx_select_mask, cf_subgraph, n, hidden_size, batch_size)
            output_cf = model.forwarding_predict(emb_cf)
            output_preds_cf = (output_cf.squeeze() > 0).type_as(labels)

            cf_si = 1 - (output_preds.eq(output_preds_cf).sum().item() / idx_select.shape[0])
            cf += cf_si
        cf /= len(cf_subgraph_list)

        eval_results = {'acc': acc, 'auc': auc_roc, 'f1': f1_s, 'parity': parity, 'equality': equality, 'cf': cf,
                        'loss': loss_result['loss'], 'loss_c': loss_result['loss_c'], 'loss_s': loss_result['loss_s']}  # counterfactual_fairness
    return eval_results



def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict


def stats_cov(data1, data2):
    '''
    :param data1: np, n x d1
    :param data2: np, n x d2
    :return:
    '''
    cov = np.cov(data1, data2)  # (d_1 + d_2) x (d_1 + d_2)

    # only when data1 and data 2 are both shaped as (n,)
    corr_pear, p_value = pearsonr(data1, data2)

    # R^2
    node_num = len(data2)
    X = data2.reshape(node_num, -1)
    reg = LinearRegression().fit(X, data1)
    y_pred = reg.predict(X)
    R2 = r2_score(data1, y_pred)
    print('R-square', R2)

    result = {
        'cov': cov,
        'pearson': corr_pear,
        'pear_p_value': p_value,
        'R-square': R2
    }
    return result





def analyze_dependency(sens, adj, ypred_tst, idx_select, type='mean'):
    if type == 'mean':
        # row-normalize
        adj_norm = normalize(adj.to_dense().numpy(), norm='l1', axis=1)
        nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

        # S_N(i), Y_i | S_i
        cov_nb_results = stats_cov(ypred_tst, nb_sens_ave[idx_select])
        # print('correlation between Y and neighbors (not include self)\' S:', cov_nb_results)

    return cov_nb_results


class GEAR(torch.nn.Module):
    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, hidden_size=1024, proj_hidden=16, num_class=1, encoder_hidden_size=1024, encoder_base_model='gcn', experiment_type='train'):
        super(GEAR, self).__init__()
        self.encoder = Encoder(features.shape[1], encoder_hidden_size, base_model=encoder_base_model)
        self.hidden_size = hidden_size
        self.num_proj_hidden = proj_hidden
        self.num_class = num_class
        self.experiment_type = experiment_type

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, self.num_proj_hidden)),
            nn.BatchNorm1d(self.num_proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(self.num_proj_hidden, self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size)
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(self.hidden_size, self.hidden_size))

        # Classifier
        self.c1 = Classifier(ft_in=self.hidden_size, nb_classes=num_class)

        for m in self.modules():
            self.weights_init(m)

        self.reset_parameters()

        self.preprocess(adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)

    def reset_parameters(self):
        reset(self.encoder)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """

        #print(x.shape)
        hidden = self.encoder(x, edge_index)  # node representation：(No. of subgraphs (=batch size) x subgraph size) x hidden_size



        if index is None:
            return hidden

        z = hidden[index]  # JM: center node, batch_size x hidden_size
        return z
      


    def preprocess(self, adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, subgraph_size=30, n_order=10, dataset="None", raw_data_info=None):

        ##  to be adapted
        data_path_root = '../'
        self.model_path = 'models_save/'
        # self.model_path = 'graphFair_subgraph/cf/'
        self.labels = labels
        self.sens = sens
        self.adj = adj
        self.n = features.shape[0]

        if raw_data_info is None:
            raw_data_info = {'adj': adj}


        # must sorted in ascending order
        self.idx_train, _ = torch.sort(idx_train)
        self.idx_val, _ = torch.sort(idx_val)
        self.idx_test, _ = torch.sort(idx_test)

        edge_index = torch.tensor(adj.to_dense().nonzero(), dtype=torch.long)

        # preprocess the input
        n = features.shape[0]
        self.data = Data(x=features, edge_index=edge_index)
        self.data.y = labels  # n

        # ============== generate counterfactual data (ground-truth) ================
        if self.experiment_type == 'cf':
            sens_rate_list = [0, 0.5, 1.0]
            path_truecf_data = 'graphFair_subgraph/cf/'
            generate_cf_true(self.data, dataset, sens_rate_list, sens_idx, path_truecf_data, save_file=True,
                                raw_data_info=raw_data_info)  # generate
            sys.exit()  # stop here

        #num_node = self.data.x.size(0)

        # Subgraph: Setting up the subgraph extractor
        self.subgraph_size = subgraph_size
        self.ppr_path = './graphFair_subgraph/' + dataset
        self.n_order = n_order

        self.subgraph = Subgraph(self.data.x, self.data.edge_index, self.ppr_path, self.subgraph_size, n_order)
        self.subgraph.build()


        # counterfactual graph generation (may not true)
        self.cf_subgraph_list = []
        subgraph_load = False
        if subgraph_load:
            if not os.path.exists(f'graphFair_subgraph/aug/'):
                        os.makedirs(f'graphFair_subgraph/aug/')
            path_cf_ag = 'graphFair_subgraph/aug/' + f'{dataset}_cf_aug_' + str(0) + '.pkl'
            with open(path_cf_ag, 'rb') as f:
                data_cf = pickle.load(f)['data_cf']
                print('loaded counterfactual augmentation data from: ' + path_cf_ag)
        else:
            sens_cf = 1 - self.data.x[:, sens_idx]
            data_cf = generate_cf_data(self.data, sens_idx, mode=1, sens_cf=sens_cf, adj_raw=adj,
                                    model_path=self.model_path, dataset=dataset)  #
            if not os.path.exists(f'graphFair_subgraph/aug/'):
                        os.makedirs(f'graphFair_subgraph/aug/')
            path_cf_ag = 'graphFair_subgraph/aug/' + f'{dataset}_cf_aug_' + str(0) + '.pkl'
            with open(path_cf_ag, 'wb') as f:
                data_cf_save = {'data_cf': data_cf}
                pickle.dump(data_cf_save, f)
                print('saved counterfactual augmentation data in: ', path_cf_ag)


        cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, self.ppr_path, self.subgraph_size, n_order)
        cf_subgraph.build(postfix='_cf' + str(0))
        self.cf_subgraph_list.append(cf_subgraph)

        # add more augmentation if wanted
        subgraph_load = False  # True
        sens_rate_list = [0.0, 1.0]
        for si in range(len(sens_rate_list)):
            sens_rate = sens_rate_list[si]
            sampled_idx = random.sample(range(n), int(sens_rate * n))
            sens_cf = torch.zeros(n)
            sens_cf[sampled_idx] = 1.
            if subgraph_load:
                path_cf_ag = 'graphFair_subgraph/aug/' + f'{dataset}_cf_aug_' + str(si + 1) + '.pkl'
                with open(path_cf_ag, 'rb') as f:
                    data_cf = pickle.load(f)['data_cf']
                    print('loaded counterfactual augmentation data from: ' + path_cf_ag)
            else:
                data_cf = generate_cf_data(self.data, sens_idx, mode=0, sens_cf=sens_cf, adj_raw=adj,
                                        model_path=self.model_path)  #
                path_cf_ag = 'graphFair_subgraph/aug/' + f'{dataset}_cf_aug_' + str(si + 1) + '.pkl'
                with open(path_cf_ag, 'wb') as f:
                    data_cf_save = {'data_cf': data_cf}
                    pickle.dump(data_cf_save, f)
                    print('saved counterfactual augmentation data in: ', path_cf_ag)

            cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, self.ppr_path, self.subgraph_size, n_order)
            cf_subgraph.build(postfix='_cf' + str(si + 1))
            self.cf_subgraph_list.append(cf_subgraph)



    def fit(self, epochs=500, lr=0.001, batch_size=100, weight_decay=1e-5, sim_coeff=0.6, encoder_name="None", dataset_name="None", device='cuda'):

        self.batch_size = batch_size

        # Setting up the model and optimizer
        # model = self.GraphCF(encoder=Encoder(self.data.num_features, encoder_hidden_size, base_model=encoder_base_model), args=args, num_class=encoder_num_class).to(device)
        print(epochs, lr, batch_size, weight_decay, sim_coeff, encoder_name, dataset_name, device)
        par_1 = list(self.encoder.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(
            self.fc3.parameters()) + list(self.fc4.parameters())
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        optimizer_1 = torch.optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
        optimizer_2 = torch.optim.Adam(par_2, lr=lr, weight_decay=weight_decay)

        # cuda
        if device == 'cuda':
            self = self.to(device)

        # train(args.epochs, model, optimizer_1, optimizer_2, data, subgraph, cf_subgraph_list, idx_train, idx_val, idx_test, exp_i)

        print("start training!")
        best_loss = 100
        labels = self.data.y
        for epoch in range(epochs + 1):
            sim_loss = 0
            cl_loss = 0
            rep = 1
            for _ in range(rep):
                self.train()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()

                # sample central node
                sample_size = min(batch_size, len(self.idx_train))
                sample_idx = random.sample(list(self.idx_train.cpu().numpy()),
                                           sample_size)  # select |batch size| central nodes

                # forward: factual subgraph
                batch, index = self.subgraph.search(sample_idx)


                z = self(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())  # center node rep, subgraph rep
                #assert 1==0
                # projector
                p1 = self.projection(z)
                # predictor
                h1 = self.prediction(p1)

                # forward: counterfactual subgraph
                sim_loss_smps = 0.0
                for si in range(len(self.cf_subgraph_list)):
                    cf_subgraph = self.cf_subgraph_list[si]



                    batch_cf, index_cf = cf_subgraph.search(sample_idx)


                    z_cf = self.forward(batch_cf.x.cuda(), batch_cf.edge_index.cuda(), batch_cf.batch.cuda(),
                                 index_cf.cuda())  # center node rep, subgraph rep

                    # projector
                    p2 = self.projection(z_cf)
                    # predictor
                    h2 = self.prediction(p2)

                    l1 = self.D(h1, p2) / 2  # cosine similarity
                    l2 = self.D(h2, p1) / 2
                    sim_loss_smps += sim_coeff * (l1 + l2)  # similarity loss
                sim_loss_smps /= len(self.cf_subgraph_list)
                sim_loss += sim_loss_smps

            (sim_loss / rep).backward(retain_graph=True)
            optimizer_1.step()

            # classifier
            z = self.forward(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(),
                      index.cuda())  # center node rep, subgraph rep
            c1 = self.classifier(z)

            # Binary Cross-Entropy
            l3 = F.binary_cross_entropy_with_logits(c1, labels[sample_idx].unsqueeze(1).float().to(device)) / 2

            cl_loss = (1 - sim_coeff) * l3
            cl_loss.backward()
            optimizer_2.step()
            loss = (sim_loss / rep + cl_loss)

            # Validation
            # model.eval()
            # eval_results_trn = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_train)
            # eval_results_val = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_val)
            # eval_results_tst = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_test)
            if epoch % 100 == 0:
                self.eval()
                eval_results_trn = evaluate(self, self.data, self.subgraph, self.cf_subgraph_list, labels, self.sens, self.idx_train, self.n, sim_coeff, self.hidden_size, batch_size, device)
                eval_results_val = evaluate(self, self.data, self.subgraph, self.cf_subgraph_list, labels, self.sens, self.idx_val, self.n, sim_coeff, self.hidden_size, batch_size, device)
                print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss / rep):.4f} | train_c_loss: {cl_loss:.4f} | "
                      f"trn_loss: {eval_results_trn['loss']:.4f} |"
                      f"trn_acc: {eval_results_trn['acc']:.4f} | trn_auc_roc: {eval_results_trn['auc']:.4f} | trn_F1: {eval_results_trn['f1']:.4f} | "
                      f"trn_Parity: {eval_results_trn['parity']:.4f} | trn_Equality: {eval_results_trn['equality']:.4f} | trn_CounterFactual Fairness: {eval_results_trn['cf']:.4f} |"
                      f"val_loss: {eval_results_val['loss']:.4f} |"
                      f"val_acc: {eval_results_val['acc']:.4f} | val_auc_roc: {eval_results_val['auc']:.4f} | val_F1: {eval_results_val['f1']:.4f} | "
                      f"val_Parity: {eval_results_val['parity']:.4f} | val_Equality: {eval_results_val['equality']:.4f} | val_CounterFactual Fairness: {eval_results_val['cf']:.4f} |"
                      # f"tst_loss: {eval_results_tst['loss']:.4f} |"
                      # f"tst_acc: {eval_results_tst['acc']:.4f} | tst_auc_roc: {eval_results_tst['auc']:.4f} | tst_F1: {eval_results_tst['f1']:.4f} | "
                      # f"tst_Parity: {eval_results_tst['parity']:.4f} | tst_Equality: {eval_results_tst['equality']:.4f} | tst_CounterFactual Fairness: {eval_results_tst['cf']:.4f} |"
                      )

                val_c_loss = eval_results_val['loss_c']
                val_s_loss = eval_results_val['loss_s']
                if (val_c_loss + val_s_loss) < best_loss:
                    # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                    self.val_loss=val_c_loss.item()+val_s_loss.item()
                    best_loss = val_c_loss + val_s_loss
                    torch.save(self.state_dict(),
                               f'models_save/weights_graphCF_{encoder_name}_{dataset_name}_exp' + '.pt')


    def predict(self, encoder_name="None", dataset_name="None", batch_size=100, sim_coeff=0.6):

        results_all_exp = {}

        # ========= test all ===========
        # evaluate on the best model, we use TRUE CF graphs
        sens_rate_list = [0.0, 1.0]
        path_true_cf_data = 'graphFair_subgraph/aug'  # no '/'

        # model = models.GraphCF(encoder=models.Encoder(data.num_features, args.hidden_size, base_model=args.encoder),
        #                        args=args, num_class=num_class).to(device)
        self.load_state_dict(
            torch.load(self.model_path + f'weights_graphCF_{encoder_name}_{dataset_name}_exp' + '.pt'))


        # eval_results = test(model, adj, data, args.dataset, subgraph, cf_subgraph_list, labels, sens, path_true_cf_data,
                            # sens_rate_list, idx_test)

        #
        self.eval()
        eval_results_orin = evaluate(self, self.data, self.subgraph, self.cf_subgraph_list, self.labels, self.sens, self.idx_test, self.n, sim_coeff, self.hidden_size, batch_size)

        n = len(self.data.y)
        idx_select_mask = (torch.zeros(n).scatter_(0, self.idx_test, 1) > 0)  # size = n, bool
        # performance
        emb = get_all_node_emb(self, idx_select_mask, self.subgraph, n, self.hidden_size, batch_size)
        output = self.forwarding_predict(emb)
        output_preds = (output.squeeze() > 0).type_as(self.data.y)

        cf_score = []
        # counterfactual fairness -- true
        for i in range(len(sens_rate_list)):
            # load cf-true data
            sens_rate = int(sens_rate_list[i])
            post_str = dataset_name + '_cf_' + str(sens_rate)
            file_path = path_true_cf_data + '/' + dataset_name + '_cf_aug_' + str(sens_rate) + '.pkl'

            with open(file_path, 'rb') as f:
                data_cf = pickle.load(f)['data_cf']
                print('loaded data from: ' + file_path)

            cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, self.ppr_path, self.subgraph_size, self.n_order)  # true
            cf_subgraph.build(postfix='true_cf' + post_str)  # true_cf_0.3

            n = len(data_cf.y)
            # performance
            emb_cf = get_all_node_emb(self, idx_select_mask, cf_subgraph, n, self.hidden_size, batch_size)
            output_cf = self.forwarding_predict(emb_cf)
            output_preds_cf = (output_cf.squeeze() > 0).type_as(data_cf.y)

            # compute how many labels are changed in counterfactual world
            cf_score_cur = (output_preds != output_preds_cf).sum()
            cf_score_cur = float(cf_score_cur.item()) / len(output_preds)
            cf_score.append(cf_score_cur)

        ave_cf_score = sum(cf_score) / len(cf_score)

        cf_eval_dict = {'ave_cf_score': ave_cf_score, 'cf_score': cf_score}

        # r-square
        col_ypred_s_summary = analyze_dependency(self.sens.cpu().numpy(), self.adj, output_preds.cpu().numpy(), self.idx_test,
                                                 type='mean')

        eval_results = dict(eval_results_orin, **cf_eval_dict)  # counterfactual_fairness

        eval_results['R-square'] = col_ypred_s_summary['R-square']
        eval_results['pearson'] = col_ypred_s_summary['pearson']

        results_all_exp = add_list_in_dict('Accuracy', results_all_exp, eval_results['acc'])
        results_all_exp = add_list_in_dict('F1-score', results_all_exp, eval_results['f1'])
        results_all_exp = add_list_in_dict('auc_roc', results_all_exp, eval_results['auc'])
        results_all_exp = add_list_in_dict('Equality', results_all_exp, eval_results['equality'])
        results_all_exp = add_list_in_dict('Parity', results_all_exp, eval_results['parity'])
        results_all_exp = add_list_in_dict('ave_cf_score', results_all_exp, eval_results['ave_cf_score'])
        results_all_exp = add_list_in_dict('CounterFactual Fairness', results_all_exp, eval_results['cf'])
        results_all_exp = add_list_in_dict('R-square', results_all_exp, eval_results['R-square'])

        print('============================= Overall =============================================')
        for k in results_all_exp:
            results_all_exp[k] = np.array(results_all_exp[k])
            print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")



        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average='micro')
        ACC = accuracy_score(labels[idx_test], output_preds, )
        AUCROC = roc_auc_score(labels[idx_test], output_preds)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(output_preds,
                                                                                                       idx_test)

        SP, EO = self.fair_metric(output_preds, self.labels[idx_test].detach().cpu().numpy(),
                                  self.sens[idx_test].detach().cpu().numpy())


        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    def D(self, x1, x2): # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1

        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)

        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

        return parity.item(), equality.item()
    def predict_sens_group(self, output, idx_test):
        #pred = self.lgreg.predict(self.embs[idx_test])
        pred=output
        result=[]
        for sens in [0,1]:
            F1 = f1_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), pred[self.sens[idx_test]==sens], average='micro')
            ACC=accuracy_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), pred[self.sens[idx_test]==sens],)
            AUCROC=roc_auc_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), pred[self.sens[idx_test]==sens])
            result.extend([ACC, AUCROC, F1])

        return result

    def forwarding_predict(self, emb):
        # projector
        p1 = self.projection(emb)
        # predictor
        h1 = self.prediction(p1)
        # classifier
        c1 = self.classifier(emb)

        return c1

