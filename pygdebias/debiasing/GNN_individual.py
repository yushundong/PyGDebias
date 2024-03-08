import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    DeepGraphInfomax,
    JumpingKnowledge,
)

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert

import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj, convert
from scipy.sparse.csgraph import laplacian
from torch_geometric.nn import GCNConv, JumpingKnowledge
import sklearn.preprocessing as skpp
import scipy.sparse as sp
import pickle


def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(
        1, x_corresponding.shape[1]
    )
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = (c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)

    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(
            x_corresponding,
            new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i],
        )
    the_range = (
        torch.arange(0.0, x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1)
        + 1
    )
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    # print("Now Average ERR@k = ", final.item())

    return final.item()


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter() - t
    return features, precompute_time


def simi(output):  # new_version

    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a == 0, the_ones, a)
    a_norm = output / a
    b_norm = output / a

    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res


def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = (
        torch.log2(
            2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)
        )
        .repeat(x_sorted_scores.shape[0], 1)
        .cuda()
    )
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = (
        torch.log2(
            2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)
        )
        .repeat(x_sorted_scores.shape[0], 1)
        .cuda()
    )
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# by rows
def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(
        2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)
    ).cuda()
    final = numerator / denominator

    return torch.sum(final)


# by rows
def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(
        2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float)
    )
    final = numerator / denominator

    return torch.sum(final)


def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)


def err_computation(score_rank, top_k):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = ((c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(
            score_rank, new_score_rank[-score_rank.shape[0] - 1 - i : -1 - i]
        )
    the_range = torch.arange(0.0, score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]

    return torch.sum(final)


def err_exchange_abs(x_corresponding, j, k, top_k):
    new_score_rank = x_corresponding
    err1 = err_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    err2 = err_computation(new_score_rank, top_k)

    return torch.abs(err1 - err2)


def lambdas_computation(x_similarity, y_similarity, top_k, k_para, sigma_1):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = (
        max_num * torch.ones_like(x_similarity[0, :])
    )
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = (
        max_num * torch.ones_like(y_similarity[0, :])
    )

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = (
        torch.arange(y_similarity.shape[0])
        .view(y_similarity.shape[0], 1)
        .repeat(1, y_similarity.shape[0])
    )
    y_ranks[the_row, y_sorted_idxs] = (
        1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    )

    # ***************************** pairwise delta ******************************
    sigma_tuned = sigma_1
    length_of_k = k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 : (length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 : (length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 : (length_of_k + 1)]
    pairs_delta = torch.zeros(
        y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0]
    )

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = (
            y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1)
            - y_sorted_scores[i, :].float()
        )

    fraction_1 = -sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(
        y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0]
    )
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        # print(i / x_corresponding.shape[0])
        x_delta[:, :, i] = (
            x_corresponding[i, :].view(x_corresponding.shape[1], 1)
            - x_corresponding[i, :].float()
        )

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************

    ndcg_delta = torch.zeros(
        x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0]
    )
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]:
            break
        idcg = idcg_computation(x_sorted_scores[i, :], top_k)
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(
                        x_corresponding[i, :], j, k, idcg, top_k
                    )
                    # print(the_delta)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])
    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(
                without_zero[:, j, i]
            )  # 本来是 -

    mid = torch.zeros_like(x_similarity)
    the_x = (
        torch.arange(x_similarity.shape[0])
        .repeat(length_of_k, 1)
        .transpose(0, 1)
        .reshape(length_of_k * x_similarity.shape[0], 1)
        .squeeze()
    )
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding


def lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = (
        max_num * torch.ones_like(x_similarity[0, :])
    )
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = (
        max_num * torch.ones_like(y_similarity[0, :])
    )

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = (
        torch.arange(y_similarity.shape[0])
        .view(y_similarity.shape[0], 1)
        .repeat(1, y_similarity.shape[0])
    )
    y_ranks[the_row, y_sorted_idxs] = (
        1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    )
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 : (length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 : (length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding


def calculate_similarity_matrix(
    adj, features, metric=None, filterSigma=None, normalize=None, largestComponent=False
):
    if metric in ["cosine", "jaccard"]:
        # build similarity matrix
        if largestComponent:
            graph = nx.from_scipy_sparse_matrix(adj)
            lcc = max(
                nx.connected_components(graph), key=len
            )  # take largest connected components
            adj = nx.to_scipy_sparse_matrix(
                graph, nodelist=lcc, dtype="float", format="csc"
            )
        sim = get_similarity_matrix(adj, metric=metric)
        if filterSigma:
            sim = filter_similarity_matrix(sim, sigma=filterSigma)
        if normalize:
            sim = symmetric_normalize(sim)
    return sim


def cosine_similarity(mat):
    """
    get cosine similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    mat_row_norm = skpp.normalize(mat, axis=1)
    sim = mat_row_norm.dot(mat_row_norm.T)
    return sim


def filter_similarity_matrix(sim, sigma):
    """
    filter value by threshold = mean(sim) + sigma * std(sim)
    :param sim: similarity matrix
    :param sigma: hyperparameter for filtering values
    :return: filtered similarity matrix
    """
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def get_similarity_matrix(mat, metric=None):
    """
    get similarity matrix of nodes in specified metric
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :param metric: similarity metric
    :return: similarity matrix of nodes
    """
    if metric == "jaccard":
        return jaccard_similarity(mat.tocsc())
    elif metric == "cosine":
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError("Please specify the type of similarity metric.")


def calculate_group_lap(sim, sens):
    unique_sens = [int(x) for x in sens.unique(sorted=True).tolist()]
    num_unique_sens = sens.unique().shape[0]
    sens = [int(x) for x in sens.tolist()]
    m_list = [0] * num_unique_sens
    avgSimD_list = [[] for i in range(num_unique_sens)]
    sim_list = [sim.copy() for i in range(num_unique_sens)]

    for row, col in zip(*sim.nonzero()):
        sensRow = unique_sens[sens[row]]
        sensCol = unique_sens[sens[col]]
        if sensRow == sensCol:
            sim_list[sensRow][row, col] = 2 * sim_list[sensRow][row, col]
            sim_to_zero_list = [x for x in unique_sens if x != sensRow]
            for sim_to_zero in sim_to_zero_list:
                sim_list[sim_to_zero][row, col] = 0
            m_list[sensRow] += 1
        else:
            m_list[sensRow] += 0.5
            m_list[sensRow] += 0.5

    lap = laplacian(sim)
    lap = lap.tocsr()
    for i in range(lap.shape[0]):
        sen_label = sens[i]
        avgSimD_list[sen_label].append(lap[i, i])
    avgSimD_list = [np.mean(l) for l in avgSimD_list]

    lap_list = [laplacian(sim) for sim in sim_list]

    return lap_list, m_list, avgSimD_list


def convert_sparse_matrix_to_sparse_tensor(X):
    X = X.tocoo()

    X = torch.sparse_coo_tensor(
        torch.tensor([X.row.tolist(), X.col.tolist()]),
        torch.tensor(X.data.astype(np.float32)),
    )
    return X


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
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
        self.convx = spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode="max")
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
        self.conv1.aggr = "mean"
        self.transition = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = "mean"

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
        self.dgi_model = DeepGraphInfomax(
            enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption
        )

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, base_model="gcn", k: int = 2
    ):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == "gcn":
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == "gin":
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == "sage":
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == "infomax":
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == "jk":
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


class GNN_individual(torch.nn.Module):
    def __init__(
        self,
        dataset_name,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        num_hidden=16,
        num_proj_hidden=16,
        lr=0.001,
        weight_decay=1e-5,
        encoder="gcn",
        sim_coeff=0.5,
        nclass=1,
        device="cuda",
    ):
        super(GNN_individual, self).__init__()

        self.device = device

        # self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        self.edge_index = adj.coalesce().indices()

        row = adj._indices()[0].cpu().numpy()
        col = adj._indices()[1].cpu().numpy()
        data = adj._values().cpu().numpy()
        shape = adj.size()
        self.adj = sp.csr_matrix((data, (row, col)), shape=shape)

        self.encoder = Encoder(
            in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder
        ).to(device)
        # model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff,
        # nclass=num_class).to(device)

        self.sim_coeff = sim_coeff
        # self.encoder = encoder
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx
        self.drop_edge_rate_1 = self.drop_edge_rate_2 = 0
        self.drop_feature_rate_1 = self.drop_feature_rate_2 = 0

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass)

        for m in self.modules():
            self.weights_init(m)

        par_1 = (
            list(self.encoder.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
            + list(self.fc4.parameters())
        )
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        self.optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
        self.optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
        self = self.to(device)

        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)

        sim = calculate_similarity_matrix(self.adj, self.features, metric="cosine")
        lap = laplacian(sim)

        try:
            with open("laplacians_{}".format(dataset_name) + ".pickle", "rb") as f:
                loadLaplacians = pickle.load(f)
            lap_list, m_list, avgSimD_list = (
                loadLaplacians["lap_list"],
                loadLaplacians["m_list"],
                loadLaplacians["avgSimD_list"],
            )
            print("Laplacians loaded from previous runs")
        except:
            print("Calculating laplacians...(this may take a while)")
            lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
            saveLaplacians = {}
            saveLaplacians["lap_list"] = lap_list
            saveLaplacians["m_list"] = m_list
            saveLaplacians["avgSimD_list"] = avgSimD_list
            with open("laplacians_{}".format(dataset_name) + ".pickle", "wb") as f:
                pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Laplacians calculated and stored.")

        self.lap = convert_sparse_matrix_to_sparse_tensor(lap)
        self.lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
        self.lap_1 = self.lap_list[0].cuda()
        self.lap_2 = self.lap_list[1].cuda()
        self.m_u1 = m_list[0]
        self.m_u2 = m_list[1]

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

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
        return (
            -torch.max(F.softmax(x2), dim=1)[0]
            * torch.log(torch.max(F.softmax(x1), dim=1)[0])
        ).mean()

    def D(self, x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c1 = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx]) / 2
        l2 = self.D(h2[idx], p1[idx]) / 2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff * (l1 + l2), l3

    def forwarding_predict(self, emb):

        # classifier
        c1 = self.classifier(emb)

        return c1

    def fit(self, epochs=300):
        best_loss = 100
        for epoch in range(epochs + 1):

            sim_loss = 0

            self.train()
            self.optimizer_2.zero_grad()
            edge_index_1 = self.edge_index
            x_1 = self.features

            # classifier
            z1 = self.forward(x_1, edge_index_1)
            c1 = self.classifier(z1)

            # Binary Cross-Entropy
            cl_loss = F.binary_cross_entropy_with_logits(
                c1[self.idx_train],
                self.labels[self.idx_train].unsqueeze(1).float().to(self.device),
            )

            cl_loss.backward()
            self.optimizer_2.step()

            # Validation
            self.eval()
            z_val = self.forward(self.features, self.edge_index)
            c_val = self.classifier(z_val)
            val_loss = F.binary_cross_entropy_with_logits(
                c_val[self.idx_val],
                self.labels[self.idx_val].unsqueeze(1).float().to(self.device),
            )

            # if epoch % 100 == 0:
            #     print(f"[Train] Epoch {epoch}: train_c_loss: {cl_loss:.4f} | val_c_loss: {val_loss:.4f}")

            if (val_loss) < best_loss:
                self.val_loss = val_loss.item()

                best_loss = val_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/weights_GNN_{self.encoder}.pt")

    def predict(self):

        self.load_state_dict(torch.load(f"data/weights_GNN_{self.encoder}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb)

        output_preds = (
            (output.squeeze() > 0)
            .type_as(self.labels)[self.idx_test]
            .detach()
            .cpu()
            .numpy()
        )

        output = output

        IF = torch.trace(
            torch.mm(output.t(), torch.sparse.mm(self.lap.cuda(), output))
        ).item()
        f_u1 = (
            torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_1, output)))
            / self.m_u1
        )
        f_u1 = f_u1.item()
        f_u2 = (
            torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_2, output)))
            / self.m_u2
        )
        f_u2 = f_u2.item()
        if_group_pct_diff = np.abs(f_u1 - f_u2) / min(f_u1, f_u2)
        GDIF = max(f_u2 / (f_u1 + 1e-9), f_u1 / (f_u2 + 1e-9))

        x_inverse = 1 - output[self.idx_test].sigmoid()
        x_inverse = torch.log(x_inverse / (1 - x_inverse))
        y_similarity = simi(torch.concat([output[self.idx_test], x_inverse], -1))
        x_similarity = simi(self.features[self.idx_test])
        x_sorted_scores, y_sorted_idxs, x_corresponding = (
            lambdas_computation_only_review(
                x_similarity, y_similarity, top_k=10, k_para=1
            )
        )
        # print(self.features[self.idx_test])
        # print(x_similarity)
        # print(y_similarity)
        # print(x_sorted_scores)
        # print(y_sorted_idxs)

        ndcg_value = avg_ndcg(
            x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k=10
        )
        print("ndcg", ndcg_value)
        print("\n")

        # print report
        print(f"IF: {IF}")
        # print(f'Individual Unfairness for Group 1: {f_u1}')
        # print(f'Individual Unfairness for Group 2: {f_u2}')
        print(f"GDIF: {GDIF}")

        idx_test = self.idx_test.cpu().numpy()
        self.labels = self.labels.cpu().numpy()

        pred = output_preds
        F1 = f1_score(self.labels[idx_test], pred, average="micro")
        ACC = accuracy_score(
            self.labels[idx_test],
            pred,
        )

        if self.labels.max() > 1:
            AUCROC = 0
        else:
            try:
                AUCROC = roc_auc_score(self.labels[idx_test], pred)
            except:
                AUCROC = "nan"

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = (
            self.predict_sens_group(pred, idx_test)
        )

        SP, EO = self.fair_metric(
            np.array(pred), self.labels[idx_test], self.sens[idx_test].cpu().numpy()
        )

        pred = output[self.idx_val].detach().cpu().numpy()
        loss_fn = torch.nn.BCELoss()
        self.val_loss = loss_fn(
            torch.FloatTensor(pred).sigmoid().squeeze(),
            torch.tensor(self.labels[self.idx_val.detach().cpu().numpy()])
            .squeeze()
            .float(),
        ).item()

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
            IF,
            GDIF,
            ndcg_value,
        )

    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))

        equality = abs(
            sum(pred[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
        )

        return parity.item(), equality.item()

    def predict_sens_group(self, pred, idx_test):

        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][self.sens[idx_test] == sens],
                pred[self.sens[idx_test] == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][self.sens[idx_test] == sens],
                pred[self.sens[idx_test] == sens],
            )
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                try:
                    AUCROC = roc_auc_score(
                        self.labels[idx_test][self.sens[idx_test] == sens],
                        pred[self.sens[idx_test] == sens],
                    )
                except:
                    AUCROC = "nan"
            result.extend([ACC, AUCROC, F1])

        return result


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1 - x[:, sens_idx]

    return x
