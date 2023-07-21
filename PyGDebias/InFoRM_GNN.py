
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import argparse
import pickle
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from scipy.sparse.csgraph import laplacian
import torch.nn as nn
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import pickle as pkl


# import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
import sklearn.preprocessing as skpp
import scipy.sparse as sp


import networkx as nx

def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = ( c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)

    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i])
    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    print("Now Average ERR@k = ", final.item())

    return final.item()



def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class JK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(JK, self).__init__()
        self.body = JK_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class JK_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(JK_Body, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.convx = GCNConv(nhid, nhid)
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

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



class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GIN, self).__init__()

        self.body = GIN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GIN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GIN_Body, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid),
        )
        self.gc1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x







def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def calculate_similarity_matrix(adj, features, metric=None, filterSigma=None, normalize=None, largestComponent=False):
    if metric in ['cosine', 'jaccard']:
        # build similarity matrix
        if largestComponent:
            graph = nx.from_scipy_sparse_matrix(adj)
            lcc = max(nx.connected_components(graph), key=len)  # take largest connected components
            adj = nx.to_scipy_sparse_matrix(graph, nodelist=lcc, dtype='float', format='csc')
        sim = get_similarity_matrix(adj, metric=metric)
        if filterSigma:
            sim = filter_similarity_matrix(sim, sigma=filterSigma)
        if normalize:
            sim = symmetric_normalize(sim)
    return sim


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

    X = torch.sparse_coo_tensor(torch.tensor([X.row.tolist(), X.col.tolist()]),
                                torch.tensor(X.data.astype(np.float32)))
    return X


def trace(mat):
    """
    calculate trace of a sparse matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: Tr(mat)
    """
    return mat.diagonal().sum()


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = np.diag(np.asarray(degrees)[0, :])     # ????????????????????   degrees = diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees


def jaccard_similarity(mat):
    """
    get jaccard similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= (aa + bb - ab.data)
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
    if metric == 'jaccard':
        return jaccard_similarity(mat.tocsc())
    elif metric == 'cosine':
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError('Please specify the type of similarity metric.')


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class InFoRM_GNN(nn.Module):
    def __init__(self, adj, features, idx_train, idx_val, idx_test, labels, sens, gnn_name='gcn', lr=0.001, hidden=16, dropout=0, weight_decay=1e-5, device="cuda"):
        super(InFoRM_GNN, self).__init__()

        row=adj._indices()[0].cpu().numpy()
        col=adj._indices ()[1].cpu().numpy()
        data=adj._values().cpu().numpy()
        shape=adj.size()
        adj=sp.csr_matrix((data,(row, col)), shape=shape)

        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        sim = calculate_similarity_matrix(adj, features, metric='cosine')
        lap = laplacian(sim)


        print("Calculating laplacians...(this may take a while)")
        lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
        saveLaplacians = {}
        saveLaplacians['lap_list'] = lap_list
        saveLaplacians['m_list'] = m_list
        saveLaplacians['avgSimD_list'] = avgSimD_list
        with open("laplacians-1" + '.pickle', 'wb') as f:
            pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Laplacians calculated and stored.")


        with open("laplacians-1" + '.pickle', 'rb') as f:
            loadLaplacians = pickle.load(f)
        lap_list, m_list, avgSimD_list = loadLaplacians['lap_list'], loadLaplacians['m_list'], loadLaplacians['avgSimD_list']
        print("Laplacians loaded from previous runs")


        self.lap = convert_sparse_matrix_to_sparse_tensor(lap)
        self.lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
        self.lap_1 = None
        self.lap_2 = None

        if device == 'cuda':
            self.lap_1 = self.lap_list[0].cuda()
            self.lap_2 = self.lap_list[1].cuda()
        else:
            self.lap_1 = self.lap_list[0]
            self.lap_2 = self.lap_list[1]
        self.m_u1 = m_list[0]
        self.m_u2 = m_list[1]

        self.features = features.to(device)
        self.edge_index = edge_index.to(device)
        self.labels = labels.to(device)
        self.device = device

        self.idx_train = idx_train.to(device)
        self.idx_val = idx_val.to(device)
        self.idx_test = idx_test.to(device)

        # Model and optimizer
        self.model = None
        self.optimizer = None
        num_class = 1
        if gnn_name == 'gcn':
            self.model = GCN(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=num_class,
                        dropout=dropout)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.model = self.model.to(device)

        elif gnn_name == 'gin':
            self.model = GIN(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=num_class,
                        dropout=dropout)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.model = self.model.to(device)

        elif gnn_name == 'jk':
            self.model = JK(nfeat=features.shape[1],
                       nhid=hidden,
                       nclass=num_class,
                       dropout=dropout)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.model = self.model.to(device)



    def fit(self, epochs=3000, alpha=5e-6, opt_if=1):

        # Train model
        t_total = time.time()
        best_loss = np.inf
        best_acc = 0
        features = self.features.to(self.device)
        edge_index = self.edge_index.to(self.device)
        labels = self.labels.to(self.device)
        lap = self.lap.to(self.device)



        for epoch in range(epochs + 1):
            t = time.time()

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features, edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze() > 0).type_as(labels)
            # output[output < 0.0] = 0.0
            # output[output > 1.0] = 1.0 
            # print(output)
            # print(labels)
            # assert 1 == 0
            loss_train = F.binary_cross_entropy_with_logits(output[self.idx_train],
                                                            labels[self.idx_train].unsqueeze(1).float().to(self.device))

            if opt_if:
                # IF loss  torch.sparse.mm
                if_loss = alpha * torch.trace(torch.mm(output.t(), torch.mm(lap, output)))
                loss_train = loss_train + if_loss


            auc_roc_train = roc_auc_score(labels.cpu().numpy()[self.idx_train.cpu().numpy()], output.detach().cpu().numpy()[self.idx_train.cpu().numpy()])
            loss_train.backward()
            self.optimizer.step()

            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(features, edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze() > 0).type_as(labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.idx_val],
                                                          labels[self.idx_val].unsqueeze(1).float().to(self.device))

            if opt_if:
                # IF loss
                if_loss = alpha * torch.trace(torch.mm(output.t(), torch.sparse.mm(lap, output)))
                loss_val = loss_val + if_loss

            auc_roc_val = roc_auc_score(labels.cpu().numpy()[self.idx_val.cpu().numpy()], output.detach().cpu().numpy()[self.idx_val.cpu().numpy()])

            if epoch % 500 == 0:
                print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                weightName = './v0_weights_vanilla.pt'
                torch.save(self.model.state_dict(), weightName)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


    def calculate_ranking_fairness(self, epoch, model_name, adj, output):
        y_similarity1 = simi(output[self.idx_train])
        x_similarity = simi(self.features[self.idx_train])
        lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, self.top_k, self.k_para, self.sigma_1)
        assert lambdas1.shape == y_similarity1.shape

        y_similarity = simi(output[self.idx_test])
        x_similarity = simi(self.features[self.idx_test])

        print("Ranking optimizing... ")
        x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, self.top_k, self.k_para)
        self.all_ndcg_list_test.append(avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, self.top_k))

        y_similarity1.backward(self.lambdas_para * lambdas1)
        self.optimizer.step()


    def predict(self):

        self.model.eval()
        output = self.model(self.features, self.edge_index).squeeze()

        # Report
        output_preds = (output.squeeze() > 0).type_as(self.labels)
        # counter_output_preds = (counter_output.squeeze() > 0).type_as(self.labels)
        # noisy_output_preds = (noisy_output.squeeze() > 0).type_as(self.labels)
        auc_roc_test = roc_auc_score(self.labels.cpu().numpy()[self.idx_test.cpu().numpy()],
                                     output.detach().cpu().numpy()[self.idx_test.cpu().numpy()])

        print(output)
        print(output_preds)

        F1 = f1_score(self.labels.cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()], average='micro')
        ACC=accuracy_score(self.labels.detach().cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()],)
        AUCROC=roc_auc_score(self.labels.cpu().numpy()[self.idx_test.cpu().numpy()], output_preds.detach().cpu().numpy()[self.idx_test.cpu().numpy()])


        # counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[self.idx_test].sum().item() / self.idx_test.shape[0])
        # robustness_score = 1 - (output_preds.eq(noisy_output_preds)[self.idx_test].sum().item() / self.idx_test.shape[0])


        output=output.unsqueeze(1)
        individual_unfairness = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap.to(self.device), output))).item()
        f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_1, output))) / self.m_u1
        f_u1 = f_u1.item()
        f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(self.lap_2, output))) / self.m_u2
        f_u2 = f_u2.item()
        if_group_pct_diff = np.abs(f_u1 - f_u2) / min(f_u1, f_u2)
        GDIF = max(f_u2 / f_u1, f_u1 / f_u2)

        # print report
        print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        print(f'Total Individual Unfairness: {individual_unfairness}')
        print(f'Individual Unfairness for Group 1: {f_u1}')
        print(f'Individual Unfairness for Group 2: {f_u2}')
        print(f'GDIF: {GDIF}')

        return F1, ACC, AUCROC, individual_unfairness, GDIF














