import torch.nn as nn
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from time import perf_counter
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import math
from scipy.sparse.csgraph import laplacian
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
import networkx as nx
import sklearn.preprocessing as skpp


"""
Currently including AUCROC, Ranking based IF, IF, and GDIF.

More is being added...

"""


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = np.diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees



def calculate_group_lap(sim, sens):
    unique_sens = [int(x) for x in sens.unique(sorted=True).tolist()]
    num_unique_sens = sens.unique().shape[0]
    sens = [int(x) for x in sens.tolist()]
    m_list = [0]*num_unique_sens
    avgSimD_list = [[] for i in range(num_unique_sens)]
    sim_list = [sim.copy() for i in range(num_unique_sens)]

    for row, col in zip(*sim.nonzero()):
        sensRow = unique_sens[sens[row]]
        sensCol = unique_sens[sens[col]]
        if sensRow == sensCol:
            sim_list[sensRow][row,col] = 2*sim_list[sensRow][row,col]
            sim_to_zero_list = [x for x in unique_sens if x != sensRow]
            for sim_to_zero in sim_to_zero_list:
                sim_list[sim_to_zero][row,col] = 0
            m_list[sensRow] += 1
        else:
            m_list[sensRow] += 0.5
            m_list[sensRow] += 0.5

    lap = laplacian(sim)
    lap = lap.tocsr()
    for i in range(lap.shape[0]):
        sen_label = sens[i]
        avgSimD_list[sen_label].append(lap[i,i])
    avgSimD_list = [np.mean(l) for l in avgSimD_list]

    lap_list = [laplacian(sim) for sim in sim_list]

    return lap_list, m_list, avgSimD_list



def calculate_similarity_matrix(adj, features, metric=None, filterSigma = None, normalize = None, largestComponent=False):
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



def convert_sparse_matrix_to_sparse_tensor(X):
    X = X.tocoo()

    X = torch.sparse_coo_tensor(torch.tensor([X.row.tolist(), X.col.tolist()]),
                              torch.tensor(X.data.astype(np.float32)))
    return X



def simi(output):  # new_version

    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res

def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):

    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k]).cuda()

    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()


def lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores.cuda(), y_sorted_idxs.cuda(), x_corresponding.cuda()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def auc_roc(y_hat, y, idx_test):
    print("****************************************")
    print("***************  AUCROC  ***************")

    print(y_hat)

    # output_preds = (y_hat.squeeze()>0).type_as(y).detach().cpu().numpy()
    output_preds = torch.zeros(y_hat.shape[0], 2)
    output_preds[y_hat[:, 1] > y_hat[:, 0], 1] = 1
    output_preds[y_hat[:, 1] <= y_hat[:, 0], 0] = 1
    output_preds = output_preds.cpu().numpy()

    print(y)
    print(output_preds)
    print(y.sum(axis=0))
    print(output_preds.sum(axis=0))

    # y = encode_onehot(y.cpu().numpy())
    auc_roc_value = roc_auc_score(y.cpu().numpy()[idx_test.cpu().numpy()], output_preds[idx_test.cpu().numpy()])
    print(auc_roc_value)
    y = 1 - y
    auc_roc_value = roc_auc_score(y.cpu().numpy()[idx_test.cpu().numpy()], output_preds[idx_test.cpu().numpy()])
    print(auc_roc_value)
    print("\n")

    return auc_roc_value



def ranking_based_IF(x, y_hat, idx_test, top_k):
    print("****************************************")
    print("**********  Ranking based IF  **********")

    y_similarity = simi(y_hat[idx_test])
    x_similarity = simi(x[idx_test])
    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para=1)
    ndcg_value = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)
    print(ndcg_value)
    print("\n")

    return ndcg_value


def IF(adj, x, y_hat):
    print("****************************************")
    print("*****************  IF  *****************")

    row=adj._indices()[0].cpu().numpy()
    col=adj._indices ()[1].cpu().numpy()
    data=adj._values().cpu().numpy()
    shape=adj.size()


    adj=sp.csr_matrix((data,(row, col)), shape=shape)

    sim = calculate_similarity_matrix(adj, x, metric='cosine')
    lap = sparse_mx_to_torch_sparse_tensor(laplacian(sim)).cuda()

    individual_unfairness = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap, y_hat))).item()
        
    print(individual_unfairness)
    print("\n")

    return individual_unfairness

def GDIF(y_hat, sens):
    print("****************************************")
    print("***************  GDIF  *****************")

    lap_list, m_list, _ = calculate_group_lap(sim, sens)
    lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
    lap_1 = lap_list[0].cuda()
    lap_2 = lap_list[1].cuda()
    m_u1 = m_list[0]
    m_u2 = m_list[1]

    f_u1 = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap_1, y_hat)))/m_u1
    f_u1 = f_u1.item()
    f_u2 = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap_2, y_hat)))/m_u2
    f_u2 = f_u2.item()

    GDIF_value = max(f_u1/f_u2, f_u2/f_u1)
    print(GDIF_value)
    print("\n")

    return GDIF_value



def individual_fairness_evaluation_cobo(adj, x, y_hat, y, sens, idx_test, top_k):

    print("****************************************")
    print("***************  AUCROC  ***************")

    print(y_hat)

    # output_preds = (y_hat.squeeze()>0).type_as(y).detach().cpu().numpy()
    output_preds = torch.zeros(y_hat.shape[0], 2)
    output_preds[y_hat[:, 1] > y_hat[:, 0], 1] = 1
    output_preds[y_hat[:, 1] <= y_hat[:, 0], 0] = 1
    output_preds = output_preds.cpu().numpy()

    print(y)
    print(output_preds)
    print(y.sum(axis=0))
    print(output_preds.sum(axis=0))

    # y = encode_onehot(y.cpu().numpy())
    auc_roc_value = roc_auc_score(y.cpu().numpy()[idx_test.cpu().numpy()], output_preds[idx_test.cpu().numpy()])
    print(auc_roc_value)
    y = 1 - y
    auc_roc_value = roc_auc_score(y.cpu().numpy()[idx_test.cpu().numpy()], output_preds[idx_test.cpu().numpy()])
    print(auc_roc_value)
    print("\n")



    # y = encode_onehot(y.cpu().numpy())
    # auc_roc_value = roc_auc_score(y[idx_test.cpu()], y_hat.detach().cpu().numpy()[idx_test.cpu()])
    # print(auc_roc_value)
    # y = 1 - y
    # auc_roc_value = roc_auc_score(y[idx_test.cpu()], y_hat.detach().cpu().numpy()[idx_test.cpu()])
    # print(auc_roc_value)
    # print("\n")


    print("****************************************")
    print("**********  Ranking based IF  **********")

    y_similarity = simi(y_hat[idx_test])
    x_similarity = simi(x[idx_test])
    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para=1)
    ndcg_value = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)
    print(ndcg_value)
    print("\n")

    print("****************************************")
    print("*****************  IF  *****************")

    row=adj._indices()[0].cpu().numpy()
    col=adj._indices ()[1].cpu().numpy()
    data=adj._values().cpu().numpy()
    shape=adj.size()


    adj=sp.csr_matrix((data,(row, col)), shape=shape)

    sim = calculate_similarity_matrix(adj, x, metric='cosine')
    lap = sparse_mx_to_torch_sparse_tensor(laplacian(sim)).cuda()

    individual_unfairness = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap, y_hat))).item()
        
    print(individual_unfairness)
    print("\n")




    print("****************************************")
    print("***************  GDIF  *****************")

    lap_list, m_list, _ = calculate_group_lap(sim, sens)
    lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
    lap_1 = lap_list[0].cuda()
    lap_2 = lap_list[1].cuda()
    m_u1 = m_list[0]
    m_u2 = m_list[1]

    f_u1 = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap_1, y_hat)))/m_u1
    f_u1 = f_u1.item()
    f_u2 = torch.trace(torch.mm(y_hat.t(), torch.sparse.mm(lap_2, y_hat)))/m_u2
    f_u2 = f_u2.item()

    GDIF = max(f_u1/f_u2, f_u2/f_u1)
    print(GDIF)
    print("\n")






























