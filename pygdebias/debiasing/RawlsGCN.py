import argparse
import numpy as np
import torch


import math
import os
import logging
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Parameter
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import statistics
import torch.nn.functional as F

from collections import defaultdict

import os

import torch

from collections import defaultdict


class RawlsGCNGrad(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RawlsGCNGrad, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # to fix gradient in trainer
        self.layers_info = {
            "gc1": 0,
            "gc2": 1,
        }

    def forward(self, x, adj):
        pre_act_embs, embs = [], [x]  # adding input node features to make index padding consistent
        x = self.gc1(x, adj)
        x.retain_grad()
        pre_act_embs.append(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        embs.append(x)

        x = self.gc2(x, adj)
        x.retain_grad()
        pre_act_embs.append(x)
        x = F.log_softmax(x, dim=1)
        embs.append(x)
        return pre_act_embs, embs


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class RawlsGCNGraph(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RawlsGCNGraph, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class SinkhornKnopp:
    """
    Sinkhorn-Knopp algorithm to compute doubly stochastic matrix for a non-negative square matrix with total support.
    For reference, see original paper: http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
    """

    def __init__(self, max_iter=1000, epsilon=1e-3):
        """
        self:
            max_iter (int): The maximum number of iterations, default is 1000.
            epsilon (float): Error tolerance for row/column sum, should be in the range of [0, 1], default is 1e-3.
        """

        assert isinstance(max_iter, int) or isinstance(max_iter, float), (
            "max_iter is not int or float: %r" % max_iter
        )
        assert max_iter > 0, "max_iter must be greater than 0: %r" % max_iter
        self.max_iter = int(max_iter)

        assert isinstance(epsilon, int) or isinstance(epsilon, float), (
            "epsilon is not of type float or int: %r" % epsilon
        )
        assert 0 <= epsilon < 1, (
            "epsilon must be between 0 and 1 exclusive: %r" % epsilon
        )
        self.epsilon = epsilon

    def fit(self, mat):
        """

        self:
            mat (scipy.sparse.matrix): The input non-negative square matrix. The matrix must have total support, i.e.,
                row/column sum must be non-zero.
        Returns:
            ds_mat (scipy.sparse.matrix): The doubly stochastic matrix of the input matrix.
        """
        assert sum(mat.data < 0) == 0  # must be non-negative
        assert mat.ndim == 2  # must be a matrix
        assert mat.shape[0] == mat.shape[1]  # must be square

        max_threshold, min_threshold = 1 + self.epsilon, 1 - self.epsilon

        right = np.ravel(mat.sum(axis=0).flatten())
        right = np.divide(1, right, where=right != 0)

        left = mat @ right
        left = np.divide(1, left, out=np.zeros_like(left), where=left != 0)

        for iter in range(self.max_iter):
            row_sum = np.ravel(mat.sum(axis=1)).flatten()
            col_sum = np.ravel(mat.sum(axis=0)).flatten()
            if (
                sum(row_sum < min_threshold) == 0
                and sum(row_sum > max_threshold) == 0
                and sum(col_sum < min_threshold) == 0
                and sum(col_sum > max_threshold) == 0
            ):
                logger.info(
                    "Sinkhorn-Knopp - Converged in {iter} iterations.".format(iter=iter)
                )
                return mat

            right = left @ mat
            right = np.divide(1, right, out=np.zeros_like(right), where=right != 0)

            left = mat @ right
            left = np.divide(1, left, out=np.zeros_like(left), where=left != 0)

            right_diag = sp.diags(right)
            left_diag = sp.diags(left)
            mat = left_diag @ mat @ right_diag
        logger.info("Sinkhorn-Knopp - Maximum number of iterations reached.")
        return mat
def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1
class GraphDataset:
    def __init__(self, configs,adj, feats, labels):
        #if not os.path.isfile("../data/{name}.pt".format(name=configs["name"])):
            #raise FileNotFoundError("Dataset does not exist!")
        # load data
        #data = torch.load("../data/{name}.pt".format(name=configs["name"]))

        #adj, feats, labels, idx_train, idx_val, idx_test, sens = process_pokec_nba('pokec_n',
        #                                                                           predict_attr_specify='gender')
        adj = sp.coo_matrix(adj.to_dense().numpy())

        # read fields
        # self.num_nodes = data["num_nodes"]
        self.num_nodes = feats.shape[0]

        # self.num_edges = data["num_edges"]
        self.num_edges = adj.row.shape[0]

        # self.num_node_features = data["num_node_features"]
        self.num_node_features = feats.shape[1]

        # self.num_classes = data["num_classes"]
        self.num_classes = (labels.max() + 1).item()
        # print(self.num_classes.dtype)

        # self.raw_graph = data["adjacency_matrix"]
        self.raw_graph = adj

        # self.features = torch.FloatTensor(
        #    np.array(
        #        row_normalize(data["node_features"])
        #    )
        # )

        feats=feature_norm(feats)



        self.features = torch.FloatTensor(np.array(feats))



        # self.labels = data["labels"]
        self.labels = labels

        self.is_ratio = configs["is_ratio"]
        self.split_by_class = configs["split_by_class"]
        self.num_train = configs["num_train"]
        self.num_val = configs["num_val"]
        self.num_test = configs["num_test"]
        self.ratio_train = configs["ratio_train"]
        self.ratio_val = configs["ratio_val"]

        # free memory


    def random_split(self):
        # initialization
        mask = torch.empty(self.num_nodes, dtype=torch.bool).fill_(False)
        if self.is_ratio:
            self.num_train = int(self.ratio_train * self.num_nodes)
            self.num_val = int(self.ratio_val * self.num_nodes)
            self.num_test = self.num_nodes - self.num_train - self.num_val

        # get indices for training
        if not self.is_ratio and self.split_by_class:
            self.train_idx = self.get_split_by_class(num_train_per_class=self.num_train)
        else:
            self.train_idx = torch.randperm(self.num_nodes)[:self.num_train]

        # get remaining indices
        mask[self.train_idx] = True
        remaining = (~mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        # get indices for validation and test
        self.val_idx = remaining[:self.num_val]
        self.test_idx = remaining[self.num_val:self.num_val + self.num_test]

        # free memory
        del mask, remaining

    def set_random_split(self, splits):
        self.train_idx = splits["train_idx"]
        self.val_idx = splits["val_idx"]
        self.test_idx = splits["test_idx"]

    def get_split_by_class(self, num_train_per_class):
        res = None
        for c in range(self.num_classes):
            idx = (self.labels == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            res = torch.cat((res, idx)) if res is not None else idx
        return res

    @staticmethod
    def get_doubly_stochastic(mat):
        sk = SinkhornKnopp(max_iter=1000, epsilon=1e-2)
        mat = matrix2tensor(
            sk.fit(mat)
        )
        return mat

    @staticmethod
    def get_row_normalized(mat):
        mat = matrix2tensor(
            row_normalize(mat)
        )
        return mat

    @staticmethod
    def get_column_normalized(mat):
        mat = matrix2tensor(
            row_normalize(mat)
        )
        mat = torch.transpose(mat, 0, 1)
        return mat

    @staticmethod
    def get_symmetric_normalized(mat):
        mat = matrix2tensor(
            symmetric_normalize(mat)
        )
        return mat

    def preprocess(self, type="laplacian"):
        if type == "laplacian":
            self.graph = matrix2tensor(
                symmetric_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        elif type == "row":
            self.graph = matrix2tensor(
                row_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        elif type == "doubly_stochastic_no_laplacian":
            self.graph = self.get_doubly_stochastic(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
        elif type == "doubly_stochastic_laplacian":
            self.graph = symmetric_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            self.graph = self.get_doubly_stochastic(self.graph)
        else:
            raise ValueError(
                "type should be laplacian, row, doubly_stochastic_no_laplacian or doubly_stochastic_laplacian"
            )

    def get_degree_splits(self):
        deg = self.raw_graph.sum(axis=0)
        self.degree_splits = defaultdict(list)
        for idx in range(self.num_nodes):
            degree = deg[0, idx]
            self.degree_splits[degree].append(idx)

    def encode_degree_splits_to_labels(self):
        label = 0
        encoded_labels = set()
        self.degree_labels = [0] * self.num_nodes
        for degree, nodes in self.degree_splits.items():
            if degree in encoded_labels:
                continue
            for node_id in nodes:
                self.degree_labels[node_id] = label
            label += 1
        self.degree_labels = torch.LongTensor(self.degree_labels)


class Evaluator:
    def __init__(self, loss, logger):
        self.loss = loss
        self.logger = logger

    @staticmethod
    def _get_accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def _get_loss_variance(self, output, labels):
        res = list()
        for i in range(labels.shape[0]):
            label = labels[i]
            if self.loss == "negative_log_likelihood":
                nll = -output[i, label].item()
            else:
                nll = -F.log_softmax(output, dim=1)[i, label].item()
            res.append(nll)
        if len(res) > 1:
            return statistics.variance(res)
        else:
            return 0

    def _get_bias(self, output, labels, idx, raw_graph):
        deg = raw_graph.sum(axis=0)
        loss_by_deg = defaultdict(list)
        deg_test = deg[0, idx.cpu().numpy()]
        if self.loss == "negative_log_likelihood":
            loss_mat = -output
        else:
            loss_mat = -F.log_softmax(output, dim=1)
        for i in range(idx.shape[0]):
            degree = int(deg_test[0, i])
            label = labels[i]
            loss_val = loss_mat[i, label].item()
            loss_by_deg[degree].append(loss_val)
        res = [statistics.mean(losses) for degree, losses in loss_by_deg.items()]
        return statistics.variance(res)

    def eval(self, output, labels, idx, raw_graph, stage):
        if self.loss == "negative_log_likelihood":
            loss_value = F.nll_loss(output, labels)
        else:
            loss_value = F.cross_entropy(output, labels)
        accuracy = self._get_accuracy(output, labels)
        bias = self._get_bias(output, labels, idx, raw_graph)
        info = "{stage} - loss: {loss}\taccuracy: {accuracy}\tbias:{bias}".format(
            stage=stage,
            loss=loss_value,
            accuracy=accuracy,
            bias=bias,
        )
        if stage in ("validation", "test"):
            info += "\n"
        self.logger.info(info)
        return accuracy.cpu().item(), bias

def encode_onehot(labels):
    """Encode label to a one-hot vector."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def row_normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv @ mx
    return mx


def symmetric_normalize(mat):
    """Symmetric-normalize sparse matrix."""
    D = np.asarray(mat.sum(axis=0).flatten())
    D = np.divide(1, D, out=np.zeros_like(D), where=D != 0)
    D = sp.diags(np.asarray(D)[0, :])
    D.data = np.sqrt(D.data)
    return D @ mat @ D


def accuracy(output, labels):
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def matrix2tensor(mat):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def tensor2matrix(t):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    indices = t.indices()
    row, col = indices[0, :].cpu().numpy(), indices[1, :].cpu().numpy()
    values = t.values().cpu().numpy()
    mat = sp.coo_matrix((values, (row, col)), shape=(t.shape[0], t.shape[1]))
    return mat


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def random_split(dataset):
    # initialization
    mask = torch.empty(dataset.num_nodes, dtype=torch.bool).fill_(False)
    if dataset.is_ratio:
        num_train = int(dataset.ratio_train * dataset.num_nodes)
        num_val = int(dataset.ratio_val * dataset.num_nodes)
        num_test = dataset.num_nodes - num_train - num_val
    else:
        num_train = dataset.num_train
        num_val = dataset.num_val
        num_test = dataset.num_test

    # get indices for training
    if not dataset.is_ratio and dataset.split_by_class:
        train_idx = dataset.get_split_by_class(num_train_per_class=num_train)
    else:
        train_idx = torch.randperm(dataset.num_nodes)[:num_train]

    # get remaining indices
    mask[train_idx] = True
    remaining = (~mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    # get indices for validation and test
    val_idx = remaining[:num_val]
    test_idx = remaining[num_val:num_val + num_test]

    print(train_idx.shape)
    print(val_idx.shape)
    print(test_idx.shape)
    print(1/0)

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

class InProcessingTrainer:
    def __init__(self, configs, data, model, on_gpu, device):
        self.configs = self.default_configs()
        self.configs.update(configs)

        self.data = data
        self.model = model

        if "ablation" in configs:
            if configs["ablation"] == "row":
                self.doubly_stochastic_graph = self.data.get_row_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "column":
                self.doubly_stochastic_graph = self.data.get_column_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "symmetric":
                self.doubly_stochastic_graph = self.data.get_symmetric_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "doubly_stochastic":
                self.doubly_stochastic_graph = self.data.get_doubly_stochastic(
                    tensor2matrix(self.data.graph.coalesce())
                )
        else:
            self.doubly_stochastic_graph = self.data.get_doubly_stochastic(
                tensor2matrix(self.data.graph.coalesce())
            )

        self.on_gpu = on_gpu
        self.device = device
        if on_gpu:
            self.data.graph = self.data.graph.to(device)
            self.data.features = self.data.features.to(device)
            self.data.labels = self.data.labels.to(device)
            self.data.train_idx = self.data.train_idx.to(device)
            self.data.val_idx = self.data.val_idx.to(device)
            self.data.test_idx = self.data.test_idx.to(device)
            self.doubly_stochastic_graph = self.doubly_stochastic_graph.to(device)
            self.model.to(device)

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
        )

        if self.configs["loss"] == "negative_log_likelihood":
            self.criterion = nn.NLLLoss()
        elif self.configs["loss"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "loss in configs should be either `negative_log_likelihood` or `cross_entropy`"
            )

        self.evaluator = Evaluator(loss=self.configs["loss"], logger=logger)

    def train(self):
        for epoch in range(self.configs["num_epoch"]):
            logger.info("Epoch {epoch}".format(epoch=epoch))

            self.model.train()
            self.opt.zero_grad()

            # training
            pre_act_embs, embs = self.model(self.data.features, self.data.graph)
            loss_train = self.criterion(
                embs[-1][self.data.train_idx], self.data.labels[self.data.train_idx]
            )
            loss_train.backward()
            self._fix_gradient(pre_act_embs, embs)
            self.opt.step()

            # validation
            self.model.eval()
            self.evaluator.eval(
                output=embs[-1][self.data.train_idx],
                labels=self.data.labels[self.data.train_idx],
                idx=self.data.train_idx,
                raw_graph=self.data.raw_graph,
                stage="train",
            )
            self.evaluator.eval(
                output=embs[-1][self.data.val_idx],
                labels=self.data.labels[self.data.val_idx],
                idx=self.data.val_idx,
                raw_graph=self.data.raw_graph,
                stage="validation",
            )

            # self._save_model()

    def test(self):
        self.model.eval()
        _, embs = self.model(self.data.features, self.data.graph)
        return self.evaluator.eval(
            output=embs[-1][self.data.test_idx],
            labels=self.data.labels[self.data.test_idx],
            idx=self.data.test_idx,
            raw_graph=self.data.raw_graph,
            stage="test",
        )

    def _fix_gradient(self, pre_act_embs, embs):
        flag = 0  # flag = 0 for weight, flag = 1 for bias
        weights, biases = list(), list()
        # group params
        for name, param in self.model.named_parameters():
            layer, param_type = name.split(".")
            if param_type == "weight":
                if flag == 1:
                    flag = 0
                weights.append(param.data)
            else:
                if flag == 0:
                    flag = 1
                biases.append(param.data)
            flag = 1 - flag

        # fix gradient
        for name, param in self.model.named_parameters():
            layer, param_type = name.split(".")
            idx = self.model.layers_info[layer]
            # idx for embs and pre_act_embs are aligned here because we add a padding in embs (i.e., input features)
            if param_type == "weight":
                normalized_grad = torch.mm(
                    embs[idx].transpose(1, 0),
                    torch.sparse.mm(
                        self.doubly_stochastic_graph, pre_act_embs[idx].grad
                    ),
                )
            else:
                normalized_grad = torch.squeeze(
                    torch.mm(
                        torch.ones(1, self.data.num_nodes).to(self.device),
                        pre_act_embs[idx].grad,
                    )
                )
            param.grad = normalized_grad

    def _save_model(self):
        folder = "/".join(self.configs["save_path"].split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), self.configs["save_path"])

    @staticmethod
    def default_configs():
        configs = {
            "name": "cora",
            "model": "gcn",
            "num_epoch": 200,
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "loss": "negative_log_likelihood",
        }
        configs["save_path"] = "ckpts/{name}/{model}/{setting}.pt".format(
            name=configs["name"],
            model=configs["model"],
            setting="lr={lr}_nepochs={nepochs}_decay={decay}".format(
                lr=configs["lr"],
                nepochs=configs["num_epoch"],
                decay=configs["weight_decay"],
            ),
        )
        return configs


class PreProcessingTrainer:
    def __init__(self, configs, data, model, on_gpu, device):
        self.configs = self.default_configs()
        self.configs.update(configs)
        self.data = data
        self.model = model
        self.device = device

        if on_gpu:
            self.data.graph = self.data.graph.to(device)
            self.data.features = self.data.features.to(device)
            self.data.labels = self.data.labels.to(device)
            self.data.train_idx = self.data.train_idx.to(device)
            self.data.val_idx = self.data.val_idx.to(device)
            self.data.test_idx = self.data.test_idx.to(device)
            self.model.to(device)

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
        )

        if self.configs["loss"] == "negative_log_likelihood":
            self.criterion = nn.NLLLoss()
        elif self.configs["loss"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "loss in configs should be either `negative_log_likelihood` or `cross_entropy`"
            )

        self.evaluator = Evaluator(loss=self.configs["loss"], logger=logger)

    def train(self):
        for epoch in range(self.configs["num_epoch"]):
            logger.info("Epoch {epoch}".format(epoch=epoch))

            self.model.train()
            self.opt.zero_grad()

            # training
            output = self.model(self.data.features, self.data.graph)
            loss_train = self.criterion(
                output[self.data.train_idx], self.data.labels[self.data.train_idx]
            )
            loss_train.backward()
            self.opt.step()

            # validation
            self.model.eval()
            self.evaluator.eval(
                output=output[self.data.train_idx],
                labels=self.data.labels[self.data.train_idx],
                idx=self.data.train_idx,
                raw_graph=self.data.raw_graph,
                stage="train",
            )
            self.evaluator.eval(
                output=output[self.data.val_idx],
                labels=self.data.labels[self.data.val_idx],
                idx=self.data.val_idx,
                raw_graph=self.data.raw_graph,
                stage="validation",
            )

            # self._save_model()

    def test(self):
        self.model.eval()
        output = self.model(self.data.features, self.data.graph)
        return self.evaluator.eval(
            output=output[self.data.test_idx],
            labels=self.data.labels[self.data.test_idx],
            idx=self.data.test_idx,
            raw_graph=self.data.raw_graph,
            stage="test",
        )

    def _save_model(self):
        folder = "/".join(self.configs["save_path"].split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), self.configs["save_path"])

    @staticmethod
    def default_configs():
        configs = {
            "name": "cora",
            "model": "gcn",
            "num_epoch": 200,
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "loss": "negative_log_likelihood",
        }
        configs["save_path"] = "ckpts/{name}/{model}/{setting}.pt".format(
            name=configs["name"],
            model=configs["model"],
            setting="lr={lr}_nepochs={nepochs}_decay={decay}".format(
                lr=configs["lr"],
                nepochs=configs["num_epoch"],
                decay=configs["weight_decay"],
            ),
        )
        return configs







class RawlsGCN():
    def generate_split(self,dataset):
        # set random seed
        if self.split_seed is not None:
            np.random.seed(self.split_seed)
            torch.manual_seed(self.split_seed)
            if self.cuda:
                torch.cuda.manual_seed(self.split_seed)

        # generate splits
        return random_split(dataset)

    def run_exp(self,dataset, split, configs):
        # set splits
        dataset.set_random_split(split)

        # set random seed
        np.random.seed(configs["seed"])
        torch.manual_seed(configs["seed"])
        if self.cuda:
            torch.cuda.manual_seed(configs["seed"])

        # init model
        if self.model == "rawlsgcn_graph":
            model = RawlsGCNGraph(
                nfeat=dataset.num_node_features,
                nhid=self.hidden,
                nclass=dataset.num_classes,
                dropout=self.dropout,
            )
        elif self.model == "rawlsgcn_grad":
            model = RawlsGCNGrad(
                nfeat=dataset.num_node_features,
                nhid=self.hidden,
                nclass=dataset.num_classes,
                dropout=self.dropout,
            )
        else:
            raise ValueError("Invalid model name!")

        # train and test
        if self.model == "rawlsgcn_graph":
            trainer = PreProcessingTrainer(
                configs=configs, data=dataset, model=model, on_gpu=self.cuda, device=self.device
            )
        elif self.model == "rawlsgcn_grad":
            trainer = InProcessingTrainer(
                configs=configs, data=dataset, model=model, on_gpu=self.cuda, device=self.device
            )
        else:
            raise ValueError("Invalid model name!")

        return trainer

    def fit(self,adj, feats, labels, idx_train, idx_val,idx_test, enable_cude=True, device_number=0, model='rawlsgcn_graph',
            seed=0, num_epoch=100, lr=0.05, weight_decay=5e-4, hidden=64, dropout=0.5, loss='negative_log_likelihood' ):

        self.enable_cuda=enable_cude
        self.device_number=device_number
        self.model=model
        self.split_seed=seed
        self.num_epoch=num_epoch
        self.lr=lr
        self.weight_decay=weight_decay
        self.hidden=hidden
        self.dropout=dropout
        self.loss=loss
        

        self.cuda = self.enable_cuda and torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device(f"cuda:{self.device_number}")
        else:
            self.device = torch.device("cpu")

        # update configs
        dataset_configs = {
            "is_ratio": False,
            "split_by_class": True,
            "num_train": 20,
            "num_val": 500,
            "num_test": 1000,
            "ratio_train": 0.8,
            "ratio_val": 0.1,
        }

        # load data
        dataset = GraphDataset(dataset_configs, adj, feats, labels)

        # get random splits
        #
        split={
            "train_idx": idx_train,
            "val_idx": idx_val,
            "test_idx": idx_test,
        }

        #split = generate_split(dataset)

        # train
        if self.model == "rawlsgcn_grad":
            dataset.preprocess(type="laplacian")
        elif self.model == "rawlsgcn_graph":
            #dataset.preprocess(type="doubly_stochastic_laplacian")
            dataset.preprocess(type="laplacian")
        else:
            raise ValueError("Invalid model name!")




        configs = {
            "model": self.model,
            "num_epoch": self.num_epoch,
            "hidden": self.hidden,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "loss": self.loss,
            "seed": seed
        }

        self.trainer=self.run_exp(dataset, split, configs)

        self.trainer.train()

    def predict(self):
        return self.trainer.test()
