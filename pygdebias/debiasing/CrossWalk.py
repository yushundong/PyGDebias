#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
import multiprocessing
import pickle
import scipy.sparse as sp
import torch
from gensim.models import Word2Vec


from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

import argparse

from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=True,
                    help='Enable CUDA training.')
parser.add_argument('--dataset', type=str, default="cora", help='One dataset from xx.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=1e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
dataset_name = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = 'cpu'
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'


def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32)



class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if p == 1.0 and q == 1.0:
            #self.random_walk_fn = torch.ops.pyg.random_walk
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        # else:
        #     if p == 1.0 and q == 1.0:
        #         raise ImportError(f"'{self.__class__.__name__}' "
        #                           f"requires either the 'pyg-lib' or "
        #                           f"'torch-cluster' package")
        #     else:
        #         raise ImportError(f"'{self.__class__.__name__}' "
        #                           f"requires the 'torch-cluster' package")

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        multi_class: str = 'auto',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')




p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)
        self.edge_weights = None
        self.attr = None
        # self.border_score = None
        self.border_distance = None

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def random_walk(self, path_length, p_modified, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]
        modified = np.random.rand() < p_modified
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    if not modified:
                        path.append(rand.choice(G[cur]))
                    elif G.edge_weights is None:
                        path.append(rand.choice(G[cur]))
                    elif isinstance(G.edge_weights, str) and (G.edge_weights.startswith('prb_')):
                        tmp = G.edge_weights.split('_')
                        p_rb, p_br = float(tmp[1]), float(tmp[3])
                        l_1 = [u for u in G[cur] if G.attr[u] == G.attr[cur]]
                        l_2 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
                        if (len(l_1) == 0) or (len(l_2) == 0):
                            path.append(rand.choice(G[cur]))
                        else:
                            p = p_rb if G.attr[cur] == 1 else p_br
                            if np.random.rand() < p:
                                path.append(rand.choice(l_2))
                            else:
                                path.append(rand.choice(l_1))
                    elif isinstance(G.edge_weights, str) and G.edge_weights.startswith('pch_'):
                        p_ch = float(G.edge_weights.split('_')[1])
                        if G.border_distance[cur] == 1:
                            l_1 = [u for u in G[cur] if G.attr[u] == G.attr[cur]]
                            l_2 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
                        else:
                            l_1 = [u for u in G[cur] if G.border_distance[u] >= G.border_distance[cur]]
                            l_2 = [u for u in G[cur] if G.border_distance[u] < G.border_distance[cur]]
                        if (len(l_1) == 0) or (len(l_2) == 0):
                            path.append(rand.choice(G[cur]))
                        else:
                            if np.random.rand() < p_ch:
                                path.append(rand.choice(l_2))
                            else:
                                path.append(rand.choice(l_1))
                    elif isinstance(G.edge_weights, str) and G.edge_weights == 'random':
                        path.append(rand.choice([v for v in G]))
                    elif isinstance(G.edge_weights, str) and G.edge_weights.startswith('smartshortcut'):
                        p_sc = float(G.edge_weights.split('_')[1])
                        if np.random.rand() < p_sc:
                            l_1 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
                            if len(l_1) == 0:
                                l_1 = [v for v in G if G.attr[v] != G.attr[cur]]
                            path.append(rand.choice(l_1))
                        else:
                            path.append(rand.choice(G[cur]))
                    else:
                        path.append(np.random.choice(G[cur], 1, p=G.edge_weights[cur])[0])
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]
def from_numpy(x, undirected=True):
    G = Graph()
    print(x.shape)
    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    for i in range(x.shape[0]):
        G[i].append(i)

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def build_deepwalk_corpus(G, num_paths, path_length, p_modified, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, p_modified=p_modified, rand=rand, alpha=alpha, start=node))

    return walks


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()

class CrossWalk():
    def run(self, adj_matrix, number_walks=5, representation_size=64, seed=0, walk_length=20, window_size=5, workers=1, pmodified=1.0):
        self.number_walks=int(number_walks)
        #parser.add_argument('--number-walks', default=5, type=int,
        #                    help='Number of random walks to start at each node')
        self.representation_size=representation_size
        #parser.add_argument('--representation-size', default=64, type=int,
        #                    help='Number of latent dimensions to learn for each node.')
        self.seed=seed
        #parser.add_argument('--seed', default=0, type=int,
        #                    help='Seed for random walk generator.')
        self.walk_length=walk_length
        #parser.add_argument('--walk-length', default=20, type=int,
        #                    help='Length of the random walk started at each node')
        self.window_size=window_size
        #parser.add_argument('--window-size', default=5, type=int,
        #                    help='Window size of skipgram model.')
        self.workers=workers
        #parser.add_argument('--workers', default=5, type=int,
        #                    help='Number of parallel processes.')
        self.pmodified=pmodified
        #parser.add_argument('--pmodified', default=1.0, type=float, help='Probability of using the modified graph')
        return self.process(adj_matrix)

    def process(self, adj_matrix):
        G = from_numpy(sp.coo_matrix(adj_matrix.to_dense().numpy()), undirected=True)
        num_walks = len(G.nodes()) * self.number_walks
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * self.walk_length
        print("Data size (walks*length): {}".format(data_size))


        print("Walking...")
        walks = build_deepwalk_corpus(G, num_paths=self.number_walks,
                                      path_length=self.walk_length, p_modified=self.pmodified,
                                      alpha=0, rand=random.Random(self.seed))
        print("Training...")
        model = Word2Vec(walks, size=self.representation_size, window=self.window_size, min_count=0, sg=1,
                         hs=1, workers=self.workers)

        print(model.wv.vectors.shape)
        return model.wv.vectors
        # model.wv.save_word2vec_format(self.output)



    def classify(self,idx_test, idx_val):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.semi_supervised import LabelPropagation
        from sklearn.metrics import pairwise_distances

        res_total = []
        res_1_total = []
        res_0_total = []
        res_diff_total = []
        res_var_total = []
        for iter in range(1): #200

            print('iter: ', iter)
            run_i = 1 + np.mod(iter, 5)

            emb, dim = self.embs, self.embs.shape[-1]
            labels = self.labels
            sens_attr = self.sens

            assert len(labels) == len(emb) == len(sens_attr)

            n = len(emb)

            X = np.zeros([n, dim])
            y = np.zeros([n])
            z = np.zeros([n])
            for i, id in enumerate(emb):
                X[i, :] = np.array(emb[i])
                y[i] = labels[i]
                z[i] = sens_attr[i]

            #idx = np.arange(n)
            #np.random.shuffle(idx)
            #n_train = int(n // 2)
            #X = X[idx, :]
            #y = y[idx]
            #z = z[idx]

            X_train = X

            X_test=X[idx_test]
            y_train=y.copy()
            y_train[idx_test]=-1
            y_test=y[idx_test]
            z_test=z[idx_test]
#
            #X_test = X[n_train:]
            #y_train = np.concatenate([y[:n_train], -1 * np.ones([n - n_train])])
            #y_test = y[n_train:]
            #z_test = z[n_train:]




            g = np.mean(pairwise_distances(X))
            clf = LabelPropagation(gamma=g).fit(X_train, y_train)

            y_pred = clf.predict(X_test)



            res = 100 * np.sum(y_pred == y_test) / y_test.shape[0]

            idx_1 = (z_test == 1)
            res_1 = 100 * np.sum(y_pred[idx_1] == y_test[idx_1]) / np.sum(idx_1)

            idx_0 = (z_test == 0)
            res_0 = 100 * np.sum(y_pred[idx_0] == y_test[idx_0]) / np.sum(idx_0)

            res_diff = np.abs(res_1 - res_0)
            res_var = np.var([res_1, res_0])

            res_total.append(res)
            res_1_total.append(res_1)
            res_0_total.append(res_0)
            res_diff_total.append(res_diff)
            res_var_total.append(res_var)

        res_avg = np.mean(np.array(res_total), axis=0)
        res_1_avg = np.mean(np.array(res_1_total), axis=0)
        res_0_avg = np.mean(np.array(res_0_total), axis=0)
        res_diff_avg = np.mean(np.array(res_diff_total), axis=0)
        res_var_avg = np.mean(np.array(res_var_total), axis=0)


        print(res_avg, ', ', res_1_avg, ', ', res_0_avg, ', ', res_diff_avg, ', ', res_var_avg)

        print(y_pred)

        F1 = f1_score(y_test, y_pred, average='micro')
        ACC = accuracy_score(y_test, y_pred, )
        AUCROC = roc_auc_score(y_test, y_pred)

        print('testing--------------')
        print(F1)
        print(ACC)
        print(AUCROC)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group( y_pred, y_test, z_test)

        SP, EO = self.fair_metric(np.array(y_pred), y_test, z_test)

        print(SP, EO)
        loss_fn=torch.nn.BCELoss()
        self.val_loss=loss_fn(torch.FloatTensor(y_pred), torch.tensor(y_test).float()).item()


        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO


    def fit(self,adj_matrix, feats, labels, idx_train, sens, number_walks=5, representation_size=64, seed=0, walk_length=20, window_size=5, workers=5, pmodified=1.0):
        #self.embs=self.run(adj_matrix, number_walks, representation_size, seed, walk_length, window_size, workers, pmodified)

        from torch_geometric.utils import from_scipy_sparse_matrix

        edge_index=from_scipy_sparse_matrix(sp.coo_matrix(adj_matrix.to_dense().numpy()))[0]


        model = Node2Vec(edge_index, embedding_dim=128, walk_length=walk_length,
                         context_size=10, walks_per_node=number_walks,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(device)
#
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        for epoch in tqdm(range(0, 50)):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


        z = model()

        self.embs=z.detach().cpu().numpy()

        self.idx_train=idx_train
        #self.embs=np.concatenate([self.embs, feats.numpy()], -1)

        self.labels=labels
        self.sens=sens.squeeze()
        #self.lgreg = LogisticRegression(random_state=1, class_weight='balanced', max_iter=100000).fit(
        #    self.embs[idx_train], labels[idx_train])

        print(self.embs.shape)
        print(self.labels.shape)

        self.lgreg=LogisticRegression(random_state=0,
                                             C=1.0, multi_class = 'auto',
                                             solver='lbfgs',
                                             max_iter=1000).fit(self.embs[idx_train], labels[idx_train])

        self.lgreg_sens = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            self.embs[idx_train], self.sens[idx_train])


        self.use_linear=False

        if self.use_linear:
            self.Linear1 = torch.nn.Linear(self.embs.shape[-1], 32)
            self.Linear2 = torch.nn.Linear(32, 1)

            optimizer = torch.optim.Adam(list(self.Linear1.parameters())+list(self.Linear1.parameters()),
                                         lr=1e-2)

            loss_fn=torch.nn.BCELoss()

            for i in range(500):
                idx=np.random.choice(idx_train, size=10, replace=False)
                optimizer.zero_grad()
                loss=loss_fn(torch.nn.functional.sigmoid(self.Linear2(self.Linear1(torch.tensor(self.embs[idx]).float())).squeeze()),
                                    torch.tensor(self.labels[idx]).float())

                if i%500==0:
                    print(self.labels[idx])
                    print(loss)
                loss.backward()
                optimizer.step()


            pred = (torch.nn.functional.sigmoid(self.Linear2(self.Linear1(torch.tensor(self.embs[idx_train]))))>0.5).float().squeeze()
            ACC = accuracy_score(self.labels[idx_train], pred.detach().numpy())
            print(self.embs)
            print(pred)
            print('train acc', ACC)



    def fair_metric(self, pred, labels, sens):


        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                     sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                       sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
        return parity.item(), equality.item()

    def predict(self,idx_test, idx_val):

        return self.classify(idx_test, idx_val)




    def predict_sens_group(self, y_pred, y_test, z_test):
        result=[]
        for sens in [0,1]:
            F1 = f1_score(y_test[z_test==sens], y_pred[z_test==sens], average='micro')
            ACC=accuracy_score(y_test[z_test==sens], y_pred[z_test==sens],)
            AUCROC=roc_auc_score(y_test[z_test==sens], y_pred[z_test==sens])
            result.extend([ACC, AUCROC,F1])

        return result

    def predict_sens(self,idx_test):
        pred = self.lgreg_sens.predict(self.embs[idx_test])
        score = f1_score(self.sens[idx_test], pred, average='micro')
        return score
