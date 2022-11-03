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
from sklearn.metrics import f1_score
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


from gensim.models import Word2Vec


from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

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

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

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
    def run(self, adj_matrix, number_walks=5, representation_size=64, seed=0, walk_length=20, window_size=5, workers=5, pmodified=1.0):
        self.number_walks=number_walks
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
        G = from_numpy(adj_matrix, undirected=True)
        num_walks = len(G.nodes()) * self.number_walks
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * self.walk_length
        print("Data size (walks*length): {}".format(data_size))


        print("Walking...")
        walks = build_deepwalk_corpus(G, num_paths=self.number_walks,
                                      path_length=self.walk_length, p_modified=self.pmodified,
                                      alpha=0, rand=random.Random(self.seed))
        print("Training...")
        model = Word2Vec(walks, vector_size=self.representation_size, window=self.window_size, min_count=0, sg=1,
                         hs=1, workers=self.workers)

        print(model.wv.vectors.shape)
        return model.wv.vectors
        # model.wv.save_word2vec_format(self.output)
    
    def fit(self,adj_matrix, labels, idx_train, number_walks=5, representation_size=64, seed=0, walk_length=20, window_size=5, workers=5, pmodified=1.0):
        self.embs=self.run(adj_matrix, number_walks, representation_size, seed, walk_length, window_size, workers, pmodified)
        self.labels=labels
        self.lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            self.embs[idx_train], labels[idx_train])

    def predict(self,idx_test):
        pred = self.lgreg.predict(self.embs[idx_test])
        score = f1_score(self.labels[idx_test], pred, average='micro')
        return score
