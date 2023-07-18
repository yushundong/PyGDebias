import os
from collections import defaultdict

import numpy as np
import networkx as nx
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import random
from tqdm import tqdm
import torch

import argparse
import scipy.sparse as sp
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

def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None,
                            first_travel_key: str = None, quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                group2neighbors = d_graph[walk[-1]][neighbors_key]
                all_possible_groups = [group for group in group2neighbors if len(group2neighbors[group]) > 0]
                random_group = np.random.choice(all_possible_groups, size=1)[0]
                walk_options = walk_options[random_group]

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key][random_group]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][random_group][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


class FairWalk():
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    GROUP_KEY = 'group'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def fit(self, adj,  labels, idx_train, sens, dimensions: int = 64, walk_length: int = 20, num_walks: int = 5, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None):
        """
        Initiates the FairWalk object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = nx.from_numpy_array(adj.to_dense().numpy())
        n = len(self.graph.nodes())
        node2group = {node: group for node, group in zip(self.graph.nodes(), (5 * np.random.random(n)).astype(int))}
        nx.set_node_attributes(self.graph, node2group, 'group')

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        #self._precompute_probabilities()
        #self.walks = self._generate_walks()
        #self.embs=gensim.models.Word2Vec(self.walks,size=self.dimensions, window=5,min_count=0, sg=1,
        #                                 hs=1).wv.vectors

        from torch_geometric.utils import from_scipy_sparse_matrix
        edge_index = from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        model = Node2Vec(edge_index, embedding_dim=128, walk_length=walk_length,
                         context_size=10, walks_per_node=num_walks,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(device)
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
        model.eval()
        z = model()
        self.embs = z.detach().cpu().numpy()
#
        #self.embs=torch.nn.functional.normalize(torch.tensor(self.embs).float(),dim=-1)

        self.labels=labels
        print('label_size', self.labels.shape)
        self.sens=sens.squeeze()
        #self.lgreg_label = LogisticRegression(random_state=0, class_weight='balanced', max_iter=5000).fit(
        #    self.embs[idx_train], labels[idx_train])

        self.lgreg_label=LogisticRegression(random_state=0,
                                             C=1.0, multi_class = 'auto',
                                             solver='lbfgs',
                                             max_iter=1000).fit(self.embs[idx_train], labels[idx_train])

        self.lgreg_sens = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            self.embs[idx_train], sens.squeeze()[idx_train])

        self.Linear1=torch.nn.Linear(self.embs.shape[-1], 32)
        self.Linear2 = torch.nn.Linear(32, 1)

        #optimizer = torch.optim.Adam(list(self.Linear1.parameters())+list(self.Linear1.parameters()),
        #                             lr=1e-3)
        #loss_fn=torch.nn.BCELoss()

        #for i in range(500):
        #    idx=np.random.choice(idx_train, size=10, replace=False)
        #    optimizer.zero_grad()
        #    loss=loss_fn(torch.nn.functional.sigmoid(self.Linear2(self.Linear1(torch.tensor(self.embs[idx]).float())).squeeze()),
        #                        torch.tensor(self.labels[idx]).float())
#
        #    if i%500==0:
        #        print(self.labels[idx])
        #        print(loss)
        #    loss.backward()
        #    optimizer.step()
#
#
        #pred = (torch.nn.functional.sigmoid(self.Linear2(self.Linear1(torch.tensor(self.embs[idx_train]))))>0.5).float().squeeze()
        #ACC = accuracy_score(self.labels[idx_train], pred.detach().numpy())
        #print(self.embs)
        #print(pred)
        #print('train acc', ACC)

    def classify(self, idx_test, idx_val):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.semi_supervised import LabelPropagation
        from sklearn.metrics import pairwise_distances

        res_total = []
        res_1_total = []
        res_0_total = []
        res_diff_total = []
        res_var_total = []
        for iter in range(1):  # 200

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

            X_train = X

            X_test = X[idx_test]
            y_train = y.copy()
            y_train[idx_test] = -1
            y_test = y[idx_test]
            z_test = z[idx_test]

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

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(y_pred,
                                                                                                       y_test,
                                                                                                       z_test)

        SP, EO = self.fair_metric(np.array(y_pred), y_test, z_test)

        print(SP, EO)
        loss_fn = torch.nn.BCELoss()
        self.val_loss = loss_fn(torch.FloatTensor(y_pred), torch.tensor(y_test).float()).item()

        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO

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

        pred = self.lgreg_label.predict(self.embs[idx_test])
        #pred = (torch.nn.functional.sigmoid(
        #    self.Linear2(self.Linear1(torch.tensor(self.embs[idx_test])))) > 0.5).float().squeeze()

        F1 = f1_score(self.labels[idx_test], pred, average='micro')
        ACC=accuracy_score(self.labels[idx_test], pred,)
        AUCROC=roc_auc_score(self.labels[idx_test], pred)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1=self.predict_sens_group(idx_test)


        SP, EO=self.fair_metric(np.array(pred), self.labels[idx_test].cpu().numpy(), self.sens[idx_test].cpu().numpy())

        pred = self.lgreg_label.predict_proba(self.embs[idx_val])
        loss_fn=torch.nn.BCELoss()
        self.val_loss=loss_fn(torch.FloatTensor(pred).softmax(-1)[:,-1], torch.tensor(self.labels[idx_val]).float()).item()


        print(F1)
        print(ACC)
        print(SP, EO)
        print(1/0)



        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO



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



    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        node2groups = nx.get_node_attributes(self.graph, self.GROUP_KEY)
        groups = np.unique(list(node2groups.values()))

        # Init probabilities dict
        for node in self.graph.nodes():
            for group in groups:
                if self.PROBABILITIES_KEY not in d_graph[node]:
                    d_graph[node][self.PROBABILITIES_KEY] = dict()
                if group not in d_graph[node][self.PROBABILITIES_KEY]:
                    d_graph[node][self.PROBABILITIES_KEY][group] = dict()

        for source in nodes_generator:
            for current_node in self.graph.neighbors(source):

                unnormalized_weights = list()
                d_neighbors = list()
                neighbor_groups = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)
                    if self.GROUP_KEY not in self.graph.nodes[destination]:
                        raise Exception('no group attribute')
                    neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

                unnormalized_weights = np.array(unnormalized_weights)
                d_neighbors = np.array(d_neighbors)
                neighbor_groups = np.array(neighbor_groups)

                for group in groups:
                    cur_unnormalized_weights = unnormalized_weights[neighbor_groups == group]
                    cur_d_neighbors = d_neighbors[neighbor_groups == group]

                    # Normalize
                    d_graph[current_node][self.PROBABILITIES_KEY][group][
                        source] = cur_unnormalized_weights / cur_unnormalized_weights.sum()

                    # Save neighbors
                    d_graph[current_node].setdefault(self.NEIGHBORS_KEY, {})[group] = list(cur_d_neighbors)

            # Calculate first_travel weights for source
            first_travel_weights = []
            first_travel_neighbor_groups = []
            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))
                first_travel_neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

            first_travel_weights = np.array(first_travel_weights)
            first_travel_neighbor_groups = np.array(first_travel_neighbor_groups)
            d_graph[source][self.FIRST_TRAVEL_KEY] = {}
            for group in groups:
                cur_first_travel_weights = first_travel_weights[first_travel_neighbor_groups == group]
                d_graph[source][self.FIRST_TRAVEL_KEY][group] = cur_first_travel_weights / cur_first_travel_weights.sum()

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks
