"""
Link Prediction Task: predicting the existence of an edge between two arbitrary nodes in a graph.
===========================================
-  Model: DGL-based graphsage and gat encoder (and many more)
-  Loss: cross entropy. You can modify the loss as you want
-  Metric: AUC
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import time
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import argparse
import os
import pandas as pd
import random
import warnings
import tqdm
from dgl.data import DGLDataset
import urllib.request
import zipfile
import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
RAW_FOLDER = './raw_data'
DATA_FOLDER = './processed_data'
WEIGHT_FOLDER = './precomputed_weights'
SENSITIVE_ATTR_DICT = {
    'movielens': ['gender', 'occupation', 'age'],
    'pokec-z': ['gender','region','AGE'],
    'pokec-n': ['gender','region','AGE'],
    'pokec': ['gender','region','AGE']
}
from sklearn.metrics import roc_auc_score
import dgl.data



import dgl
from dgl.nn import GraphConv  # Define a GCN model
from dgl.nn import GATConv  # Define a GAT model
from dgl.nn import SGConv  # Define a SGC model
from dgl.nn import SAGEConv  # Define a GraphSAGE model
import torch
import dgl.function as fn


######################################################################
# build a two-layer SGConv model
class SGC(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(SGC, self).__init__()
        self.conv = SGConv(in_feats=in_dim,
                           out_feats=out_dim,
                           k=2)
        self.graph = graph

    def forward(self, in_feat):
        h = self.conv(self.graph, in_feat)
        return h


######################################################################
# build a two-layer vanilla GCN model
class GCN(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats=in_dim,
                               out_feats=hidden_dim,
                               norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(in_feats=hidden_dim,
                               out_feats=out_dim,
                               norm='both', weight=True, bias=True)
        self.graph = graph

    def forward(self, in_feat):
        h = self.conv1(self.graph, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(self.graph, h)
        return h


######################################################################
# build a two-layer GAT model
class GATLayer(nn.Module):
    def __init__(self, graph, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.graph = graph
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # attention
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, in_feat):
        z = self.fc(in_feat)
        self.graph.ndata['z'] = z
        self.graph.apply_edges(self.edge_attention)
        self.graph.update_all(self.message_func, self.reduce_func)
        return self.graph.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, graph, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(graph, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(graph, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(graph, hidden_dim * num_heads, out_dim, 1)

    def forward(self, in_feat):
        h = self.layer1(in_feat)
        h = F.elu(h)
        h = self.layer2(h)
        return h


######################################################################
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_dim,
                              out_feats=hidden_dim,
                              aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=hidden_dim,
                              out_feats=out_dim,
                              aggregator_type='mean')
        self.graph = graph

    def forward(self, in_feat):
        h = self.conv1(self.graph, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(self.graph, h)
        return h


######################################################################
# build a node2vec model
class Node2vec(nn.Module):
    def __init__(self, graph, in_dim, out_dim):
        super(Node2vec, self).__init__()
        self.embed = torch.nn.Embedding(in_dim, out_dim, sparse=False)
        self.graph = graph

    def forward(self, in_feat):
        h = self.embed(in_feat)
        return h


class DotLinkPredictor(nn.Module):
    """
    Dot product to compute the score of link
    The benefit of treating the pairs of nodes as a graph is that the score
    on edge can be easily computed via the ``DGLGraph.apply_edges`` method
    """

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPLinkPredictor(nn.Module):
    """MLP to predict the score of link
    """

    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_bpr_loss(pos_score, neg_score, pos_weights):
    """Compute bpr loss for pairwise ranking
    """

    diff = (pos_score - neg_score)
    log_likelh = torch.log(1 / (1 + torch.exp(-diff))) * pos_weights

    return -torch.sum(log_likelh) / log_likelh.shape[0]


def compute_entropy_loss(pos_score, neg_score, pos_weights):
    """Compute cross entropy loss for link prediction
    """


    neg_weights = torch.ones(len(neg_score)).to(neg_score.device)
    weights = torch.cat([pos_weights, neg_weights])
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(neg_score.device)



    return F.binary_cross_entropy_with_logits(scores, labels, weights)


def compute_metric(pos_score, neg_score):
    """Compute AUC, NDCG metric for link prediction
    """

    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))  # the probability of positive label
    scores_flip = 1.0 - scores  # the probability of negative label
    y_pred = torch.transpose(torch.stack((scores, scores_flip)), 0, 1)

    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    labels_flip = 1 - labels  # to generate one-hot labels
    y_true = torch.transpose(torch.stack((labels, labels_flip)), 0, 1).int()

    # print(y_true.cpu(), y_pred.cpu())
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    # ndcg = 0
    ndcg = ndcg_score(np.expand_dims(labels.cpu(), axis=0),
                      np.expand_dims(scores.cpu(), axis=0))  # super slow!

    return auc, ndcg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)




# check and set config for valid debiasing method
def config_debias(debias_method, debias_attr):
    # if no debiasing, set debias_attr to none
    if debias_method == 'none':
        if debias_attr != 'none':
            warnings.warn('no debias method specified, debias_attr will be reset as none')
            debias_attr = 'none'

    return debias_method, debias_attr


def process_raw_pokec(raw_folder, processed_folder, data_name):
    print('Converting to csv format...')

    if data_name == 'pokec-z':
        edge_file = '{}/pokec/region_job_relationship.txt'.format(raw_folder)
        node_file = '{}/pokec/region_job.csv'.format(raw_folder)

    elif data_name == 'pokec-n':
        edge_file = '{}/pokec/region_job_2_relationship.txt'.format(raw_folder)
        node_file = '{}/pokec/region_job_2.csv'.format(raw_folder)

    edges = pd.read_csv(edge_file, sep='\t', names=['Src', 'Dst'], engine='python')
    nodes = pd.read_csv(node_file, sep=',', header=0, engine='python')

    print('-- raw data loaded')

    feature_ls = ['Label', 'user_id', 'public',
                  'completion_percentage', 'gender', 'region', 'AGE']

    sensitive_attributes_predefined = SENSITIVE_ATTR_DICT['pokec']

    node_ids = list(nodes['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    nodes['Label'] = nodes['public']
    node_labels = nodes.filter(['Label'])

    edges['Weight'] = np.ones(edges.shape[0])
    edges['Src'].replace(id_map, inplace=True)
    edges['Dst'].replace(id_map, inplace=True)

    node_attributes = nodes.filter(sensitive_attributes_predefined)
    node_features = nodes.drop(columns=feature_ls)


    print('-- feature and attribute filtered')

    node_attributes.to_csv('{}/{}_node_attribute.csv'.format(processed_folder, data_name), sep=',', index=False)
    node_features.to_csv('{}/{}_node_feature.csv'.format(processed_folder, data_name), sep=',', index=False)
    node_labels.to_csv('{}/{}_node_label.csv'.format(processed_folder, data_name), sep=',', index=False)
    edges.to_csv('{}/{}_edge.csv'.format(processed_folder, data_name), sep=',', index=False)

    print('Processed data to {}'.format(processed_folder))


class MyDataset(DGLDataset):
    def __init__(self, data_name, data_folder=DATA_FOLDER, weight_folder=WEIGHT_FOLDER, raw_folder=RAW_FOLDER, adj=None,
                 feats=None, labels=None):
        self.data_name = data_name
        self.data_folder = data_folder
        self.weight_folder = weight_folder
        self.raw_folder = raw_folder
        self.adj = adj
        self.feats = feats
        self.labels = labels
        self.num_classes = 2

        super().__init__(name='customized_dataset')

    def process(self):
        adj = sp.coo_matrix(self.adj.to_dense().numpy())
        feats = self.feats
        labels = self.labels

        node_features = feats.float()
        node_labels = labels.long()
        edge_features = torch.from_numpy(np.ones(adj.row.shape)).float()

        edges_src = torch.from_numpy(adj.row).long()
        edges_dst = torch.from_numpy(adj.col).long()



        g = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        # !Key place to triger UGE-W or not by data_name
        g.edata['weight'] = edge_features if 'debias' in self.data_name else torch.ones_like(edge_features)


        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = dgl.to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)

        # zero in-degree nodes will lead to invalid output value
        # a common practice to avoid this is to add a self-loop
        self.graph = dgl.add_self_loop(g)


        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.graph.ndata['feat'].shape[1]))


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
def construct_link_prediction_data_by_node(data_name='movielens', adj=None, feats=None, labels=None):
    """Construct train/test dataset by node for link prediction

    Parameters
    ----------
    data_name :
        name of dataset

    Returns
    -------
    link_pred_data_dict:
        store all data objects required for link prediction task; dictionary
        {
            train_g :
                graph reconstructed by all training nodes; dgl.graph
            features :
                node feature; torch tensor
            train_pos_g :
                graph reconstructed by positive training edges; dgl.graph
            train_neg_g :
                graph reconstructed by negative training edges; dgl.graph
            test_pos_g :
                graph reconstructed by positive testing edges; dgl.graph
            test_neg_g :
                graph reconstructed by negative testing edges; dgl.graph
            train_pos_weights:
                edge weights which are original 0/1 indicators or precomputed weights for weighting-based uge-w; torch tensor
            train_index_dict:
                index dictionary for negative training edges; dictionary: key=src node, value=index of dst node in train_neg_g
            test_index_dict:
                index dictionary for negative testing edges; dictionary: key=src node, value=index of dst node in test_neg_g
        }

    """

    dataset = MyDataset(data_name=data_name,adj=adj, feats=feats, labels=labels)

    graph = dataset[0]
    features = graph.ndata['feat']

    weights = graph.edata['weight'].numpy().tolist()

    # # Key place to include precomputed weighting for UGE-W
    # if not uge_w:  # use original 0/1 edge weights if do not include uge_w
    #     weights = torch.ones(graph.number_of_edges())
    # else:  # assign precomputed weights for weighting-based debiasing
    #     weights = graph.edata['weight'].numpy().tolist()
    #     print('Precomputed weights for weighting-based debiasing Loaded')

    u, v, eids = graph.edges(form='all')

    # edges grouped by node
    src_nodes = set(u.numpy().tolist())  # all source node idx
    des_nodes = set(v.numpy().tolist())  # all destination node idx
    edge_dict = {}
    eid_dict = {}
    for i in range(int(len(u.numpy().tolist()) / 1)):
        if u.numpy()[i] not in edge_dict:
            edge_dict[u.numpy()[i]] = []
        edge_dict[u.numpy()[i]].append(v.numpy()[i])
        eid_dict[(u.numpy()[i], v.numpy()[i])] = eids.numpy()[i]

    # For each node, split its edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample 20 times more negative examples in both sets
    neg_rate = 20
    test_rate = 0.1
    test_pos_u, test_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_pos_u, train_pos_v = [], []
    train_neg_u, train_neg_v = [], []
    test_eids = []
    train_pos_weights = []
    train_index_dict = {}
    test_index_dict = {}
    pbar = tqdm.tqdm(edge_dict.items())

    k = 0
    for src_n, des_ns in pbar:
        k += 1
        pbar.set_description("Splitting train/test edges by node")

        pos_des_ns = np.random.permutation(des_ns)
        candidate_negs = np.array(list(des_nodes - set(pos_des_ns)))
        all_neg_des_ns_idx = np.random.randint(low=0, high=len(candidate_negs),
                                               size=(len(pos_des_ns), neg_rate))
        all_neg_des_ns = candidate_negs[all_neg_des_ns_idx]

        # split test/train while sampling neg
        test_pos_size = int(len(pos_des_ns) * test_rate)
        for n in range(len(pos_des_ns)):
            # for each pos edge, sample neg_rate neg edges
            neg_des_ns = all_neg_des_ns[n]

            if n < test_pos_size:  # testing set
                test_neg_v += list(neg_des_ns)
                test_neg_u += [src_n] * len(neg_des_ns)
                test_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                test_pos_u += [src_n] * len(neg_des_ns)
                test_eids.append(eid_dict[(src_n, pos_des_ns[n])])
                # store index grouped by node
                test_index_dict[src_n] = [len(test_neg_v) - 1 - i for i in range(len(neg_des_ns))]
            else:  # training set
                train_neg_v += list(neg_des_ns)
                train_neg_u += [src_n] * len(neg_des_ns)
                train_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                train_pos_u += [src_n] * len(neg_des_ns)

                train_pos_weights += [weights[eid_dict[(src_n, pos_des_ns[n])]]] * len(neg_des_ns)
                # store index grouped by node
                train_index_dict[src_n] = [len(train_neg_v) - 1 - i for i in range(len(neg_des_ns))]

        if k > 1000: break

    # tranform to tensor
    test_pos_u, test_pos_v = torch.tensor(test_pos_u).type(torch.int64), torch.tensor(test_pos_v).type(torch.int64)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u).type(torch.int64), torch.tensor(test_neg_v).type(torch.int64)
    train_pos_u, train_pos_v = torch.tensor(train_pos_u).type(torch.int64), torch.tensor(train_pos_v).type(torch.int64)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u).type(torch.int64), torch.tensor(train_neg_v).type(torch.int64)
    test_eids = torch.tensor(test_eids)
    test_eids = test_eids.type(torch.int64)
    train_pos_weights = torch.tensor(train_pos_weights)

    print('Finish constructing train/test set for link Prediction.')
    print('  TrainPosEdges: ', len(train_pos_u))
    print('  TrainNegEdges: ', len(train_neg_u))
    print('  TestPosEdges: ', len(test_pos_u))
    print('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    # print(test_eids)
    train_g = dgl.remove_edges(graph, test_eids)

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    # put all data objects into a dict
    link_pred_data_dict = {
        'train_g': train_g,
        'features': features,
        'train_pos_g': train_pos_g,
        'train_neg_g': train_neg_g,
        'test_pos_g': test_pos_g,
        'test_neg_g': test_neg_g,
        'train_pos_weights': train_pos_weights,
        'train_index_dict': train_index_dict,
        'test_index_dict': test_index_dict
    }

    return link_pred_data_dict


class UGE():
    def fit(self,adj, feats, labels,idx_train, sens, model='gcn', debias_method='uge-w', debias_attr='gender', reg_weight=0.5, loss='entropy',lr=0.01/20,
            weight_decay=5e-4, dim1=64, dim2=32, predictor='dot', seed=0, device=0, epochs=50):
        self.labels=labels
        self.sens=sens.squeeze()
        self.model=model
        self.debias_method=debias_method
        self.debias_attr=debias_attr
        self.reg_weight=reg_weight
        self.loss=loss
        self.lr=lr
        self.weight_decay=weight_decay
        self.dim1=dim1
        self.dim2=dim2
        self.predictor=predictor
        self.seed=seed
        self.device=device
        self.epochs=epochs


        setup_seed(self.seed)
        # set up device
        print('==== Environment ====')
        device = torch.device('cuda:{}'.format(self.device) if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(1)  # limit cpu use
        print('  pytorch version: ', torch.__version__)

        # check if the parsed debiasing arguments are valid; config debiasing env
        debias_method, debias_attr = config_debias(self.debias_method, self.debias_attr)

        ######################################################################
        # load and construct data for link prediction task
        # ! if weighting-based debiasing method is triggered, train_weights will load precomputed weights for debiasing



        link_pred_data_dict = construct_link_prediction_data_by_node(adj=adj, feats=feats, labels=labels)

        graph = link_pred_data_dict['train_g'].to(device)
        features = link_pred_data_dict['features'].to(device)
        train_pos_g = link_pred_data_dict['train_pos_g'].to(device)
        train_neg_g = link_pred_data_dict['train_neg_g'].to(device)
        test_pos_g = link_pred_data_dict['test_pos_g'].to(device)
        test_neg_g = link_pred_data_dict['test_neg_g'].to(device)
        train_weights = link_pred_data_dict['train_pos_weights'].to(device)
        n_features = features.shape[1]

        ######################################################################
        # Initialize embedding model

        print('==== Embedding model {} + predictor {} ===='.format(self.model, self.predictor))

        if self.model == 'gcn':
            model = GCN(graph, in_dim=n_features, hidden_dim=self.dim1, out_dim=self.dim2)
        elif self.model == 'gat':
            model = GAT(graph, in_dim=n_features, hidden_dim=self.dim1 // 8, out_dim=self.dim2, num_heads=8)
        elif self.model == 'sgc':
            model = SGC(graph, in_dim=n_features, hidden_dim=self.dim1, out_dim=self.dim2)
        elif self.model == 'sage':
            model = GraphSAGE(graph, in_dim=n_features, hidden_dim=self.dim1, out_dim=self.dim2)
        elif self.model == 'node2vec':
            model = Node2vec(graph, in_dim=features.shape[0], out_dim=self.dim2)
        else:
            raise AssertionError(f"unknown gcn model: {self.model}")

        model = model.to(device)

        # Initialize link predictor
        if self.predictor == 'dot':
            pred = DotLinkPredictor()
        else:
            pred = MLPLinkPredictor(self.dim2)

        pred = pred.to(device)

        optimizer = torch.optim.Adam(itertools.chain(model.parameters(),
                                                     pred.parameters()), lr=self.lr)



        print('==== Training with {} debias method ===='.format(debias_method))

        dur = []
        cur = time.time()
        pbar = tqdm.tqdm(range(self.epochs))
        for e in pbar:
            pbar.set_description("learning node embedding")

            model.train()

            # forward propagation on training set
            if self.model == 'node2vec':
                input_x = torch.Tensor([i for i in range(features.shape[0])]).long().to(device)
            else:
                input_x = features

            h = model(input_x)

            train_pos_score = pred(train_pos_g, h)
            train_neg_score = pred(train_neg_g, h)

            if self.loss == 'entropy':
                loss = compute_entropy_loss(train_pos_score, train_neg_score, train_weights)
            elif self.loss == 'bpr':
                loss = compute_bpr_loss(train_pos_score, train_neg_score, train_weights)
            else:
                raise AssertionError(f"unknown loss: {self.loss}")

            #print(train_pos_score)
            #print(train_neg_score)
            #print(train_weights)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dur.append(time.time() - cur)
            cur = time.time()

            #if e % 50 == 49:
                # evaluation on test set
                # print ('evaluating at epoch {}...'.format(e))
                #model.eval()
                #with torch.no_grad():
                    #test_pos_score = pred(test_pos_g, h)
                    #test_neg_score = pred(test_neg_g, h)
                    #test_auc, test_ndcg = compute_metric(test_pos_score, test_neg_score)
                    #train_auc, train_ndcg = compute_metric(train_pos_score, train_neg_score)

                #print(
                #    "-- Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Train NDCG {:.4f} | Test AUC {:.4f} | Test NDCG {:.4f} | Time {:.4f}".format(
                #        e, loss.item(), train_auc, train_ndcg, test_auc, test_ndcg, dur[-1]))

        # Save learned embedding dynamically
        self.embs = h.detach().cpu().numpy()

        print(self.embs.shape)
        print(labels.shape)

        self.lgreg = LogisticRegression(penalty='none',random_state=1, class_weight='balanced',solver='lbfgs', max_iter=50000).fit(
            self.embs[idx_train], labels[idx_train])



        self.lgreg_sens = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            self.embs[idx_train], self.sens[idx_train])


        pred = self.lgreg.predict(self.embs[idx_train])
        print(pred)
        print(self.labels[idx_train])

        F1 = f1_score(self.labels[idx_train], pred, average='micro')
        ACC=accuracy_score(self.labels[idx_train], pred,)
        AUCROC=roc_auc_score(self.labels[idx_train], pred)


        print(F1, ACC)




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

        pred = self.lgreg.predict(self.embs[idx_test])
        F1 = f1_score(self.labels[idx_test], pred, average='micro')
        ACC=accuracy_score(self.labels[idx_test], pred,)

        if self.labels.max()>1:
            AUCROC=0
        else:
            AUCROC=roc_auc_score(self.labels[idx_test], pred)

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1=self.predict_sens_group(idx_test)

        SP, EO=self.fair_metric(np.array(pred), self.labels[idx_test].cpu().numpy(), self.sens[idx_test].cpu().numpy())


        pred = self.lgreg.predict_proba(self.embs[idx_val])
        loss_fn=torch.nn.BCELoss()
        self.val_loss=loss_fn(torch.FloatTensor(pred).softmax(-1)[:,-1], torch.tensor(self.labels[idx_val]).float()).item()

        print(F1, ACC)
        print(SP, EO)


        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO



    def predict_sens_group(self, idx_test):
        pred = self.lgreg.predict(self.embs[idx_test])

        result=[]
        for sens in [0,1]:
            F1 = f1_score(self.labels[idx_test][self.sens[idx_test]==sens], pred[self.sens[idx_test]==sens], average='micro')
            ACC=accuracy_score(self.labels[idx_test][self.sens[idx_test]==sens], pred[self.sens[idx_test]==sens],)
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                AUCROC=roc_auc_score(self.labels[idx_test][self.sens[idx_test]==sens], pred[self.sens[idx_test]==sens])
            result.extend([ACC, AUCROC,F1])

        return result

    def predict_sens(self,idx_test):
        pred = self.lgreg_sens.predict(self.embs[idx_test])
        F1 = f1_score(self.labels[idx_test], pred, average='micro')
        ACC=accuracy_score(self.labels[idx_test], pred,)
        AUCROC=roc_auc_score(self.labels[idx_test], pred)
        return ACC, AUCROC, F1
