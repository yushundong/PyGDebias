from os import error
from aif360.sklearn.metrics.metrics import equal_opportunity_difference
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert, to_networkx
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from torch_geometric.utils.homophily import homophily 
from torch_geometric.utils.subgraph import k_hop_subgraph
import matplotlib as mpl
from networkx.algorithms.centrality import closeness_centrality
import matplotlib.pyplot as plt
import os
import dgl
import random
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import ipdb
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import APPNP as APPNP_base
from deeprobust.graph import utils
from copy import deepcopy

from sklearn.metrics import f1_score, roc_auc_score
from scipy.sparse import coo_matrix

from torch_geometric.utils import convert
from time import time

from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
import copy
from math import sqrt, floor
from inspect import signature
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx, sort_edge_index, dense_to_sparse, to_dense_adj
from ismember import ismember

EPS = 1e-15

# python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1
# --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn
# --dataset german --sim_coeff 0.6 --seed 1

import time
import argparse
import warnings
warnings.filterwarnings('ignore')


from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import ipdb
import torch

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import APPNP as APPNP_base
from aif360.sklearn.metrics import statistical_parity_difference as SPD
from aif360.sklearn.metrics import equal_opportunity_difference as EOD

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score
from torch.nn.utils import spectral_norm


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc1 = spectral_norm(GCNConv(nfeat, nhid))

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
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

class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=2, alpha=0.1, dropout=0.5):
        super(APPNP, self).__init__()
        self.model_name = 'appnp'

        self.lin1 = torch.nn.Linear(nfeat, nhid)
        self.lin2 = torch.nn.Linear(nhid, nclass)
        self.prop1 = APPNP_base(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x



class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 base_model='gcn', k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == 'gcn': # should take nfeat and nhid
            self.conv = GCN(nfeat=in_channels, nhid=out_channels)
        elif self.base_model == 'sage':
            self.conv = SAGE(nfeat=in_channels, nhid=out_channels, dropout=0.5)
        elif self.base_model == 'appnp':
            self.conv = APPNP(nfeat=in_channels,nhid=out_channels, nclass=1)

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


class SSF(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 sim_coeff: float = 0.5, nclass: int=1):
        super(SSF, self).__init__()
        self.encoder: Encoder = encoder
        self.sim_coeff: float = sim_coeff

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden)
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
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
        return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    def D(self, x1, x2): # negative cosine similarity
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

        l1 = self.D(h1[idx], p2[idx])/2
        l2 = self.D(h2[idx], p1[idx])/2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff*(l1+l2), l3

    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1

        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)

        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

        return parity.item(), equality.item()

    def predict(self, emb):

        # projector
        p1 = self.projection(emb)

        # predictor
        h1 = self.prediction(p1)

        # classifier
        c1 = self.classifier(emb)

        return c1

    def linear_eval(self, emb, labels, idx_train, idx_test):
        x = emb.detach()
        classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
        classifier = classifier.to('cuda')
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
        for i in range(1000):
            optimizer.zero_grad()
            preds = classifier(x[idx_train])
            loss = F.cross_entropy(preds, labels[idx_train])
            loss.backward()
            optimizer.step()
            if i%100==0:
                print(loss.item())
        classifier.eval()
        preds = classifier(x[idx_test]).argmax(dim=1)
        correct = (preds == labels[idx_test]).sum().item()
        return preds, correct/preds.shape[0]

def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x

def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y,idx_val,device,sim_coeff):
    '''
    A supporting function for the main function
    '''

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = sim_coeff*(l1+l2) ######

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4

# Encoder output
# model = ['gcn','sage']

def nifty(features,edge_index,labels,device,sens,sens_idx,idx_train,idx_test,idx_val,num_class,lr,weight_decay,self,sim_coeff):
    '''
    Main Function for NIFTY. Choose 'encode' to be 'gcn' or 'sage' or 'appnp' to comply with training.
    Input: listed above. Mostly from self. Some additional been set default value.
    Output: accuracy, f1, parity, counterfactual fairness
    '''
    encoder = Encoder(in_channels=features.shape[1], out_channels=self.hidden, base_model=self.model).to(device)
    model = SSF(encoder=encoder, num_hidden=self.hidden, num_proj_hidden=self.proj_hidden, sim_coeff=self.sim_coeff, nclass=num_class).to(device)
    val_edge_index_1 = dropout_adj(edge_index.to(device), p=self.drop_edge_rate_1)[0]
    val_edge_index_2 = dropout_adj(edge_index.to(device), p=self.drop_edge_rate_2)[0]
    val_x_1 = drop_feature(features.to(device), self.drop_feature_rate_1, sens_idx, sens_flag=False)
    val_x_2 = drop_feature(features.to(device), self.drop_feature_rate_2, sens_idx)
    par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
    par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
    optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
    optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    # Fairness Training
    t_total = time.time()
    best_loss = 100
    best_acc = 0
    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)


    for epoch in range(self.epochs+1):
        t = time.time()

        sim_loss = 0
        cl_loss = 0
        rep = 1
        # Lipschtz weight normalization
        for _ in range(rep):
            model.train()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            edge_index_1 = dropout_adj(edge_index, p=self.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=self.drop_edge_rate_2)[0]
            x_1 = drop_feature(features, self.drop_feature_rate_2, sens_idx, sens_flag=False)
            x_2 = drop_feature(features, self.drop_feature_rate_2, sens_idx)
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            # projector
            p1 = model.projection(z1)
            p2 = model.projection(z2)

            # predictor
            h1 = model.prediction(p1)
            h2 = model.prediction(p2)

            l1 = model.D(h1[idx_train], p2[idx_train])/2
            l2 = model.D(h2[idx_train], p1[idx_train])/2
            sim_loss += self.sim_coeff*(l1+l2)

        # Fairness Training
        (sim_loss/rep).backward()
        optimizer_1.step()

        # classifier
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        c1 = model.classifier(z1)
        c2 = model.classifier(z2)

        # Binary Cross-Entropy
        l3 = F.binary_cross_entropy_with_logits(c1[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2
        l4 = F.binary_cross_entropy_with_logits(c2[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2

        cl_loss = (1-self.sim_coeff)*(l3+l4)
        cl_loss.backward()
        optimizer_2.step()
        loss = (sim_loss/rep + cl_loss)

        # Validation
        model.eval()
        val_s_loss, val_c_loss = ssf_validation(model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2, labels, idx_val, device, sim_coeff)
        emb = model(val_x_1, val_edge_index_1)
        output = model.predict(emb)
        preds = (output.squeeze()>0).type_as(labels)
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])

        # if epoch % 100 == 0:
        #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if (val_c_loss + val_s_loss) < best_loss:
            # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
            best_loss = val_c_loss + val_s_loss
            torch.save(model.state_dict(), f'weights_ssf_{self.model}.pt')

    model.load_state_dict(torch.load(f'weights_ssf_{self.model}.pt'))
    model.eval()
    emb = model(features.to(device), edge_index.to(device))
    output = model.predict(emb)
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model.predict(model(noisy_features.to(device), edge_index.to(device)))

    # Report
    output_preds = (output.squeeze()>0).type_as(labels)
    counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
    noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
    counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
    robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

    parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
    f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

    return auc_roc_test, f1_s, parity, counterfactual_fairness, robustness_score, equality

class GNNExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, lr: float = 0.01,
                 num_hops: Optional[int] = None,
                 log: bool = True, **kwself):
        super().__init__()
        self.model = model
        self.model_p = copy.deepcopy(model)
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.coeffs.update(kwself)

    def __set_masks__(self, x, edge_index, perturbed_edge_index, init="normal"):
        (N, F) = x.size()
        E, E_p = edge_index.size(1), perturbed_edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.perturbed_mask = torch.nn.Parameter(torch.randn(E_p) * std)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.perturbed_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None
        self.perturbed_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __loss__(self, pred, pred_perturb):

        loss = torch.norm(pred - pred_perturb, 1)

        return loss

    def explain_graph(self, x, edge_index, perturbed_edge_index, **kwself):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        self:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwself (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.model_p.eval()
        self.__clear_masks__()

        self.__set_masks__(x, edge_index, perturbed_edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.edge_mask, self.perturbed_mask], lr=self.lr)

        for e in range(0, 10):
            #print('gnn_explainer: ' + str(e))
            optimizer.zero_grad()
            out = self.model(x=x, edge_index=edge_index, **kwself)
            out_p = self.model_p(x=x, edge_index=perturbed_edge_index, **kwself)

            loss = self.__loss__(out, out_p)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach()
        perturbed_mask = self.perturbed_mask.detach()
        self.__clear_masks__()
        return edge_mask, perturbed_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class fair_edit_trainer():
    def __init__(self, model=None, dataset=None, optimizer=None, features=None, edge_index=None,
                 labels=None, device=None, train_idx=None, val_idx=None, sens_idx=None, edit_num=10, sens=None,test_idx=None):
        self.model = model
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.edge_index_orign = copy.deepcopy(self.edge_index)
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx=test_idx
        self.edit_num = edit_num
        self.perturbed_edge_index = None
        self.sens_idx = sens_idx
        self.sens = sens
        counter_features = features.clone()
        counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
        self.counter_features = counter_features
        sens_att = self.features[:, self.sens_idx].int()
        sens_matrix_base = sens_att.reshape(-1, 1) + sens_att
        self.sens_matrix_delete = torch.where(sens_matrix_base != 1, 1, 0).fill_diagonal_(0).int()
        self.sens_matrix_add = torch.where(sens_matrix_base == 1, 1, 0).fill_diagonal_(0).int()

    def add_drop_edge_random(self, add_prob=0.001, del_prob=0.01):

        N, F = self.features.size()
        E = self.edge_index.size(1)

        # Get the current graph and filter sensitives based on what is already there
        dense_adj = torch.abs(to_dense_adj(self.edge_index)[0, :, :]).fill_diagonal_(0).int()
        to_delete = torch.logical_and(dense_adj, self.sens_matrix_delete).int()
        to_add = torch.logical_and(dense_adj, self.sens_matrix_add).int()

        # Generate scores
        scores = torch.Tensor(np.random.uniform(0, 1, (N, N))).cuda()

        # DELETE
        masked_scores = scores * to_delete
        masked_scores = torch.triu(masked_scores, diagonal=1)
        num_non_zero = torch.count_nonzero(masked_scores)
        edits_to_make = floor(E * del_prob)
        if num_non_zero < edits_to_make:
            edits_to_make = num_non_zero
        top_delete = torch.topk(masked_scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_delete, N)
        base_start = torch.floor_divide(top_delete, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        delete_indices = torch.stack([end, start])

        # ADD
        masked_scores = scores * to_add
        masked_scores = torch.triu(masked_scores, diagonal=1)
        num_non_zero = torch.count_nonzero(masked_scores)

        edits_to_make = floor(N**2 * add_prob)
        if num_non_zero < edits_to_make:
            edits_to_make = num_non_zero
        top_adds = torch.topk(masked_scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_adds, N)
        base_start = torch.floor_divide(top_adds, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        add_indices = torch.stack([end, start])



        return delete_indices, add_indices


    def perturb_graph(self, deleted_edges, add_edges):

        # Edges deleted from original edge_index
        delete_indices = []
        self.perturbed_edge_index = copy.deepcopy(self.edge_index)
        for edge in deleted_edges.T:
            vals = (self.edge_index == torch.tensor([[edge[0]], [edge[1]]]).cuda())
            sum = torch.sum(vals, dim=0).cpu()
            col_idx = np.where(sum == 2)[0][0]
            delete_indices.append(col_idx)

        delete_indices.sort(reverse=True)
        for col_idx in delete_indices:
            self.perturbed_edge_index = torch.cat((self.edge_index[:, :col_idx], self.edge_index[:, col_idx+1:]), axis=1)

        # edges added to perturbed edge_index
        start_edges = self.perturbed_edge_index.shape[1]
        add_indices = [i for i in range(start_edges, start_edges + add_edges.shape[1], 1)]
        self.perturbed_edge_index = torch.cat((self.perturbed_edge_index, add_edges), axis=1)

        return delete_indices, add_indices

    def fair_graph_edit(self):

        grad_gen = GNNExplainer(self.model)

        # perturb graph (return the ACTUAL edges)
        deleted_edges, added_edges = self.add_drop_edge_random(add_prob=0.5)
        # get indices of pertubations in edge list (indices in particular edge_lists)
        del_indices, add_indices = self.perturb_graph(deleted_edges, added_edges)
        # generate gradients on these perturbations
        edge_mask, perturbed_mask = grad_gen.explain_graph(self.features, self.edge_index, self.perturbed_edge_index)
        added_grads = perturbed_mask[add_indices]
        deleted_grads = edge_mask[del_indices]

        # figure out which perturbations were best
        #print('addedge',added_edges)

        add=False

        if add:
            best_add_score = torch.min(added_grads)
            best_add_idx = torch.argmin(added_grads)
            best_add = added_edges[:, best_add_idx]

        # we want to add edge since better
            if best_add_score < best_delete_score:
                # add both directions since undirected graph
                best_add_comp = torch.tensor([[best_add[1]], [best_add[0]]]).cuda()
                self.edge_index = torch.cat((self.edge_index, best_add.view(2, 1), best_add_comp), axis=1)
        #else: # delete
        best_delete_score = torch.min(deleted_grads)
        best_delete_idx = torch.argmin(deleted_grads)
        best_delete = deleted_edges[:, best_delete_idx]
        val_del = (self.edge_index == torch.tensor([[best_delete[1]], [best_delete[0]]]).cuda())
        sum_del = torch.sum(val_del, dim=0).cuda()
        col_idx_del = np.where(sum_del.cpu() == 2)[0][0]
        self.edge_index = torch.cat((self.edge_index[:, :col_idx_del], self.edge_index[:, col_idx_del+1:]), axis=1)

        best_delete_comp = torch.tensor([[best_delete[1]], [best_delete[0]]]).cuda()
        val_del_comp = (self.edge_index == torch.tensor([[best_delete_comp[1]], [best_delete_comp[0]]]).cuda())
        sum_del_comp = torch.sum(val_del_comp, dim=0).cuda()
        col_idx_del_comp = np.where(sum_del_comp.cpu() == 2)[0][0]
        self.edge_index = torch.cat((self.edge_index[:, :col_idx_del_comp], self.edge_index[:, col_idx_del_comp+1:]), axis=1)

    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)
            #print('epoch: ' + str(epoch))

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))
            f1_train = f1_score(self.labels[self.train_idx].cpu().numpy(), preds[self.train_idx].cpu().numpy())
            loss_train.backward()
            self.optimizer.step()

            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device)).item()


            acc_val = accuracy_score(self.labels[self.val_idx ].cpu().numpy(), preds[self.val_idx ].cpu().numpy())
            #loss_val=-acc_val


            #           Counter factual fairness
            counter_output = self.model(self.counter_features.to(self.device),self.edge_index.to(self.device))
            counter_preds = (counter_output.squeeze()>0).type_as(self.labels)
            fair_score = 1 - (preds.eq(counter_preds)[self.val_idx].sum().item()/self.val_idx.shape[0])
            #           Robustness
            noisy_features = self.features.clone() + torch.ones(self.features.shape).normal_(0, 1).to(self.device)
            noisy_output = self.model(noisy_features, self.edge_index)
            noisy_output_preds = (noisy_output.squeeze()>0).type_as(self.labels)
            robustness_score = 1 - (preds.eq(noisy_output_preds)[self.val_idx].sum().item()/self.val_idx.shape[0])
            parity, equality = fair_metric(preds[self.val_idx].cpu().numpy(), self.labels[self.val_idx].cpu().numpy(), self.sens[self.val_idx].numpy())

            if epoch < self.edit_num:
                self.fair_graph_edit()


            if loss_val < best_loss:
                best_loss = loss_val
                #torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'fairedit', self.dataset))

                # Report
                idx_test=self.test_idx
                labels=self.labels.detach().cpu().numpy()
                sens=self.sens
                output_preds = (output.squeeze() > 0).type_as(self.labels)[idx_test].detach().cpu().numpy()
                #counter_output_preds = (counter_output.squeeze() > 0).type_as(labels)
                #noisy_output_preds = (noisy_output.squeeze() > 0).type_as(labels)

                #auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()],
                #                             output.detach().cpu().numpy()[idx_test.cpu()])
#
                #parity_s, equality_s = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                #                               sens[idx_test].numpy())
                #f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
#
                #acc_s= torch.eq(labels[idx_test],output_preds[idx_test]).cpu().float().mean().item()

                F1 = f1_score(labels[idx_test], output_preds, average='micro')
                ACC=accuracy_score(labels[idx_test], output_preds,)
                AUCROC=roc_auc_score(labels[idx_test], output_preds)

                ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1=self.predict_sens_group(output_preds, idx_test)


                SP, EO=self.fair_metric(output_preds, self.labels[idx_test].detach().cpu().numpy(), self.sens[idx_test].detach().cpu().numpy())

        self.val_loss=best_loss
        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO



        #print("== f1: {} fair: {} robust: {}, parity:{} equility: {}".format(f1_val,fair_score,robustness_score,parity,equality))
        #return auc_roc_test, f1_s ,acc_s,parity_s, equality_s


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


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def flipAdj(edge_idx: torch.Tensor,i,j,n):

    # i,j : edit idx
    # n, num of node in graph
    # edge_idx : torch_tensor, shape [m,2]. m = num of existing edges

    # restore the sparse mat
    data = np.ones(edge_idx.shape[1])
    t_mat = coo_matrix((data,edge_idx.cpu().numpy()),shape=(n,n)).tocsr()

    # flip
    if (t_mat[i,j] == 0):
        t_mat[i,j] = 1.
        t_mat[j,i] = 1.
    else:
        t_mat[i,j] = 0.
        t_mat[j,i] = 0.

    # Change back
    return convert.from_scipy_sparse_matrix(t_mat)[0]



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nclass):
        super(GCN, self).__init__()
        self.model_name = 'gcn'
        self.body1 = GCN_Body(nfeat,nhid,dropout)
        self.body2 = GCN_Body(nhid,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body1(x, edge_index)
        x = self.body2(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nclass):
        super(SAGE, self).__init__()
        self.model_name = 'sage'
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'
        self.fc = nn.Linear(nhid, nclass)

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
        return self.fc(x)

class SAGE_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SAGE_Body, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x



class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=2, alpha=0.1, dropout=0.5):
        super(APPNP, self).__init__()
        self.model_name = 'appnp'

        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclass)
        self.prop1 = APPNP_base(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)

    return idx_map


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
    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
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


class FairEdit():
    def fit(self,adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, model_name='gcn', epochs=100, lr=1e-3, weight_decay=5e-4, hidden=16, dropout=0.5, edit_num=10):
        self.model_name=model_name
        self.epochs=epochs
        self.lr=lr
        self.weight_decay=weight_decay
        self.hidden=hidden
        self.dropout=dropout

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

        edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        num_class = 1#labels.unique().shape[0] - 1

        #### Load Models ####
        if self.model_name == 'gcn':
            model = GCN(nfeat=features.shape[1],
                        nhid=self.hidden,
                        nclass=num_class,
                        dropout=self.dropout)

        elif self.model_name == 'sage':
            model = SAGE(nfeat=features.shape[1],
                         nhid=self.hidden,
                         nclass=num_class,
                         dropout=self.dropout)

        elif self.model_name == 'appnp':
            model = APPNP(nfeat=features.shape[1],
                          nhid=16,
                          nclass=num_class,
                          K=2, alpha=0.1, dropout=self.dropout)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        model = model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)


        trainer = fair_edit_trainer(model=model, dataset=None, optimizer=optimizer,
                                    features=features, edge_index=edge_index,
                                    labels=labels, device=device, train_idx=idx_train,
                                    val_idx=idx_val, sens_idx=sens_idx, sens=sens, test_idx=idx_test, edit_num=edit_num)


            # moved up because training epochs are already incorporated into nifty
        self.trainer=trainer


    def predict(self):
        ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = self.trainer.train(epochs=100)
        self.val_loss=self.trainer.val_loss
        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO

