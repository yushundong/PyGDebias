import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
from deeprobust.graph.defense.pgd import PGD, prox_operators
import numpy as np
import os
# import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
import math
from tqdm import tqdm
from torch_geometric.utils import dropout_adj, convert
import time
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score



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





def normalize_scipy(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx



def binarize(A_debiased, adj_ori, threshold_proportion):

    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    A_debiased = normalize_scipy(A_debiased)
    return A_debiased



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


def metric_wd(feature, adj_norm, flag, weakening_factor, max_hop):

    feature = (feature / feature.norm(dim=0)).detach().cpu().numpy()
    adj_norm = (0.5 * adj_norm + 0.5 * sp.eye(adj_norm.shape[0])).toarray()
    emd_distances = []
    cumulation = np.zeros_like(feature)

    if max_hop == 0:
        cumulation = feature
    else:
        for i in range(max_hop):
            cumulation += pow(weakening_factor, i) * adj_norm.dot(feature)

    for i in range(feature.shape[1]):
        class_1 = cumulation[torch.eq(flag, 0), i]
        class_2 = cumulation[torch.eq(flag, 1), i]
        emd = wasserstein_distance(class_1, class_2)
        emd_distances.append(emd)

    emd_distances = [0 if math.isnan(x) else x for x in emd_distances]

    if max_hop == 0:
        print('Attribute bias : ')
    else:
        print('Structural bias : ')

    print("Sum of all Wasserstein distance value across feature dimensions: " + str(sum(emd_distances)))
    print("Average of all Wasserstein distance value across feature dimensions: " + str(np.mean(np.array(emd_distances))))

    return emd_distances


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()



class EDITS(nn.Module):

    def __init__(self, lr, weight_decay, nfeat, node_num, nclass, nfeat_out, adj_lambda, layer_threshold=2, dropout=0.1):
        super(EDITS, self).__init__()
        self.x_debaising = X_debaising(nfeat)
        self.layer_threshold = layer_threshold
        self.adj_renew = Adj_renew(node_num, nfeat, nfeat_out, adj_lambda)
        self.fc = nn.Linear(nfeat * (layer_threshold + 1), 1)
        self.lr = lr
        self.optimizer_feature_l1 = PGD(self.x_debaising.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=self.lr, alphas=[5e-6])
        self.dropout = nn.Dropout(dropout)
        G_params = list(self.x_debaising.parameters())
        self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.lr, eps=1e-04, weight_decay=weight_decay)
        self.optimizer_A = torch.optim.RMSprop(self.fc.parameters(), lr=self.lr, eps=1e-04, weight_decay=weight_decay)

    def propagation_cat_new_filter(self, X_de, A_norm, layer_threshold):
        A_norm = A_norm.half()
        X_agg = X_de.half()
        for i in range(layer_threshold):
            X_de = A_norm.mm(X_de)
            X_agg = torch.cat((X_agg, X_de), dim=1)

        return X_agg.half()


    def binarize(A_debiased, adj_ori, threshold_proportion):

        the_con1 = (A_debiased - adj_ori).A
        the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
        the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
        the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
        A_debiased = adj_ori + sp.coo_matrix(the_con1)
        assert A_debiased.max() == 1
        assert A_debiased.min() == 0
        A_debiased = normalize_scipy(A_debiased)
        return A_debiased

    def forward(self, A, X):
        X_de = self.x_debaising(X)
        adj_new = self.adj_renew()
        agg_con = self.propagation_cat_new_filter(X_de.half(), adj_new.half(), layer_threshold=self.layer_threshold).half()  # A_de or A
        D_pre = self.fc(agg_con)
        D_pre = self.dropout(D_pre)
        return adj_new, X_de, D_pre, D_pre, agg_con

    def optimize(self, adj, features, idx_train, sens, epoch, lr, alpha=5e-2, beta=0.1):
        """
        credit: alpha = 5e-2; beta = 0.1;
        german: alpha = 30e-2; beta = 0.8;
        bail: alpha = 3e-2; beta = 1;
        """
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        self.lr = lr
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_A.param_groups:
            param_group["lr"] = lr

        # optimize attribute debiasing module
        # *************************  attribute debiasing  *************************
        self.train()
        self.optimizer_G.zero_grad()
        self.optimizer_feature_l1.zero_grad()
        self.fc.requires_grad_(False)

        if epoch == 0:
            self.adj_renew.fit(adj, self.lr)

        _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)
        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles))

        loss_train = alpha * (X_debiased - features).norm(2) + beta * adv_loss

        loss_train.backward()
        self.optimizer_G.step()
        
        self.optimizer_feature_l1.step()

        # optimize structural debiasing module
        # *************************  structural debiasing  *************************
        _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)

        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)

        adv_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles))
        self.adj_renew.train_adj(X_debiased, adj, adv_loss, epoch, lr)

        # *************************  PGD  *************************
        param = self.state_dict()
        zero = torch.zeros_like(param["x_debaising.s"])
        one = torch.ones_like(param["x_debaising.s"])
        param["x_debaising.s"] = torch.where(param["x_debaising.s"] > 1, one, param["x_debaising.s"])
        param["x_debaising.s"] = torch.where(param["x_debaising.s"] < 0, zero, param["x_debaising.s"])
        # param["x_debaising.s"] = torch.clamp(param["x_debaising.s"], min=0, max=1)
        self.load_state_dict(param)

        # optimize WD approximator
        # *************************  optimize WD approximator  *************************
        for i in range(8):
            self.fc.requires_grad_(True)
            self.optimizer_A.zero_grad()
            _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)

            positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
            negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)

            loss_train = torch.mean(positive_eles) - torch.mean(negative_eles)
            loss_train.backward()
            self.optimizer_A.step()
            for p in self.fc.parameters():
                p.data.clamp_(-0.02, 0.02)



    def fit(self, adj, features, sens, epochs, lr, idx_train, idx_val, normalize=True, k=-1, device='cuda', half=True, truncation=4):

        """
        Args:
            adj:
            features:
            sens:
            k: which dimension represents the sensitive feature dimension
            epochs:
            lr:
            device:
            half:
            truncation:

        """

        features1 = features
        print("****************************Before debiasing****************************")
        # if args.dataset != 'german':
        #     preserve = features
        #     features1 = feature_norm(features)
        #     if args.dataset == 'credit':
        #         features1[:, 1] = preserve[:, 1]  # credit
        #     elif args.dataset == 'bail':
        #         features1[:, 0] = preserve[:, 0]  # bail
        if normalize:
            features1 = feature_norm(features)
        if k >= 0:
            features1[:, k] = features[:, k]


        metric_wd(features1, normalize_scipy(adj), sens, 0.9, 0)
        metric_wd(features1, normalize_scipy(adj), sens, 0.9, 2)
        print("****************************************************************************")

        features_preserve = features.clone()
        features = features / features.norm(dim=0)
        adj_preserve = adj
        adj = sparse_mx_to_torch_sparse_tensor(adj)


        if half:
            self.cuda().half()
            adj = adj.cuda().half()
            features = features.cuda().half()
            features_preserve = features_preserve.cuda().half()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            sens = sens.cuda()
        else:
            self.to(device)
            adj = adj.to(device)
            features = features.to(device)
            features_preserve = features_preserve.to(device)
            idx_train = idx_train.to(device)
            idx_val = idx_val.to(device)
            sens = sens.to(device)

        A_debiased, X_debiased = adj, features
        val_adv = []
        for epoch in tqdm(range(epochs)):
            if epoch > 400:
                lr = 0.001
            self.train()
            self.optimize(adj, features, idx_train, sens, epoch, lr)
            A_debiased, X_debiased, predictor_sens, show, _ = self.forward(adj, features)
            positive_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] > 0)
            negative_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] <= 0)
            loss_val = - (torch.mean(positive_eles) - torch.mean(negative_eles))
            val_adv.append(loss_val.data)

        param = self.state_dict()

        indices = torch.argsort(param["x_debaising.s"])[:truncation]
        for i in indices:
            features_preserve[:, i] = torch.zeros_like(features_preserve[:, i])
        self.X_debiased = features_preserve
        self.adj1 = sp.csr_matrix(A_debiased.detach().cpu().numpy())




        
        
    def predict(self, adj_ori, labels, sens, idx_train, idx_val, idx_test, epochs, lr, nhid, dropout, weight_decay, model='GCN', device='cuda', threshold_proportion = 0.015):

        """
        GCN: {credit: 0.02, german: 0.29, bail: 0.015}
        """

        A_debiased, features = self.adj1, self.X_debiased
        the_con1 = (A_debiased - adj_ori).A
        the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
        the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
        the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
        A_debiased = adj_ori + sp.coo_matrix(the_con1)
        assert A_debiased.max() == 1
        assert A_debiased.min() == 0
        features = features[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
        A_debiased = normalize_scipy(A_debiased)

        print("****************************After debiasing****************************")
        metric_wd(features, A_debiased, sens, 0.9, 0)
        metric_wd(features, A_debiased, sens, 0.9, 2)
        print("****************************************************************************")
        X_debiased = features.float()
        edge_index = convert.from_scipy_sparse_matrix(A_debiased)[0].cuda()


        if model != 'GCN':
            return "Not Implemented"



        # Model and optimizer
        model = GCN(nfeat=X_debiased.shape[1], nhid=nhid, nclass=labels.max().item(), dropout=dropout).float()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if device == 'cuda':
            model.cuda()
            X_debiased = X_debiased.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()


        def train(epoch, pa, eq, test_f1, val_loss, test_auc):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
            preds = (output.squeeze() > 0).type_as(labels)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
            auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], output.detach().cpu().numpy()[idx_train.cpu().numpy()])
            f1_train = f1_score(labels[idx_train.cpu().numpy()].cpu().numpy(), preds[idx_train.cpu().numpy()].cpu().numpy())
            loss_train.backward()
            optimizer.step()
            _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

            model.eval()
            output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
            preds = (output.squeeze() > 0).type_as(labels)
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
            auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
            f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            # print('Epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'F1_train: {:.4f}'.format(f1_train),
            #       'AUC_train: {:.4f}'.format(auc_roc_train),
            #       'loss_val: {:.4f}'.format(loss_val.item()),
            #       'F1_val: {:.4f}'.format(f1_val),
            #       'AUC_val: {:.4f}'.format(auc_roc_val),
            #       'time: {:.4f}s'.format(time.time() - t))

            if epoch < 15:
                return 0, 0, 0, 1e5, 0
            if loss_val < val_loss:
                val_loss = loss_val.data
                pa, eq, test_f1, test_auc = test(test_f1)
                # print("Parity of val: " + str(pa))
                # print("Equality of val: " + str(eq))
            return pa, eq, test_f1, val_loss, test_auc


        def test(test_f1):
            model.eval()
            output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
            preds = (output.squeeze() > 0).type_as(labels)
            loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
            f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
            test_auc = auc_roc_test
            test_f1 = f1_test
            # print("Test set results:",
            #       "loss= {:.4f}".format(loss_test.item()),
            #       "F1_test= {:.4f}".format(test_f1),
            #       "AUC_test= {:.4f}".format(test_auc))
            parity_test, equality_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(),
                                                    labels[idx_test.cpu().numpy()].cpu().numpy(),
                                                    sens[idx_test.cpu().numpy()].cpu().numpy())
            # print("Parity of test: " + str(parity_test))
            # print("Equality of test: " + str(equality_test))
            return parity_test, equality_test, test_f1, test_auc

        # Train model
        t_total = time.time()
        val_loss = 1e5
        pa = 0
        eq = 0
        test_auc = 0
        test_f1 = 0
        for epoch in tqdm(range(epochs)):
            pa, eq, test_f1, val_loss, test_auc = train(epoch, pa, eq, test_f1, val_loss, test_auc)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("Delta_{SP}: " + str(pa))
        print("Delta_{EO}: " + str(eq))
        print("F1: " + str(test_f1))
        print("AUC: " + str(test_auc))






class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n), requires_grad=True)
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            adj = adj.to_dense()
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

class Adj_renew(nn.Module):

    def __init__(self, node_num, nfeat, nfeat_out, adj_lambda):
        super(Adj_renew, self).__init__()
        self.node_num = node_num
        self.nfeat = nfeat
        self.nfeat_out = nfeat_out
        self.adj_lambda = adj_lambda

        self.reset_parameters()

    def fit(self, adj, lr):
        estimator = EstimateAdj(adj, symmetric=False, device='cuda').to('cuda').half()
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=lr)   # 0.005

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=lr, alphas=[5e-4])  # 5e-4
        self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear],
                  lr=lr, alphas=[1.5])

    def reset_parameters(self):
        pass

    def the_norm(self):
        return self.estimator._normalize(self.estimator.estimated_adj)

    def forward(self):
        return self.estimator.estimated_adj

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


    def train_adj(self, features, adj, adv_loss, epoch, lr):
        for param_group in self.optimizer_adj.param_groups:
            param_group["lr"] = lr

        estimator = self.estimator
        estimator.train()
        self.optimizer_adj.zero_grad()

        delta = estimator.estimated_adj - adj
        loss_fro = torch.sum(delta.mul(delta))
        loss_diffiential = 1 * loss_fro + 15 * adv_loss
        loss_diffiential.backward()
        self.optimizer_adj.step()
        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))
        estimator.estimated_adj.data.copy_((estimator.estimated_adj.data + estimator.estimated_adj.data.transpose(0, 1)) / 2)

        return estimator.estimated_adj


class X_debaising(nn.Module):

    def __init__(self, in_features):
        super(X_debaising, self).__init__()
        self.in_features = in_features
        self.s = Parameter(torch.FloatTensor(in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.s.data.uniform_(1, 1)

    def forward(self, feature):
        return torch.mm(feature, torch.diag(self.s))