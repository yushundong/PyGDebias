import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
import dgl
import networkx as nx
from typing import Dict, Tuple
import pickle
from os.path import join, dirname, realpath
import csv
import pickle as pkl


class dataset(object):
    def __init__(self) -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.train_mask_ = None
        self.val_mask_ = None
        self.test_mask_ = None
        self.sens_ = None
    
    def adj(self, datatype: str = 'torch.sparse'):
        assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == 'torch.sparse':
            return self.adj_
        elif datatype == 'scipy.sparse':
            return sp.coo_matrix(self.adj.to_dense())
        elif datatype == 'tf.sparse':
            self.adj_ = self.adj_.to_dense().numpy()
            self.adj_ = tf.convert_to_tensor(self.adj_)
            self.adj_ = tf.sparse.from_dense(self.adj_)
            return self.adj_
        elif datatype == 'np.array':
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError('datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse')
    
    def features(self, datatype: str = 'torch.tensor'):
        if self.features is None:
            return self.features_
        if datatype == 'torch.tensor':
            return self.features_
        elif datatype == 'np.array':
            return self.features_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.features_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')
    
    def labels(self, datatype: str = 'torch.tensor'):
        if self.labels_ is None:
            return self.labels_
        if datatype == 'torch.tensor':
            return self.labels_
        elif datatype == 'np.array':
            return self.labels_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.labels_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')
    
    def train_mask(self, datatype: str = 'torch.tensor'):
        if self.train_mask_ is None:
            return self.train_mask_
        if datatype == 'torch.tensor':
            return self.train_mask_
        elif datatype == 'np.array':
            return self.train_mask_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.train_mask_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')
    
    def val_mask(self, datatype: str = 'torch.tensor'):
        if self.val_mask_ is None:
            return self.val_mask_
        if datatype == 'torch.tensor':
            return self.val_mask_
        elif datatype == 'np.array':
            return self.val_mask_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.val_mask_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')
        
    def test_mask(self, datatype: str = 'torch.tensor'):
        if self.test_mask_ is None:
            return self.test_mask_
        if datatype == 'torch.tensor':
            return self.test_mask_
        elif datatype == 'np.array':
            return self.test_mask_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.test_mask_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')
    
    def sens(self, datatype: str = 'torch.tensor'):
        if self.sens_ is None:
            return self.sens_
        if datatype == 'torch.tensor':
            return self.sens_
        elif datatype == 'np.array':
            return self.sens_.numpy()
        elif datatype == 'tf.tensor':
            return tf.convert_to_tensor(self.sens_.numpy())
        else:
            raise ValueError('datatype should be torch.tensor, tf.tensor, or np.array')



def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not is_sparse:
        sparse_mx=sp.coo_matrix(sparse_mx)
    else:
        sparse_mx=sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class Google(dataset):
    def __init__(self) -> None:
        super().__init__()
        id='111058843129764709244'
        edges_file=open('./dataset/gplus/{}.edges'.format(id))
        edges=[]
        for line in edges_file:
            edges.append([int(one) for one in line.strip('\n').split(' ')])

        feat_file=open('./dataset/gplus/{}.feat'.format(id))
        feats=[]
        for line in feat_file:
            feats.append([int(one) for one in line.strip('\n').split(' ')])

        feat_name_file = open('./dataset/gplus/{}.featnames'.format(id))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip('\n').split(' '))
        names={}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]]=name[1]
            if 'gender' in name[1]:
                print(name)

        #print(feat_name)
        feats=np.array(feats)

        node_mapping={}
        for j in range(feats.shape[0]):
            node_mapping[feats[j][0]]=j

        feats=feats[:,1:]

        print(feats.shape)
        for i in range(len(feat_name)):
            if feats[:,i].sum()>100:
                print(i, feat_name[i], feats[:,i].sum())

        feats=np.array(feats,dtype=float)


        sens=feats[:,0]
        labels=feats[:,164]


        feats=np.concatenate([feats[:,:164],feats[:,165:]],-1)
        feats=feats[:,1:]



        edges=np.array(edges)
        #edges=torch.tensor(edges)
        #edges=torch.stack([torch.tensor(one) for one in edges],0)

        print(len(edges))
        
        node_num=feats.shape[0]
        adj=np.zeros([node_num,node_num])


        for j in range(edges.shape[0]):
            adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        self.features_ = torch.FloatTensor(feats)
        self.sens_ = torch.FloatTensor(sens)
        self.idx_train_ = torch.LongTensor(idx_train)
        self.idx_val_ = torch.LongTensor(idx_val)
        self.idx_test_ = torch.LongTensor(idx_test)
        self.labels_ = torch.LongTensor(labels)
        self.features_ = torch.cat([self.features_, self.sens_.unsqueeze(-1)], -1)

        self.adj_=mx_to_torch_sparse_tensor(adj)


class Facebook(dataset):
    def __init__(self, path: str = './dataset/facebook/') -> None:
        super().__init__()
        edges_file=open('./dataset/facebook/facebook/107.edges')
        edges=[]
        for line in edges_file:
            edges.append([int(one) for one in line.strip('\n').split(' ')])

        feat_file=open('./dataset/facebook/facebook/107.feat')
        feats=[]
        for line in feat_file:
            feats.append([int(one) for one in line.strip('\n').split(' ')])

        feat_name_file = open('./dataset/facebook/facebook/107.featnames')
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip('\n').split(' '))
        names={}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]]=name[1]
            if 'gender' in name[1]:
                print(name)

        print(feat_name)
        feats=np.array(feats)

        node_mapping={}
        for j in range(feats.shape[0]):
            node_mapping[feats[j][0]]=j

        feats=feats[:,1:]

        print(feats.shape)
        #for i in range(len(feat_name)):
        #    print(i, feat_name[i], feats[:,i].sum())

        sens=feats[:,264]
        labels=feats[:,220]

        feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)

        feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)

        edges=np.array(edges)
        #edges=torch.tensor(edges)
        #edges=torch.stack([torch.tensor(one) for one in edges],0)
        print(len(edges))

        node_num=feats.shape[0]
        adj=np.zeros([node_num,node_num])


        for j in range(edges.shape[0]):
            adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        self.features_ = torch.FloatTensor(feats)
        self.sens_ = torch.FloatTensor(sens)
        self.idx_train_ = torch.LongTensor(idx_train)
        self.idx_val_ = torch.LongTensor(idx_val)
        self.idx_test_ = torch.LongTensor(idx_test)
        self.labels_ = torch.LongTensor(labels)

        self.features_=torch.cat([self.features_,self.sens_.unsqueeze(-1)],-1)
        self.adj_=mx_to_torch_sparse_tensor(adj)




class Nba(dataset):
    def __init__(self, dataset_name='nba', predict_attr_specify=None, return_tensor_sparse=True):
        super().__init__()
        if dataset_name!='nba':
            if dataset_name == 'pokec_z':
                dataset = 'region_job'
            elif dataset_name== 'pokec_n':
                dataset = 'region_job_2'
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path="./dataset/pokec/"
            test_idx=False
        else:
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = self.load_pokec(dataset,
                                                                                            sens_attr,
                                                                                            predict_attr if predict_attr_specify==None else predict_attr_specify,
                                                                                            path=path,
                                                                                            label_number=label_number,
                                                                                            sens_number=sens_number,
                                                                                            seed=seed, test_idx=test_idx)

        #adj=adj.todense()
        print(adj.shape)
        print(features.shape)
        print(sens.shape)
        print(labels.shape)
        print(idx_train.shape)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True,return_tensor_sparse=return_tensor_sparse)
        
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens

    def load_pokec(self, dataset, sens_attr, predict_attr, path="../dataset/pokec/", label_number=1000, sens_number=500,
                   seed=19, test_idx=False):
        """Load data"""
        print('Loading {} dataset from {}'.format(dataset, path))

        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int64).reshape(edges_unordered.shape)

        print(len(edges))


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random
        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)):]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class Pokec_z(dataset):
    def __init__(self, dataset_name='pokec_z', predict_attr_specify=None, return_tensor_sparse=True):
        super().__init__()
        if dataset_name!='nba':
            if dataset_name == 'pokec_z':
                dataset = 'region_job'
            elif dataset_name== 'pokec_n':
                dataset = 'region_job_2'
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path="./dataset/pokec/"
            test_idx=False
        else:
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = self.load_pokec(dataset,
                                                                                            sens_attr,
                                                                                            predict_attr if predict_attr_specify==None else predict_attr_specify,
                                                                                            path=path,
                                                                                            label_number=label_number,
                                                                                            sens_number=sens_number,
                                                                                            seed=seed, test_idx=test_idx)

        #adj=adj.todense()
        print(adj.shape)
        print(features.shape)
        print(sens.shape)
        print(labels.shape)
        print(idx_train.shape)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True,return_tensor_sparse=return_tensor_sparse)
        
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens

    def load_pokec(self, dataset, sens_attr, predict_attr, path="../dataset/pokec/", label_number=1000, sens_number=500,
                   seed=19, test_idx=False):
        """Load data"""
        print('Loading {} dataset from {}'.format(dataset, path))

        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int64).reshape(edges_unordered.shape)

        print(len(edges))


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random
        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)):]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

class Pokec_n(dataset):
    def __init__(self, dataset_name='pokec_n', predict_attr_specify=None, return_tensor_sparse=True):
        super().__init__()
        if dataset_name!='nba':
            if dataset_name == 'pokec_z':
                dataset = 'region_job'
            elif dataset_name== 'pokec_n':
                dataset = 'region_job_2'
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path="./dataset/pokec/"
            test_idx=False
        else:
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = self.load_pokec(dataset,
                                                                                            sens_attr,
                                                                                            predict_attr if predict_attr_specify==None else predict_attr_specify,
                                                                                            path=path,
                                                                                            label_number=label_number,
                                                                                            sens_number=sens_number,
                                                                                            seed=seed, test_idx=test_idx)

        #adj=adj.todense()
        print(adj.shape)
        print(features.shape)
        print(sens.shape)
        print(labels.shape)
        print(idx_train.shape)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True,return_tensor_sparse=return_tensor_sparse)
        
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens

    def load_pokec(self, dataset, sens_attr, predict_attr, path="../dataset/pokec/", label_number=1000, sens_number=500,
                   seed=19, test_idx=False):
        """Load data"""
        print('Loading {} dataset from {}'.format(dataset, path))

        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int64).reshape(edges_unordered.shape)

        print(len(edges))


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random
        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)):]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)

        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train



class Twitter(dataset):
    def __init__(self):
        super().__init__()
        edges_file=open('./dataset/twitter/twitter/428333.edges')
        edges=[]
        for line in edges_file:
            edges.append([int(one) for one in line.strip('\n').split(' ')])

        feat_file=open('./dataset/twitter/twitter/428333.feat')
        feats=[]
        for line in feat_file:
            feats.append([int(one) for one in line.strip('\n').split(' ')])

        feat_name_file = open('./dataset/twitter/twitter/428333.featnames')
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip('\n').split(' '))

        print(feat_name)
        names={}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]]=name[1]
            if 'pol' in name[1]:
                print(name)


        feats=np.array(feats)

        node_mapping={}
        for j in range(feats.shape[0]):
            node_mapping[feats[j][0]]=j

        feats=feats[:,1:]

        print(feats.shape)
        for i in range(len(feat_name)):
            if feats[:,i].sum()>30:
                print(i, feat_name[i], feats[:,i].sum())

        sens=feats[:,264]
        labels=feats[:,220]

        feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)

        feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)

        edges=np.array(edges)
        #edges=torch.tensor(edges)

        #edges=torch.stack([torch.tensor(one) for one in edges],0)

        node_num=feats.shape[0]
        adj=np.zeros([node_num,node_num])


        for j in range(edges.shape[0]):
            adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        features = torch.FloatTensor(feats)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        adj=mx_to_torch_sparse_tensor(adj)
        self.adj_=adj
        self.features_=features
        self.labels_=labels
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens   


class Cora(dataset):
    def __init__(self) -> None:
        super().__init__()
        self.cora_label = {
            "Genetic_Algorithms": 0,
            "Reinforcement_Learning": 1,
            "Neural_Networks": 2,
            "Rule_Learning": 3,
            "Case_Based": 4,
            "Theory": 5,
            "Probabilistic_Methods": 6,
        }
        test_ratio=0.1
        G, adj, features, sensitive, test_edges_true, test_edges_false, _ = self.cora(test_ratio=test_ratio)
        print(adj.shape)
        print(features.shape)


        features = torch.FloatTensor(features)
        sens = torch.FloatTensor(sensitive)
        node_num = features.shape[0]

        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        adj=mx_to_torch_sparse_tensor(adj)
        self.adj_=adj
        self.features_=features
        self.labels_=None
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens

    def build_test(self, G: nx.Graph, nodelist: Dict, ratio: float) -> Tuple:
        """
        Split training and testing set for link prediction in graph G.
        :param G: nx.Graph
        :param nodelist: idx -> node_id in nx.Graph
        :param ratio: ratio of positive links that used for testing
        :return: Graph that remove all test edges, list of index for test edges
        """

        edges = list(G.edges.data(default=False))
        num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
        num_test = int(np.floor(num_edges * ratio))
        test_edges_true = []
        test_edges_false = []

        # generate false links for testing
        while len(test_edges_false) < num_test:
            idx_u = np.random.randint(0, num_nodes - 1)
            idx_v = np.random.randint(idx_u, num_nodes)

            if idx_u == idx_v:
                continue
            if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
                continue
            if (idx_u, idx_v) in test_edges_false:
                continue
            else:
                test_edges_false.append((idx_u, idx_v))

        # generate true links for testing
        all_edges_idx = list(range(num_edges))
        np.random.shuffle(all_edges_idx)
        test_edges_true_idx = all_edges_idx[:num_test]
        for test_idx in test_edges_true_idx:
            u, v, _ = edges[test_idx]
            G.remove_edge(u, v)
            test_edges_true.append((self.get_key(nodelist, u), self.get_key(nodelist, v)))

        return G, test_edges_true, test_edges_false

    def get_key(self, dict, value):
        return [k for k, v in dict.items() if v == value][0]

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def cora(self, feat_path="./dataset/cora/cora.content", edge_path="./dataset/cora/cora.cites",
             test_ratio=0.1):
        idx_features_labels = np.genfromtxt(feat_path, dtype=np.dtype(str))
        idx_features_labels = idx_features_labels[idx_features_labels[:, 0].astype(np.int32).argsort()]

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        nodelist = {idx: node for idx, node in enumerate(idx)}
        X = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
        sensitive = np.array(list(map(self.cora_label.get, idx_features_labels[:, -1])))


        G = nx.read_edgelist(edge_path, nodetype=int)
        G, test_edges_true, test_edges_false = self.build_test(G, nodelist, test_ratio)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

        print(len(G.edges))

        print(G)
        print(adj.shape)
        print(X.shape)
        print(sensitive.shape)

        return G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist



class Citeseer(dataset):
    def __init__(self) -> None:
        super().__init__()
        self.cora_label = {
            "Genetic_Algorithms": 0,
            "Reinforcement_Learning": 1,
            "Neural_Networks": 2,
            "Rule_Learning": 3,
            "Case_Based": 4,
            "Theory": 5,
            "Probabilistic_Methods": 6,
        }
        test_ratio=0.1

        G, adj, features, sensitive, test_edges_true, test_edges_false, _ = self.citeseer(test_ratio=test_ratio)


        print(adj.shape)
        print(features.shape)
        node_num = features.shape[0]


        features = torch.FloatTensor(features)
        sens = torch.FloatTensor(sensitive)

        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        adj=mx_to_torch_sparse_tensor(adj)
        self.adj_=adj
        self.features_=features
        self.labels_=None
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens

    def build_test(self, G: nx.Graph, nodelist: Dict, ratio: float) -> Tuple:
        """
        Split training and testing set for link prediction in graph G.
        :param G: nx.Graph
        :param nodelist: idx -> node_id in nx.Graph
        :param ratio: ratio of positive links that used for testing
        :return: Graph that remove all test edges, list of index for test edges
        """

        edges = list(G.edges.data(default=False))
        num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
        num_test = int(np.floor(num_edges * ratio))
        test_edges_true = []
        test_edges_false = []

        # generate false links for testing
        while len(test_edges_false) < num_test:
            idx_u = np.random.randint(0, num_nodes - 1)
            idx_v = np.random.randint(idx_u, num_nodes)

            if idx_u == idx_v:
                continue
            if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
                continue
            if (idx_u, idx_v) in test_edges_false:
                continue
            else:
                test_edges_false.append((idx_u, idx_v))

        # generate true links for testing
        all_edges_idx = list(range(num_edges))
        np.random.shuffle(all_edges_idx)
        test_edges_true_idx = all_edges_idx[:num_test]
        for test_idx in test_edges_true_idx:
            u, v, _ = edges[test_idx]
            G.remove_edge(u, v)
            test_edges_true.append((self.get_key(nodelist, u), self.get_key(nodelist, v)))

        return G, test_edges_true, test_edges_false

    def get_key(self, dict, value):
        return [k for k, v in dict.items() if v == value][0]

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index


    def citeseer(self,data_dir="./dataset/citeseer",  test_ratio=0.1):
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            with open(os.path.join(data_dir, "ind.citeseer.{}".format(names[i])), 'rb') as rf:
                u = pkl._Unpickler(rf)
                u.encoding = 'latin1'
                cur_data = u.load()
                objects.append(cur_data)

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        X = sp.vstack((allx, tx)).toarray()
        sensitive = sp.vstack((ally, ty))
        sensitive = np.where(sensitive.toarray() == 1)[1]

        G = nx.from_dict_of_lists(graph)
        test_idx_reorder = self.parse_index_file(os.path.join(data_dir, "ind.citeseer.test.index"))
        test_idx_range = np.sort(test_idx_reorder)

        missing_idx = set(range(min(test_idx_range), max(test_idx_range) + 1)) - set(test_idx_range)
        for idx in missing_idx:
            G.remove_node(idx)

        nodes = sorted(G.nodes())
        nodelist = {idx: node for idx, node in zip(range(G.number_of_nodes()), list(nodes))}

        G, test_edges_true, test_edges_false = self.build_test(G, nodelist, test_ratio)
        print(len(G.edges))
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        adj = nx.adjacency_matrix(G, nodelist=nodes)

        return G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist

class German(dataset):
    def __init__(self):
        super(German, self).__init__()
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test,sens_idx =self.load_german('german')

        node_num=features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features=self.feature_norm(features)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_=adj
        self.features_=features
        self.labels_=labels
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens
    
    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1
    
    def load_german(self, dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="./dataset/german/",
                    label_number=100):
        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')


        # Sensitive Attribute
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)

        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0

class Bail(dataset):
    def __init__(self):
        super(Bail, self).__init__()
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test,sens_idx = self.load_bail('bail')

        node_num=features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features=self.feature_norm(features)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_=adj
        self.features_=features
        self.labels_=labels
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens
    
    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1
    
    def load_bail(self, dataset, sens_attr="WHITE", predict_attr="RECID", path="./dataset/bail/", label_number=100):
        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)


        # build relationship

        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)
        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0


class Credit(dataset):
    def __init__(self):
        super(Credit, self).__init__()
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test,sens_idx = self.load_credit('credit')

        node_num=features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features=self.feature_norm(features)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
    
    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1
    
    def load_credit(self, dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/",
                    label_number=6000):
        from scipy.spatial import distance_matrix

        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('Single')


        # build relationship
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)

        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 1



class LCC(dataset):
    def __init__(self):
        super(LCC, self).__init__()
        path='./dataset/raw_LCC'
        name='LCC'


        edgelist=csv.reader(open(path+'/edgelist_{}.txt'.format(name)))

        edges=[]
        for line in edgelist:
            edge=line[0].split('\t')
            edges.append([int(one) for one in edge])


        edges=np.array(edges)
        print(len(edges))

        labels_file=csv.reader(open(path+'/labels_{}.txt'.format(name)))
        labels=[]
        for line in labels_file:
            labels.append(float(line[0].split('\t')[1]))
        labels=np.array(labels)

        sens_file=csv.reader(open(path+'/sens_{}.txt'.format(name)))
        sens=[]
        for line in sens_file:
            sens.append([float(line[0].split('\t')[1])])
        sens=np.array(sens)


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        node_num=labels.shape[0]
        #adj=adj.todense()
        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        sens = torch.FloatTensor(sens)
        features=np.load(path+'/X_{}.npz'.format(name))


        features=torch.FloatTensor(sp.coo_matrix((features['data'], (features['row'], features['col'])),
                    shape=(labels.shape[0], np.max(features['col'])+1),
                    dtype=np.float32).todense())
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_=adj
        self.features_=features
        self.labels_=labels
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens


class LCC_small(dataset):
    def __init__(self):
        super(LCC_small, self).__init__()
        path='./dataset/raw_small'
        name='Small'

        edgelist=csv.reader(open(path+'/edgelist_{}.txt'.format(name)))

        edges=[]
        for line in edgelist:
            edge=line[0].split('\t')
            edges.append([int(one) for one in edge])


        edges=np.array(edges)
        print(len(edges))

        labels_file=csv.reader(open(path+'/labels_{}.txt'.format(name)))
        labels=[]
        for line in labels_file:
            labels.append(float(line[0].split('\t')[1]))
        labels=np.array(labels)

        sens_file=csv.reader(open(path+'/sens_{}.txt'.format(name)))
        sens=[]
        for line in sens_file:
            sens.append([float(line[0].split('\t')[1])])
        sens=np.array(sens)


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        node_num=labels.shape[0]
        #adj=adj.todense()
        idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
        idx_val=list(set(list(range(node_num)))-set(idx_train))
        idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
        idx_val=list(set(idx_val)-set(idx_test))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        sens = torch.FloatTensor(sens)
        features=np.load(path+'/X_{}.npz'.format(name))


        features=torch.FloatTensor(sp.coo_matrix((features['data'], (features['row'], features['col'])),
                    shape=(labels.shape[0], np.max(features['col'])+1),
                    dtype=np.float32).todense())
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        adj=mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_=adj
        self.features_=features
        self.labels_=labels
        self.idx_train_=idx_train
        self.idx_val_=idx_val
        self.idx_test_=idx_test
        self.sens_=sens

class Amazon(dataset):
    def __init__(self):
        super(Amazon, self).__init__()
        dataname='Amazon-2'

        train_df = pkl.load(open('./dataset/' + dataname + '/training_df.pkl','rb'))
        vali_df = pkl.load(open('./dataset/' + dataname + '/valiing_df.pkl','rb'))  # for validation
        # vali_df = pkl.load(open('./' + dataname + '/testing_df.pkl'))  # for testing
        key_genre = pkl.load(open('./dataset/' + dataname + '/key_genre.pkl','rb'))
        item_idd_genre_list = pkl.load(open('./dataset/' + dataname + '/item_idd_genre_list.pkl','rb'))
        #genre_item_vector = pkl.load(open('./' + dataname + '/genre_item_vector.pkl','rb'))
        genre_count = pkl.load(open('./dataset/' + dataname + '/genre_count.pkl','rb'))
        #user_genre_count = pkl.load(open('./' + dataname + '/user_genre_count.pkl','rb'))

        num_item = len(train_df['item_id'].unique())
        num_user = len(train_df['user_id'].unique())
        num_genre = len(key_genre)

        item_genre_list = []
        for u in range(num_item):
            gl = item_idd_genre_list[u]
            tmp = []
            for g in gl:
                if g in key_genre:
                    tmp.append(g)
            item_genre_list.append(tmp)

        print(num_item)
        print(num_genre)
        print(len(item_genre_list))

        print('number of positive feedback: ' + str(len(train_df)))
        print('estimated number of training samples: ' + str(5* len(train_df)))


        # genreate item_genre matrix
        item_genre = np.zeros((num_item, num_genre))
        for i in range(num_item):
            gl = item_genre_list[i]
            for k in range(num_genre):
                if key_genre[k] in gl:
                    item_genre[i, k] = 1.0

        genre_count_mean_reciprocal = []
        for k in key_genre:
            genre_count_mean_reciprocal.append(1.0 / genre_count[k])
        genre_count_mean_reciprocal = (np.array(genre_count_mean_reciprocal)).reshape((num_genre, 1))
        genre_error_weight = np.dot(item_genre, genre_count_mean_reciprocal)

        self.item_genre_ = item_genre
        self.genre_error_weight_ = genre_error_weight
    
    def item_genre(self, datatype: str = 'np.array'):
        if self.item_genre_ is None:
            return self.item_genre_
        if datatype == 'np.array':
            return self.item_genre_
        elif datatype == 'torch.tensor':
            return torch.tensor(self.item_genre_)
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
        
    def genre_error_weight(self, datatype: str = 'np.array'):
        if self.genre_error_weight_ is None:
            return self.genre_error_weight_
        if datatype == 'np.array':
            return self.genre_error_weight_
        elif datatype == 'torch.tensor':
            return torch.tensor(self.genre_error_weight_)
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Yelp(dataset):
    def __init__(self):
        super(Yelp, self).__init__()
        dataname = Yelp

        train_df = pkl.load(open('./dataset/' + dataname + '/training_df.pkl','rb'))
        vali_df = pkl.load(open('./dataset/' + dataname + '/valiing_df.pkl','rb'))  # for validation
        # vali_df = pkl.load(open('./' + dataname + '/testing_df.pkl'))  # for testing
        key_genre = pkl.load(open('./dataset/' + dataname + '/key_genre.pkl','rb'))
        item_idd_genre_list = pkl.load(open('./dataset/' + dataname + '/item_idd_genre_list.pkl','rb'))
        #genre_item_vector = pkl.load(open('./' + dataname + '/genre_item_vector.pkl','rb'))
        genre_count = pkl.load(open('./dataset/' + dataname + '/genre_count.pkl','rb'))
        #user_genre_count = pkl.load(open('./' + dataname + '/user_genre_count.pkl','rb'))

        num_item = len(train_df['item_id'].unique())
        num_user = len(train_df['user_id'].unique())
        num_genre = len(key_genre)

        item_genre_list = []
        for u in range(num_item):
            gl = item_idd_genre_list[u]
            tmp = []
            for g in gl:
                if g in key_genre:
                    tmp.append(g)
            item_genre_list.append(tmp)

        print(num_item)
        print(num_genre)
        print(len(item_genre_list))

        print('number of positive feedback: ' + str(len(train_df)))
        print('estimated number of training samples: ' + str(5* len(train_df)))


        # genreate item_genre matrix
        item_genre = np.zeros((num_item, num_genre))
        for i in range(num_item):
            gl = item_genre_list[i]
            for k in range(num_genre):
                if key_genre[k] in gl:
                    item_genre[i, k] = 1.0

        genre_count_mean_reciprocal = []
        for k in key_genre:
            genre_count_mean_reciprocal.append(1.0 / genre_count[k])
        genre_count_mean_reciprocal = (np.array(genre_count_mean_reciprocal)).reshape((num_genre, 1))
        genre_error_weight = np.dot(item_genre, genre_count_mean_reciprocal)

        self.item_genre_ = item_genre
        self.genre_error_weight_ = genre_error_weight
    
    def item_genre(self, datatype: str = 'np.array'):
        if self.item_genre_ is None:
            return self.item_genre_
        if datatype == 'np.array':
            return self.item_genre_
        elif datatype == 'torch.tensor':
            return torch.tensor(self.item_genre_)
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
        
    def genre_error_weight(self, datatype: str = 'np.array'):
        if self.genre_error_weight_ is None:
            return self.genre_error_weight_
        if datatype == 'np.array':
            return self.genre_error_weight_
        elif datatype == 'torch.tensor':
            return torch.tensor(self.genre_error_weight_)
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


from scipy.io import loadmat
class data_handler():
    def __init__(self, rating_path, trust_path, dataset_name):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.n_users = 0
        self.n_prod = 0 
        self.n_cat = 6
        self.dataset_name = dataset_name

    def load_matrices(self):
        # Loading Matrices from data
        f1 = open(self.rating_path, 'rb')
        f2 = open(self.trust_path, 'rb')
        R = loadmat(f1)
        W = loadmat(f2)
        # Converting R and W from dictionary to array
        if self.dataset_name=='epinion':
            R = R['rating_with_timestamp']
        elif self.dataset_name=='ciao':
            R = R['rating']
        # print R
        R_old = np.asarray([row for row in R if row[0] < 1000 and row[1] < 1000])
        # R = R['rating']
        W = W['trust']
        # print W
        W_old = np.asarray([row for row in W if row[0] < 1000 and row[1] < 1000])
        # print W

        self.n_users = 1000
        self.n_prod = 1000

        self.n_users = max(R[:, 0])
        self.n_prod = max(R[:, 1])

        print(R.shape)
        print(self.n_users)
        print(self.n_prod)

        # print self.n_users
        # print self.n_prod
        # Selecting entries with the 6 categories given in the paper
        cat_id = [7, 8, 9, 10, 11, 19]
        cat_map = {7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 19: 5}
        # cat_id = [8, 14, 17, 19, 23, 24]
        # cat_map = {8:0, 14:1, 17:2, 19:3, 23:4, 24:5}
        R = R[np.in1d(R[:, 2], cat_id)]
        R = R[R[:, 5].argsort()]
        R_size = R.shape[0]
        # Choosing 70% data for training and rest for testing
        R_train = R[: R_size*7//10]
        R_test = R[R_size*7//10:]
        # Making all eligible Product-Category pairs
        #print (R_train.shape[0])
        ones = np.ones(R_train.shape[0])
        prod_cat = dict(zip(zip(R_train[:, 1], R_train[:, 2]), ones))
        prod_cat_old = {}

        # print prod_cat
        # Making the mu matrix
        mu = np.zeros(6)
        for cat in cat_id:
            cat_rating = R_train[np.where(R_train[:, 2] == cat), 3]
            mu[cat_map[cat]] = np.mean(cat_rating)


        print(W.shape)

        return R_train, R_test, W, prod_cat, mu

    def get_stats(self):
        return self.n_users, self.n_prod, self.n_cat


class Epinion(dataset):
    def __init__(self):
        super(Epinion, self).__init__()

        data = data_handler("./dataset/Epinion&Ciao/rating_with_timestamp.mat", "./dataset/Epinion&Ciao/trust.mat", 'epinion')

        R_train, R_test, W, PF_pair, mu = data.load_matrices()
        self.R_train_ = R_train
        self.R_test_ = R_test
        self.W_ = W.astype('uint8') 
        self.PF_pair_ = PF_pair
        self.mu_ = mu

        self.R_train_ = torch.tensor(self.R_train_)
        self.R_test_ = torch.tensor(self.R_test_)
        self.W_ = torch.tensor(self.W_)
        self.PF_pair_ = self.PF_pair_
        self.mu_ = torch.tensor(self.mu_)
    
    def R_train(self, datatype: str = 'torch.tensor'):
        if self.R_train_ is None:
            return self.R_train_
        if datatype == 'torch.tensor':
            return self.R_train_
        elif datatype == 'np.array':
            return self.R_train_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def R_test(self, datatype: str = 'torch.tensor'):
        if self.R_test_ is None:
            return self.R_test_
        if datatype == 'torch.tensor':
            return self.R_test_
        elif datatype == 'np.array':
            return self.R_test_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

    def W(self, datatype: str = 'torch.tensor'):
        if self.W_ is None:
            return self.W_
        if datatype == 'torch.tensor':
            return self.W_
        elif datatype == 'np.array':
            return self.W_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def PF_pair(self, datatype: str = 'dict'):
        if self.PF_pair_ is None:
            return self.PF_pair_
        if datatype == 'dict':
            return self.PF_pair_
        else:
            raise ValueError('datatype should be dict')
    
    def mu(self, datatype: str = 'torch.tensor'):
        if self.mu_ is None:
            return self.mu_
        if datatype == 'torch.tensor':
            return self.mu_
        elif datatype == 'np.array':
            return self.mu_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


class Ciao(dataset):
    def __init__(self):
        super(Ciao, self).__init__()

        data = data_handler("./dataset/Epinion&Ciao/ciao/rating_with_timestamp.mat", "./dataset/Epinion&Ciao/ciao/trust.mat", "ciao")

        R_train, R_test, W, PF_pair, mu = data.load_matrices()
        self.R_train_ = R_train
        self.R_test_ = R_test
        self.W_ = W.astype('uint8') 
        self.PF_pair_ = PF_pair
        self.mu_ = mu

        self.R_train_ = torch.tensor(self.R_train_)
        self.R_test_ = torch.tensor(self.R_test_)
        self.W_ = torch.tensor(self.W_)
        self.PF_pair_ = self.PF_pair_
        self.mu_ = torch.tensor(self.mu_)
    
    def R_train(self, datatype: str = 'torch.tensor'):
        if self.R_train_ is None:
            return self.R_train_
        if datatype == 'torch.tensor':
            return self.R_train_
        elif datatype == 'np.array':
            return self.R_train_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def R_test(self, datatype: str = 'torch.tensor'):
        if self.R_test_ is None:
            return self.R_test_
        if datatype == 'torch.tensor':
            return self.R_test_
        elif datatype == 'np.array':
            return self.R_test_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

    def W(self, datatype: str = 'torch.tensor'):
        if self.W_ is None:
            return self.W_
        if datatype == 'torch.tensor':
            return self.W_
        elif datatype == 'np.array':
            return self.W_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def PF_pair(self, datatype: str = 'dict'):
        if self.PF_pair_ is None:
            return self.PF_pair_
        if datatype == 'dict':
            return self.PF_pair_
        else:
            raise ValueError('datatype should be dict')
    
    def mu(self, datatype: str = 'torch.tensor'):
        if self.mu_ is None:
            return self.mu_
        if datatype == 'torch.tensor':
            return self.mu_
        elif datatype == 'np.array':
            return self.mu_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


class Dblp(dataset):
    def __init__(self):
        super(Dblp, self).__init__()
        dataset_path = './dataset/dblp/'

        with open(
                join(dataset_path, "author-author.csv"), mode="r", encoding="ISO-8859-1"
        ) as file_name:
            edges = np.genfromtxt(file_name, delimiter=",", dtype=int)

        with open(
                join(dataset_path, "countries.csv"), mode="r", encoding="ISO-8859-1"
        ) as file_name:
            attributes = np.genfromtxt(file_name, delimiter=",", dtype=str)

        sensitive = self.encode_classes(attributes[:, 1])
        num_classes = len(np.unique(sensitive))
        N = sensitive.shape[0]
        m = np.random.choice(len(edges), int(len(edges) * 0.8), replace=False)
        tr_mask = np.zeros(len(edges), dtype=bool)
        tr_mask[m] = True
        pos_edges_tr = edges[tr_mask]
        pos_edges_te = edges[~tr_mask]

        pos_edges_te = torch.LongTensor(pos_edges_te.T)
        #neg_edges_te = negative_sampling(
        #    edge_index=pos_edges_te, num_nodes=N, num_neg_samples=pos_edges_te.size(1)
        #)

        pos_edges_tr = torch.LongTensor(pos_edges_tr.T)
        #neg_edges_tr = negative_sampling(
        #    edge_index=pos_edges_tr, num_nodes=N, num_neg_samples=pos_edges_tr.size(1)
        #)

        self.pos_edges_tr_ = pos_edges_tr
        self.pos_edges_te_ = pos_edges_te

        self.pos_edges_te_ = torch.LongTensor(pos_edges_te)
        self.pos_edges_tr_ = torch.LongTensor(pos_edges_tr)

    def encode_classes(self, col):
        """
        Input:  categorical vector of any type
        Output: categorical vector of int in range 0-num_classes
        """
        classes = set(col)
        classes_dict = {c: i for i, c in enumerate(classes)}
        labels = np.array(list(map(classes_dict.get, col)), dtype=np.int32)
        return labels
    
    def pos_edges_tr(self, datatype: str = 'torch.tensor'):
        if self.pos_edges_tr_ is None:
            return self.pos_edges_tr_
        if datatype == 'torch.tensor':
            return self.pos_edges_tr_
        elif datatype == 'np.array':
            return self.pos_edges_tr_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def pos_edges_te(self, datatype: str = 'torch.tensor'):
        if self.pos_edges_te_ is None:
            return self.pos_edges_te_
        if datatype == 'torch.tensor':
            return self.pos_edges_te_
        elif datatype == 'np.array':
            return self.pos_edges_te_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


class Filmtrust(dataset):
    def __init__(self):
        super(Filmtrust, self).__init__()
        trust_file = open('./dataset/filmtrust/trust.txt')
        user_id=[]
        for line in trust_file.readlines():
            rating = line.strip().split(' ')
            user_id.append(int(rating[0]))
            user_id.append(int(rating[1]))
        user_num=max(user_id)


        ratings_file=open('./dataset/filmtrust/ratings.txt')
        user_id=[]
        item_id=[]
        rating_value=[]
        for line in ratings_file.readlines():
            rating=line.strip().split(' ')
            user_id.append(int(rating[0]))
            item_id.append(int(rating[1]))
            rating_value.append(float(rating[2]))

        print(max(user_id))
        print(max(item_id))
        print(len(rating_value))

        rating_matrix = np.zeros([user_num,max(item_id)])

        for uid, iid, value in zip(user_id,item_id,rating_value):
            rating_matrix[uid-1,iid-1]=value



        trust_file = open('./dataset/filmtrust/trust.txt')
        trust_matrix=np.zeros([user_num,user_num])
        for line in trust_file.readlines():
            rating=line.strip().split(' ')
            uid=int(rating[0])
            uid_=int(rating[1])
            value=float(rating[2])
            trust_matrix[uid - 1, uid_ - 1] = value

        self.rating_matrix_=rating_matrix
        self.trust_matrix_=trust_matrix

        self.rating_matrix_=torch.tensor(rating_matrix)
        self.trust_matrix_=torch.tensor(trust_matrix)
    
    def rating_matrix(self, datatype: str = 'torch.tensor'):
        if self.rating_matrix_ is None:
            return self.rating_matrix_
        if datatype == 'torch.tensor':
            return self.rating_matrix_
        elif datatype == 'np.array':
            return self.rating_matrix_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def trust_matrix(self, datatype: str = 'torch.tensor'):
        if self.trust_matrix_ is None:
            return self.trust_matrix_
        if datatype == 'torch.tensor':
            return self.trust_matrix_
        elif datatype == 'np.array':
            return self.trust_matrix_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Lastfm(dataset):
    def __init__(self):
        super(Lastfm, self).__init__()
        V=np.loadtxt('./dataset/lastfm/LF.csv',delimiter=',')
        print("relevance scoring data loaded")

        m=V.shape[0] # number of customers
        n=V.shape[1] # number of producers

        print(m)
        print(n)
        print(V)

        print((V!=0).sum())

        U=range(m) # list of customers
        P=range(n) # list of producers

        self.V_ = V
        self.U_ = U

        self.V_ = torch.tensor(self.V_)
        self.U_ = torch.tensor(self.U_)
    
    def V(self, datatype: str = 'torch.tensor'):
        if self.V_ is None:
            return self.V_
        if datatype == 'torch.tensor':
            return self.V_
        elif datatype == 'np.array':
            return self.V_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def U(self, datatype: str = 'torch.tensor'):
        if self.U_ is None:
            return self.U_
        if datatype == 'torch.tensor':
            return self.U_
        elif datatype == 'np.array':
            return self.U_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Ml_1m(dataset):
    def __init__(self):
        super(Lm, self).__init__()
        dataset_name='ml-1m'
        user_num=6040
        item_num=3952
        data=open('./dataset/{}/ratings.dat'.format(dataset_name))
        rating_matrix=np.zeros([user_num,item_num])

        print(len(data.readlines()))

        for line in data.readlines():
            rating=line.strip().split('::')
            rating_matrix[int(rating[0])-1,int(rating[1])-1]=float(rating[2])


        user_info=open('./dataset/{}/users.dat'.format(dataset_name))
        user_feat=[]
        for line in user_info:
            infor=line.strip().split('::')
            print(infor)
            user_feat.append(infor[1:])
        print(user_feat)
        user_sens=[0 if one[0]=='F' else 1 for one in user_feat]

        self.rating_matrix_=rating_matrix
        self.user_sens_=user_sens

        self.rating_matrix_=torch.tensor(self.rating_matrix_)
        self.user_sens_=torch.tensor(self.user_sens_)
    
    def rating_matrix(self, datatype: str = 'torch.tensor'):
        if self.rating_matrix_ is None:
            return self.rating_matrix_
        if datatype == 'torch.tensor':
            return self.rating_matrix_
        elif datatype == 'np.array':
            return self.rating_matrix_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
        
    def user_sens(self, datatype: str = 'torch.tensor'):
        if self.user_sens_ is None:
            return self.user_sens_
        if datatype == 'torch.tensor':
            return self.user_sens_
        elif datatype == 'np.array':
            return self.user_sens_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Ml_100k(dataset):
    def __init__(self):
        super(Ml1m, self).__init__()
        dataset_name='ml-100k'
        data=open('./dataset/{}/u.data'.format(dataset_name))
        user_num=943
        item_num=1682

        rating_matrix=np.zeros([user_num,item_num])

        print(len(data.readlines()))

        for line in data.readlines():
            rating=line.strip().split('\t')
            rating_matrix[int(rating[0])-1,int(rating[1])-1]=float(rating[2])

        user_info=open('./dataset/{}/u.user'.format(dataset_name))
        user_feat=[]
        for line in user_info:
            infor=line.strip().split('|')
            #print(infor)
            user_feat.append(infor[1:])

        print(user_feat)

        user_sens=[0 if one[1]=='F' else 1 for one in user_feat]

        self.rating_matrix_=rating_matrix
        self.user_sens_=user_sens
        self.rating_matrix = torch.tensor(self.rating_matrix_)
        self.user_sens = torch.tensor(self.user_sens_)

    def rating_matrix(self, datatype: str = 'torch.tensor'):
        if self.rating_matrix_ is None:
            return self.rating_matrix_
        if datatype == 'torch.tensor':
            return self.rating_matrix_
        elif datatype == 'np.array':
            return self.rating_matrix_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def user_sens(self, datatype: str = 'torch.tensor'):
        if self.user_sens_ is None:
            return self.user_sens_
        if datatype == 'torch.tensor':
            return self.user_sens_
        elif datatype == 'np.array':
            return self.user_sens_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Ml_20m(dataset):
    def __init__(self):
        super(Ml1m, self).__init__()
        dataset_name='ml-20m'
        user_num=138493
        item_num=27278
        data = pd.read_csv('./dataset/{}/ratings.csv'.format(dataset_name))

        movies = pd.read_csv('./dataset/{}/movies.csv'.format(dataset_name))
        print(movies)
        print(data.shape[0])
        movieid2id=dict()
        for i in range(movies.shape[0]):
            movieid2id[int(movies.iloc[i,0])]=i

        rating_matrix=np.zeros([user_num,item_num],dtype=np.short)

        for i in range(data.shape[0]):
            rating_matrix[int(data.iloc[i,0])-1,movieid2id[int(data.iloc[i,1])]]=int(data.iloc[i,2]*10)

        np.save('./dataset/{}/rating_matrix.npy'.format(dataset_name),rating_matrix)
        self.rating_matrix_=rating_matrix
        self.rating_matrix_ = torch.tensor(self.rating_matrix_)
    
    def rating_matrix(self, datatype: str = 'torch.tensor'):
        if self.rating_matrix_ is None:
            return self.rating_matrix_
        if datatype == 'torch.tensor':
            return self.rating_matrix_
        elif datatype == 'np.array':
            return self.rating_matrix_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')

class Oklahoma(dataset):
    def __init__(self):
        super(Oklahoma, self).__init__()
        dataset_name='oklahoma'
        from scipy.io import loadmat
        if dataset_name=='oklahoma':
            dataset_name='Oklahoma97'
        elif dataset_name=='unc28':
            dataset_name='UNC28'


        feats=pkl.load(open('./dataset/oklahoma&unc/{}/{}_feat.pkl'.format(dataset_name,dataset_name), 'rb'))
        sens=pkl.load(open('./dataset/oklahoma&unc/{}/{}_user_sen.pkl'.format(dataset_name,dataset_name), 'rb'))
        sens=[sens[idx] for idx in range(feats.shape[0])]
        train_items=pkl.load(open('./dataset/oklahoma&unc/UNC28/UNC28_train_items.pkl', 'rb'))
        test_items=pkl.load(open('./dataset/oklahoma&unc/UNC28/UNC28_test_set.pkl', 'rb'))

        adj=np.zeros([feats.shape[0], feats.shape[0]])

        for item in [train_items, test_items]:
            for key, value in item.items():
                for one in value:
                    adj[key][one]=1
        
        self.adj_=mx_to_torch_sparse_tensor(adj)
        features = torch.FloatTensor(feats)
        sens = torch.FloatTensor(sens)
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        train_items = torch.LongTensor(train_items)
        test_items = torch.LongTensor(test_items)

        self.features_=features
        self.train_items_=train_items
        self.test_items_=test_items
        self.sens_=sens
    
    def adj(self, datatype: str = 'torch.sparse'):
        if self.adj_ is None:
            return self.adj_
        if datatype == 'torch.sparse':
            return self.adj_
        elif datatype == 'np.array':
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError('datatype should be torch.sparse or np.array')
        
    def features(self, datatype: str = 'torch.tensor'):
        if self.features_ is None:
            return self.features_
        if datatype == 'torch.tensor':
            return self.features_
        elif datatype == 'np.array':
            return self.features_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def train_items(self, datatype: str = 'torch.tensor'):
        if self.train_items_ is None:
            return self.train_items_
        if datatype == 'torch.tensor':
            return self.train_items_
        elif datatype == 'np.array':
            return self.train_items_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def test_items(self, datatype: str = 'torch.tensor'):
        if self.test_items_ is None:
            return self.test_items_
        if datatype == 'torch.tensor':
            return self.test_items_
        elif datatype == 'np.array':
            return self.test_items_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def sens(self, datatype: str = 'torch.tensor'):
        if self.sens_ is None:
            return self.sens_
        if datatype == 'torch.tensor':
            return self.sens_
        elif datatype == 'np.array':
            return self.sens_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


class UNC(dataset):
    def __init__(self):
        super(UNC, self).__init__()
        from scipy.io import loadmat
        dataset_name='unc28'
        if dataset_name=='oklahoma':
            dataset_name='Oklahoma97'
        elif dataset_name=='unc28':
            dataset_name='UNC28'


        feats=pkl.load(open('./dataset/oklahoma&unc/{}/{}_feat.pkl'.format(dataset_name,dataset_name), 'rb'))
        sens=pkl.load(open('./dataset/oklahoma&unc/{}/{}_user_sen.pkl'.format(dataset_name,dataset_name), 'rb'))
        sens=[sens[idx] for idx in range(feats.shape[0])]
        train_items=pkl.load(open('./dataset/oklahoma&unc/UNC28/UNC28_train_items.pkl', 'rb'))
        test_items=pkl.load(open('./dataset/oklahoma&unc/UNC28/UNC28_test_set.pkl', 'rb'))

        adj=np.zeros([feats.shape[0], feats.shape[0]])

        for item in [train_items, test_items]:
            for key, value in item.items():
                for one in value:
                    adj[key][one]=1
        
        self.adj_=mx_to_torch_sparse_tensor(adj)
        features = torch.FloatTensor(feats)
        sens = torch.FloatTensor(sens)
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        train_items = torch.LongTensor(train_items)
        test_items = torch.LongTensor(test_items)

        self.features_=features
        self.train_items_=train_items
        self.test_items_=test_items
        self.sens_=sens
    
    def adj(self, datatype: str = 'torch.sparse'):
        if self.adj_ is None:
            return self.adj_
        if datatype == 'torch.sparse':
            return self.adj_
        elif datatype == 'np.array':
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError('datatype should be torch.sparse or np.array')
        
    def features(self, datatype: str = 'torch.tensor'):
        if self.features_ is None:
            return self.features_
        if datatype == 'torch.tensor':
            return self.features_
        elif datatype == 'np.array':
            return self.features_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def train_items(self, datatype: str = 'torch.tensor'):
        if self.train_items_ is None:
            return self.train_items_
        if datatype == 'torch.tensor':
            return self.train_items_
        elif datatype == 'np.array':
            return self.train_items_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def test_items(self, datatype: str = 'torch.tensor'):
        if self.test_items_ is None:
            return self.test_items_
        if datatype == 'torch.tensor':
            return self.test_items_
        elif datatype == 'np.array':
            return self.test_items_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')
    
    def sens(self, datatype: str = 'torch.tensor'):
        if self.sens_ is None:
            return self.sens_
        if datatype == 'torch.tensor':
            return self.sens_
        elif datatype == 'np.array':
            return self.sens_.numpy()
        else:
            raise ValueError('datatype should be torch.tensor or np.array')


google = Google()
facebook = Facebook()
oklahoma = Oklahoma()
unc = UNC()
twitter = Twitter()
lastfm = Lastfm()
nba = Nba()
ml_1m = Ml_1m()
ml_20m = Ml_20m()
ml_100k = Ml_100k()
german = German()
bail = Bail()
credit = Credit()
cora = Cora()
pokec_n = Pokec_n()
pokec_z = Pokec_z()
filmtrust = Filmtrust()
citeseer = Citeseer()
yelp = Yelp()
amazon = Amazon()
