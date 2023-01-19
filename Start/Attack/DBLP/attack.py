import torch
from torch_geometric.utils import  from_scipy_sparse_matrix, to_dense_adj
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from deeprobust.graph.global_attack import NodeEmbeddingAttack,Random
import numpy as np
from torch_geometric.utils import to_dense_adj
from deeprobust.graph.defense import RGCN,GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.defense import GCN
import torch_geometric
from scipy import sparse
from torch_geometric.utils.convert import to_networkx
import networkx as nx
#from torch_geometric.utils.convert import to_scipy_sparse_matrix
import random as rd
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE

def perturbation(train_loader, attack_type, perturbation_rate): #perturbation by random


    if attack_type== 'random_node':
        train_loader= random_node_perturbation(train_loader, perturbation_rate)
    elif attack_type== 'random_link':
        train_loader = random_link_perturbation(train_loader, perturbation_rate)
    elif attack_type == 'edge_attack':
        train_loader = edge_attack_perturbation(train_loader, perturbation_rate)
    elif attack_type == 'meta':
        train_loader = random_link_perturbation(train_loader, perturbation_rate)

    # model = Link_Prediction.train('GAT2_GRU',train_loader,test_loader,origin_train_loader)

    return train_loader

def random_link_perturbation(train_loader, perturbation_rate):
    rand = Random()
    edge_indices = []
    edge_weights = []
    #features = []
    for time, snapshot in enumerate(train_loader):
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        adj = to_scipy_sparse_matrix(edge_index, edge_attr).tocsr()
        n_perturbations = round(snapshot.x.shape[0]*perturbation_rate)
        rand.attack(adj, type="flip", n_perturbations=n_perturbations)
        # rand.attack(adj, attack_type="remove", min_span_tree=True,n_perturbations=100)
        modified_adj = rand.modified_adj
        # w = (adj!=new_a).nnz==0
        # rand.attack(adj, attack_type="add", n_candidates=10000)
        # rand.attack(adj, attack_type="add_by_remove", n_candidates=10000)
        new_edge_index, new_edge_attr = from_scipy_sparse_matrix(modified_adj)
        #edge_indices.append(new_edge_index.tolist())
        #edge_weights.append(new_edge_attr.tolist())
        edge_indices.append(new_edge_index.tolist())
        edge_weights.append(new_edge_attr.tolist())
    train_loader.edge_indices = edge_indices
    train_loader.edge_weights = edge_weights
    return train_loader

def random_node_perturbation(train_loader, perturbation_rate):
    #edge_indices = []
    #edge_weights = []
    features = []
    for time, snapshot in enumerate(train_loader):
        std = torch.std(snapshot.x,dim=0)
        mean = torch.mean(snapshot.x,dim=0)
        n_perturbations = round(snapshot.x.shape[0] * perturbation_rate)
        attack_nodes = rd.sample(np.arange(0,snapshot.x.shape[0]).tolist(), n_perturbations)
        for i in attack_nodes:
            noise = torch.normal(mean=mean, std=std)
            snapshot.x[i] = noise
        features.append(snapshot.x.tolist())
    train_loader.features = features

    return train_loader

def edge_attack_perturbation(train_loader, perturbation_rate):
    edge_indices = []
    edge_weights = []

    for time, snapshot in enumerate(train_loader):
        adj = to_scipy_sparse_matrix(snapshot.edge_index, snapshot.edge_attr).tocsr()
        features = snapshot.x
        labels = torch.argmax(snapshot.y, dim=1)
        n_perturbations = round(snapshot.x.shape[0] * perturbation_rate)
        model =DICE()
        model.attack(adj, labels, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
        # w = (adj!=new_a).nnz==0
        # rand.attack(adj, attack_type="add", n_candidates=10000)
        # rand.attack(adj, attack_type="add_by_remove", n_candidates=10000)
        new_edge_index, new_edge_attr = from_scipy_sparse_matrix(modified_adj)
        # edge_indices.append(new_edge_index.tolist())
        # edge_weights.append(new_edge_attr.tolist())
        edge_indices.append(new_edge_index.tolist())
        edge_weights.append(new_edge_attr.tolist())
    train_loader.edge_indices = edge_indices
    train_loader.edge_weights = edge_weights
    return train_loader

def perturbation2(train_loader, attach_type, name_data): #perturbation by strategy
    all_features = []
    edge_indices = []
    edge_weights = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perturbations = 100
    #<class 'scipy.sparse.csr.csr_matrix'> features,adj

    for time, snapshot in enumerate(train_loader):
        num_nodes = snapshot.x.shape[0]

        #adj = to_scipy_sparse_matrix(snapshot.edge_index, snapshot.edge_attr,num_nodes=snapshot.x.shape[0]).tocsr()
        #features = sparse.csr_matrix(snapshot.x)
        adj = to_dense_adj(edge_index=snapshot.edge_index,edge_attr=snapshot.edge_attr,max_num_nodes=snapshot.x.shape[0]).squeeze()
        features = snapshot.x
        labels = torch.argmax(snapshot.y, dim=1)
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
        idx = np.random.permutation(np.arange(num_nodes))
        train_ratio = 0.7
        val_ratio = 0.1
        idx_train = idx[:round(num_nodes * train_ratio)]
        idx_val = idx[round(num_nodes * train_ratio):round(num_nodes * (train_ratio + val_ratio))]
        idx_test = idx[round(num_nodes * (train_ratio + val_ratio)):]
        idx_unlabeled = np.union1d(idx_val, idx_test)
        surrogate = surrogate.to(device)
        surrogate.fit(features, adj, labels, idx_train)
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                           attack_features=False, device=device, lambda_=0.5)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        all_features.append(model.modified_features.toarray())
        #edge_indic

    train_loader.features = all_features
    train_loader.edge_indices = edge_indices
    train_loader.edge_weights = edge_weights
