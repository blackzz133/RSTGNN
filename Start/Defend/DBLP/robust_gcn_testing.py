import torch_geometric.data.data
import torch
from dataloader import DBLPLoader,RedditLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import argparse
import copy as cp
from Node_Classification import GCN_selection_Node_Classification
from Link_Prediction import GCN_selection_Link_Prediction
from Start.Attack.DBLP.attack import perturbation
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj
from deeprobust.graph.defense import RGCN
from scipy import sparse
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox,Metattack

import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")

parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name')
parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    # Number of negative samples per positive pair.
parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='64,64',
                        help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')



def train(args, GCN_type, data_type, attack_type, attack_method,defend_method):
    print('start')
    args = parser.parse_args()
    loader = DBLPLoader(data_type)
    dataset = loader.get_dataset()
    train_loader, test_loader = temporal_signal_split(dataset, 0.7)
    origin_train_loader = cp.deepcopy(train_loader)
    train_loader = perturbation(train_loader)
    print('the raw and perturbed shape of edges in snapshot 0')
    for time, snapshot in enumerate(origin_train_loader):
        print(snapshot.edge_index.shape)
        break
    for time, snapshot in enumerate(train_loader):
        print(snapshot.edge_index.shape)
        break
    #adj, features, labels =
    data_preprossing2(train_loader, test_loader)


    '''
    if attack_type == 'node':
        model = GCN_selection_Node_Classification(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method)
    elif attack_type =='link':
        model = GCN_selection_Link_Prediction(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method)
    '''

def data_preprossing(train_dataset,test_dataset):
    graphs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_num = torch.tensor(train_dataset.features).shape[1]
    feature_num = torch.tensor(train_dataset.features).shape[2]
    class_num = torch.tensor(train_dataset.features).shape[2]
    model = RGCN(nnodes=node_num, nfeat=feature_num, nclass=class_num,
                 nhid=32, device=device)
    model = model.to(device)
    for time, snapshot in enumerate(train_dataset):
        num_nodes = snapshot.x.shape[0]
        adj = to_scipy_sparse_matrix(edge_index=snapshot.edge_index,edge_attr=snapshot.edge_attr,num_nodes=num_nodes).tocsr()
        features = sparse.csr_matrix(snapshot.x.numpy())
        labels = torch.argmax(snapshot.y, dim=1).numpy()
        idx = np.random.permutation(np.arange(num_nodes)).tolist()
        train_ratio = 0.7
        val_ratio = 0.1
        idx_train = idx[:round(num_nodes*train_ratio)]
        idx_val = idx[round(num_nodes*train_ratio):round(num_nodes*(train_ratio+val_ratio))]
        idx_test = idx[round(num_nodes*(train_ratio+val_ratio)):]

        model.fit(features, adj, labels, idx_train, idx_val, train_iters=30, verbose=True)
        # You can use the inner function of model to test
        model.test(idx_train)
        #model.test(idx_test)

def data_preprossing2(train_dataset,test_dataset):
    graphs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_num = torch.tensor(train_dataset.features).shape[1]
    feature_num = torch.tensor(train_dataset.features).shape[2]
    class_num = torch.tensor(train_dataset.features).shape[2]

    for time, snapshot in enumerate(train_dataset):
        num_nodes = snapshot.x.shape[0]
        adj = torch.squeeze(to_dense_adj(edge_index=snapshot.edge_index,edge_attr=snapshot.edge_attr,max_num_nodes=num_nodes))
        features = snapshot.x
        labels = torch.argmax(snapshot.y, dim=1)
        idx = np.random.permutation(np.arange(num_nodes))
        train_ratio = 0.7
        val_ratio = 0.1
        idx_train = idx[:round(num_nodes*train_ratio)]
        idx_val = idx[round(num_nodes*train_ratio):round(num_nodes*(train_ratio+val_ratio))]
        idx_test = idx[round(num_nodes*(train_ratio+val_ratio)):]
        idx_unlabeled = np.union1d(idx_val, idx_test)
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        # Setup Attack Model

        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                           attack_structure=True, attack_features=True, device=device, lambda_=0).to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=100, ll_constraint=True)
        print('111')

        #model.test(idx_test)



method = 'original'
GCN_type = 'Robust_RGNN' #GCN2,GCN3,DCRNN,EVOLVEGCNO,TGCN,A3TGCN,GCN2_p,GCN2_attention,GCN2_GAT,GAT #GAT:node classification, GAT2_GRU:Link_prediction
data_type ='DBLP5' #DBLP3,DBLP5,reddit, Brain
attack_type = 'link'# node, link
attack_method = ''
defend_method = ''
args = parser.parse_args()
train(args, GCN_type, data_type, attack_type, attack_method,defend_method)


