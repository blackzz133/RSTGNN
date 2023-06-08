import torch_geometric.data.data

from dataloader import DBLPLoader,RedditLoader, TwitterTennisDatasetLoader,EnglandCovidDatasetLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import argparse
import copy as cp
from Node_Classification import GCN_selection_Node_Classification
from Link_Prediction import GCN_selection_Link_Prediction
from Start.Attack.DBLP.attack import perturbation, perturbation2
import torch

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
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='64,32',
                        help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='8',
                        help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')



def train(args, GCN_type, data_type, attack_type, attack_method,defend_method, perturbation_rate):
    print('start')
    args = parser.parse_args()
    #loader = TwitterTennisDatasetLoader()
    loader = DBLPLoader(data_type)
    dataset = loader.get_dataset()


    train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    #train_loader, test_loader = temporal_signal_split(train_loader, 0.7)
    #_, test_loader = temporal_signal_split(train_loader, 2 / 3)
    origin_train_loader = cp.deepcopy(train_loader)
    train_loader = perturbation(train_loader, attack_method, perturbation_rate)

    print('the raw and perturbed shape of edges in snapshot 0')
    for time, snapshot in enumerate(origin_train_loader):
        print(snapshot.edge_index.shape)
        break
    for time, snapshot in enumerate(train_loader):
        print(snapshot.edge_index.shape)
        break

    if attack_type == 'node':
        model = GCN_selection_Node_Classification(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method)
    elif attack_type =='link':
        model = GCN_selection_Link_Prediction(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method)

def data_preprossing(dataset):
    graphs = []
    for time, snapshot in enumerate(dataset):
        graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y)
        graphs.append(graph)
    return graphs





method = 'original'
GCN_type = 'Robust_RGNN' #GCN2,GCN3,DCRNN,EVOLVEGCNO,TGCN,A3TGCN,GCN2_p,GCN2_attention,GCN2_GAT,GAT #GAT:node classification, GAT2_GRU:Link_prediction
data_type ='reddit' #DBLP3,DBLP5,reddit, Brain
attack_type = 'link'# node, link
attack_method = 'node_embedding'#random_node, random_link,edge_attack #node_embedding
#attack_method =
defend_method = ''
perturbation_rate = 0.1
args = parser.parse_args()
train(args, GCN_type, data_type, attack_type, attack_method,defend_method,perturbation_rate)
