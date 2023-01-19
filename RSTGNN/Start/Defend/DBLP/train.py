# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Start.Defend.DBLP import *
import Link_Prediction
import Node_Classification
import torch
from dataloader import DBLPLoader,RedditLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import argparse
import copy as cp
from Node_Classification import GCN_selection_Node_Classification
from Link_Prediction import GCN_selection_Link_Prediction
from Start.Attack.DBLP.attack import perturbation
parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="chickenpox",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=30,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="first_query",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-q","--queries", type=int, default=10,
                    help="Number of queries in a circle")
parser.add_argument("-b","--budget", type=int, default=20 ,
                    help="queried node budget at a snapshot")
parser.add_argument("-p","--span", type=int, default=2,
                    help="time span of a subquery")
parser.add_argument("-k","--sampling_rate", type=float, default=0.5,
                    help="time span of a subquery")
parser.add_argument("-cn","--class_num", type=int, default=5,
                    help="the number of classes")
parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help="learning rate")



def train(args, GCN_type, data_type, attack_type, attack_method,defend_method,perturbation_rate):
    print('start')
    args = parser.parse_args()
    loader = DBLPLoader(data_type)
    dataset = loader.get_dataset()
    train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    #test_loader, _ = temporal_signal_split(test_loader,1/3)
    #print(torch.tensor(train_loader.features).shape)
    #print(torch.tensor(test_loader.features).shape)
    #print(max([torch.tensor(train_loader.edge_indices[t]).shape[1] for t in range(len(train_loader.edge_indices))]))
    #print(min([torch.tensor(train_loader.edge_indices[t]).shape[1] for t in range(len(train_loader.edge_indices))]))
    #print(torch.tensor(train_loader.targets).shape)
    #exit()
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




#method = 'Evolve_GAT'
GCN_type = 'GCLSTM' #GCN2,GCN3,GCLSTM,DCRNN,EVOLVEGCNO,TGCN,A3TGCN,GCN2_p,GCN2_attention,GCN2_GAT,Evolve_GAT,,GAT #GAT:node classification, GAT2_GRU:Link_prediction
data_type ='reddit' #DBLP3,DBLP5,reddit, Brain
attack_type = 'link'# node, link
attack_method = 'edge_attack'#random_node, random_link ,edge_attack
defend_method = ''
perturbation_rate = 0.1
args = parser.parse_args()
train(args, GCN_type, data_type, attack_type, attack_method,defend_method,perturbation_rate)

