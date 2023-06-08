from Models.GNN.models import GCN2, GCN3
from Models.GNN.models_node import GCN2_attention,GCN2_GAT,GAT,New_GAT_LSTM
from Models.RGCN.models import DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
from dataloader import DBLPLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np
import torch
from deeprobust.graph.global_attack import Metattack, MetaApprox
from torch_geometric.utils import to_dense_adj
def GCN_selection(GCN_type, train_loader, test_loader, origin_train_loader):
    features = torch.tensor(train_loader.features).shape[2]
    num_classes = torch.tensor(train_loader.targets).shape[2]
    print(torch.tensor(train_loader.features).shape)
    print(torch.tensor(train_loader.targets).shape)
    if GCN_type == 'GAT':
        model = GAT(node_features=features, second_features=32, num_classes=num_classes)
    elif GCN_type=='GCN2':
        model = GCN2(node_features=features, second_features=32, num_classes=num_classes)
    else:
        return 0
    if torch.cuda.is_available():
        model = model.cuda()
    #train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    # print(train_loader.features)
    # print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    model.train()
    for epoch in tqdm(range(1000)):
        cost = 0
        hidden1 = None
        hidden2 = None
        for time, snapshot in enumerate(train_loader):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().cuda()
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.cuda()
            edge_weight = snapshot.edge_attr.cuda()
            y_hat,hidden1,hidden2 = model(x, edge_index, edge_weight,hidden1,hidden2)
            # a= criterion(y_hat, labels)
            cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))
        # print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    model.eval()
    cost = 0
    hidden1 = None
    hidden2 = None
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().cuda()
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.cuda()
        edge_weight = snapshot.edge_attr.cuda()
        y_hat,hidden1,hidden2 = model(x, edge_index, edge_weight,hidden1,hidden2)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    objective_function(model, origin_train_loader, device)
    objective_function(model, test_loader, device)
    return model


def objective_function(model, dataset, device):
    accuracy = 0
    total_time = 0
    hidden1 = None
    hidden2 = None
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y, hidden1, hidden2 = model.to(device)(x, edge_index, edge_attr,hidden1,hidden2)
            victim_labels = torch.argmax(y.detach(), dim=1).long().clone().to(
                device)
            accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))

#model = GCN_selection()