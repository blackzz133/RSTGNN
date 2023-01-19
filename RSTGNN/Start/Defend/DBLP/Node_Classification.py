from Models.RGNN.models import GCN2, GCN3
from Models.RGNN.models_node import GCN2_attention,GCN2_GAT,GAT,New_GAT_LSTM,Evolve_GAT, Robust_RGNN, GCLSTM_RGCN, DCRNN_RGCN
from Models.RGCN.models import DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
from dataloader import DBLPLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx
import networkx as nx
#from deeprobust.graph.global_attack import Metattack, MetaApprox
from torch_geometric.utils import to_dense_adj
def GCN_selection_Node_Classification(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method):
    time_length = torch.tensor(train_loader.features).shape[0]
    num_features = torch.tensor(train_loader.features).shape[2]
    num_classes = torch.tensor(train_loader.targets).shape[2]
    train_graphs = data_preprossing(train_loader)
    test_graphs = data_preprossing(test_loader)
    origin_graphs = data_preprossing(origin_train_loader)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #print(torch.tensor(train_loader.features).shape)
    #print(torch.tensor(train_loader.targets).shape)

    if GCN_type == 'GAT':
        model = GAT(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'RGCN'
    elif GCN_type == 'GCLSTM':
        model = GCLSTM_RGCN(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'RGCN'
    elif GCN_type == 'DCRNN':
        model = DCRNN_RGCN(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'RGCN'
    elif GCN_type == 'Evolve_GAT':
        model = Evolve_GAT(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'RGCN'
    elif GCN_type =='GCN2':
        model = GCN2(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'GCN'
    elif GCN_type == 'Attention2':
        model = GAT(node_features=num_features, second_features=32, num_classes=num_classes)
        train_type = 'RGCN'
    elif GCN_type == 'Robust_RGNN':
        model = Robust_RGNN(args, device, num_features, time_length, num_classes)
        train_type = 'Robust'

    else:
        return 0
    if torch.cuda.is_available():
        model = model.cuda()
    #train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    # print(train_loader.features)
    # print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    model.train()
    if train_type =='GCN':
        model = gcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)
    elif train_type == 'RGCN':
        model = rgcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)
    elif train_type =='Robust':
        model = robust_train(model, criterion, optimizer, scheduler, train_graphs, test_graphs, origin_graphs)
    else:
        return model
    return model

def rgcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    for epoch in tqdm(range(800)):
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
            y_hat, hidden1, hidden2 = model(x, edge_index, edge_weight, hidden1, hidden2)
            # a= criterion(y_hat, labels)
            cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))
            model.eval()
            objective_function(model, train_loader, device)
            objective_function(model, origin_train_loader, device)
            objective_function(model, test_loader, device)
            model.train()
        # print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
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





def gcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    for epoch in tqdm(range(800)):
        cost = 0
        for time, snapshot in enumerate(train_loader):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().cuda()
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.cuda()
            edge_weight = snapshot.edge_attr.cuda()
            y_hat = model(x, edge_index, edge_weight)
            # a= criterion(y_hat, labels)
            cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))
            model.eval()
            objective_function2(model, train_loader, device)
            objective_function2(model, origin_train_loader, device)
            objective_function2(model, test_loader, device)
            model.train()
        # print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().cuda()
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.cuda()
        edge_weight = snapshot.edge_attr.cuda()
        y_hat = model(x, edge_index, edge_weight)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))

def robust_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    cost = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    degrees = []
    time = len(train_loader)
    #centrality
    cens = []
    cen_var = []
    for t in range(time):
        graph = to_networkx(train_loader[t])
        degree = torch.tensor(list(dict(graph.degree).values())) / 2 - 1
        degrees.append(degree.cuda())
        cen = torch.tensor(list(dict(nx.closeness_centrality(graph)).values()))
        cens.append(cen.cuda())
    cen_var.append(torch.zeros(cens[0].shape).cuda())
    for t in range(1,time-1):
        cen_var.append((abs(cens[t]-cens[t-1])+abs(cens[t+1]-cens[t])).cuda())
    cen_var.append(torch.zeros(cens[time-1].shape).cuda())
    for epoch in tqdm(range(800)):
        cost = model.get_total_loss(train_loader, degrees, cen_var)
        #cost, coe1, coe2 = model.get_loss(train_loader)
        #cost += model.get_spatial_loss(train_loader, coe1, coe2)
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))
            model.eval()
            objective_function3(model, train_loader, device)
            objective_function3(model, origin_train_loader, device)
            objective_function3(model, test_loader, device)
            model.train()
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
    model.eval()

def objective_function(model, dataset, device): #RGCN_objective_function
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
            y_hat,hidden1,hidden2 = model.to(device)(x, edge_index, edge_attr,hidden1,hidden2)
            victim_labels = torch.argmax(y_hat.detach(), dim=1).long().clone().to(
                device)
            accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))

def objective_function2(model, dataset, device): #GCN_objective_function
    accuracy = 0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y_hat = model.to(device)(x, edge_index, edge_attr)
            victim_labels = torch.argmax(y_hat.detach(), dim=1).long().clone().to(
                device)
            accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
            total_time += 1
    print('The accuracy result of gcn is ' + str(accuracy / total_time))

def objective_function3(model, dataset, device): #robust
    y, coe1, coe2 = model(dataset)
    accuracy = 0
    for time,snapshot in enumerate(dataset):
        y_hat = torch.argmax(y[time], axis=1)
        y_label = torch.argmax(snapshot.y, axis=1)
        accuracy += torch.eq(y_hat, y_label).sum() / y_hat.shape[0]
    print('The accuracy result of rgcn is ' + str(accuracy/len(y)))

def data_preprossing(dataset):
    graphs = []
    for time, snapshot in enumerate(dataset):
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        edge_index_0 = torch.cat((edge_index[0], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        edge_index_1 = torch.cat((edge_index[1], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        new_edge_index = torch.tensor([edge_index_0,edge_index_1])
        new_edge_attr = torch.cat((edge_attr, torch.ones(snapshot.x.shape[0])), dim=0)
        graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=snapshot.y)
        graphs.append(graph)
    return graphs

#with spatial_loss
#The accuracy result of rgcn is tensor(0.9021, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7368, device='cuda:0')

#without spatial_loss
#The accuracy result of rgcn is tensor(0.8972, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7375, device='cuda:0')

#with_loss mu1=0.1
#The accuracy result of rgcn is tensor(0.8967, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7431, device='cuda:0')
#mu1=0.2
#The accuracy result of rgcn is tensor(0.8960, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7407, device='cuda:0')