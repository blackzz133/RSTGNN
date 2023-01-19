from Models.RGNN.models_link import GCN2,GAT2,GCN2_LSTM,GAT2_GRU,Robust_RGNN,DCRNN_RGCN,GCLSTM_RGCN
from Models.RGNN.models_node import GCN2_attention,GCN2_GAT,GAT
from Models.RGCN.models import DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
from dataloader import DBLPLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import copy as cp
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score,f1_score
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import negative_sampling,to_dense_adj
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils.convert import to_networkx
import networkx as nx

def negative_sample(data, force_undirected=True):
    # 从训练集中采样与正边相同数量的负边
    #print(data.num_nodes)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
    cycle_edge_index = [torch.arange(data.num_nodes).tolist(), torch.arange(data.num_nodes).tolist()]
    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    #print(edge_label_index.shape)
    edge_label_index = torch.cat(
        [edge_label_index, torch.tensor(cycle_edge_index)],
        dim=-1,
    )
    #print(edge_label_index.shape)
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    #print(edge_label.shape)
    edge_label = torch.cat([
        edge_label, torch.ones(data.num_nodes)
    ], dim=0)
    #print(edge_label.shape)
    #exit()
    return edge_label, edge_label_index




def objective_function(model, dataset, device):
    accuracy = 0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            victim_labels = torch.argmax(model.to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))

def GCN_selection_Link_Prediction(args, GCN_type, train_loader, test_loader, origin_train_loader, attack_method, defend_method):
    time_length = torch.tensor(train_loader.features).shape[0]
    num_features = torch.tensor(train_loader.features).shape[2]
    num_classes = torch.tensor(train_loader.targets).shape[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    #train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    if GCN_type == 'GAT2_GRU':
        model = GAT2_GRU(node_features=num_features, second_features=64, num_classes=16)
        train_type = 'RGCN'
    elif GCN_type == 'GCN2':
        model = GCN2(node_features=num_features, second_features=64, num_classes=16)
        train_type = 'GCN'
    elif GCN_type == 'Robust_RGNN':
        model = Robust_RGNN(args, device, criterion, num_features, time_length, 16)
        train_type = 'Robust'
    elif GCN_type == 'DCRNN':
        model = DCRNN_RGCN(node_features=num_features, second_features=64, num_classes=16)
        train_type = 'RGCN'
    elif GCN_type == 'GCLSTM':
        model = GCLSTM_RGCN(node_features=num_features, second_features=64, num_classes=16)
        train_type = 'RGCN2'
    else:
        return 0
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    if train_type =='RGCN':
        model = rgcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)
    if train_type =='RGCN2':
        model = rgcn_train2(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)
    elif train_type =='GCN':
        model = gcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)
    elif train_type =='Robust':
        model = robust_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader)

    return model

def robust_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    train_graphs = data_preprossing(train_loader)
    test_graphs = data_preprossing(test_loader)
    origin_graphs = data_preprossing(origin_train_loader)
    cost = 0
    edge_labels = []
    edge_label_indexes = []
    best_val_auc = 0
    min_epochs = 10
    degrees = []
    cens = []
    cen_var = []
    model.train()
    for time, snapshot in enumerate(train_graphs):
        graph = to_networkx(train_graphs[time])
        degree = torch.tensor(list(dict(graph.degree).values())) / 2 - 1
        degrees.append(degree.cuda())
        cen = torch.tensor(list(dict(nx.closeness_centrality(graph)).values()))
        cens.append(cen.cuda())

        snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
        snapshot.edge_label = snapshot.edge_attr
        snapshot.edge_label_index = snapshot.edge_index
        edge_label, edge_label_index = negative_sample(snapshot)
        edge_labels.append(edge_label.cuda())
        #print(train_graphs[time].edge_index.shape)
        #print(train_graphs[time].edge_attr.shape)
        train_graphs[time].edge_index = edge_label_index
        #print(train_graphs[time].edge_index.shape)
        train_graphs[time].edge_attr = edge_label
        #print(train_graphs[time].edge_attr.shape)
        #exit()
        edge_label_indexes.append(edge_label_index.cuda())
    time_ = len(train_graphs)
    cen_var.append(torch.zeros(cens[0].shape).cuda())
    for t in range(1, time_ - 1):
        cen_var.append((abs(cens[t] - cens[t - 1]) + abs(cens[t + 1] - cens[t])).cuda())
    cen_var.append(torch.zeros(cens[time_ - 1].shape).cuda())
    train_graphs_cpu = cp.deepcopy(train_graphs)
    for epoch in tqdm(range(800)):
        cost = model.get_total_loss(train_graphs, edge_label_indexes, edge_labels, degrees, cen_var)
        #if (epoch + 1) % 20 == 0:
            #print('The ' + str(epoch) + ' training loss is ' + str(cost))
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
        per_auc = 0

        #if epoch + 1 > min_epochs and val_auc > best_val_auc:
            #best_val_auc = val_auc
            #final_test_auc = test_auc
        if (epoch + 1) % 20 == 0:
            #per_auc = test3(model, train_graphs_cpu)
            #val_auc = test3(model, origin_graphs)
            per_auc=0
            val_auc=0
            test_auc = test3(model, test_graphs)

            print('epoch {:03d} train_loss {:.8f} train_auc {:.4f} ori_auc {:.4f} test_auc {:.4f}'
                  .format(epoch, cost.item(),per_auc, val_auc, test_auc))
    #model.eval()
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    #objection_function3(model, test_loader, device)


def gcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    best_model = None
    best_val_auc = 0
    final_test_auc = 0
    min_epochs = 10
    val_auc = test2(model, origin_train_loader)
    test_auc = test2(model, test_loader)
    print('epoch -1  val_auc {:.4f} test_auc {:.4f}'
          .format(val_auc, test_auc))
    model.train()
    for epoch in tqdm(range(800)):
        cost = 0
        for time, snapshot in enumerate(train_loader):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            out= model(snapshot.x.cuda(), snapshot.edge_index.cuda(), edge_label_index.cuda())
            out = out.view(-1)
            cost += criterion(out[:-node_num].cuda(), edge_label[:-node_num].cuda())
        optimizer.zero_grad(0)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        # validation

        #if epoch + 1 > min_epochs and val_auc > best_val_auc:
            #best_val_auc = val_auc
            #final_test_auc = test_auc
        if (epoch + 1) % 20 == 0:
            per_auc = test2(model, train_loader)
            val_auc = test2(model, origin_train_loader)
            test_auc = test2(model, test_loader)
            print('epoch {:03d} train_loss {:.8f} train_auc {:.4f} origin_auc {:.4f} test_auc {:.4f}'
                  .format(epoch, cost.item(), per_auc, val_auc, test_auc))
    return model

def rgcn_train(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    best_model = None
    best_val_auc = 0
    final_test_auc = 0
    min_epochs = 10
    val_auc = test(model, origin_train_loader)
    test_auc = test(model, test_loader)
    print('epoch -1  val_auc {:.4f} test_auc {:.4f}'
          .format(val_auc, test_auc))
    model.train()
    for epoch in tqdm(range(800)):
        hidden = None
        cost = 0
        for time, snapshot in enumerate(train_loader):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            out, hidden = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), edge_label_index.cuda(), hidden)
            out = out.view(-1)
            cost += criterion(out[:-node_num].cuda(), edge_label[:-node_num].cuda())
        optimizer.zero_grad(0)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        # validation

        #if epoch + 1 > min_epochs and val_auc > best_val_auc:
            #best_val_auc = val_auc
            #final_test_auc = test_auc
        if (epoch + 1) % 20 == 0:
            #per_auc = test(model, train_loader)
            #val_auc = test(model, origin_train_loader)
            per_auc = 0
            val_auc = 0
            test_auc = test(model, test_loader)
            print('epoch {:03d} train_loss {:.8f} train_auc {:.4f} origin_auc {:.4f} test_auc {:.4f}'
                  .format(epoch, cost.item(), per_auc, val_auc, test_auc))
    return model

def rgcn_train2(model, criterion, optimizer, scheduler, train_loader, test_loader, origin_train_loader):
    best_model = None
    best_val_auc = 0
    final_test_auc = 0
    min_epochs = 10
    val_auc = test5(model, origin_train_loader)
    test_auc = test5(model, test_loader)
    print('epoch -1  val_auc {:.4f} test_auc {:.4f}'
          .format(val_auc, test_auc))
    model.train()
    for epoch in tqdm(range(800)):
        hidden1 = None
        hidden2 =None
        cost = 0
        for time, snapshot in enumerate(train_loader):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            out, hidden1, hidden2 = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), edge_label_index.cuda(), hidden1, hidden2)
            out = out.view(-1)
            cost += criterion(out[:-node_num].cuda(), edge_label[:-node_num].cuda())
        optimizer.zero_grad(0)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        # validation
        #if epoch + 1 > min_epochs and val_auc > best_val_auc:
            #best_val_auc = val_auc
            #final_test_auc = test_auc
        if (epoch + 1) % 20 == 0:
            #per_auc = test5(model, train_loader)
            #val_auc = test5(model, origin_train_loader)
            per_auc = 0
            val_auc = 0
            test_auc = test5(model, test_loader)
            print('epoch {:03d} train_loss {:.8f} train_auc {:.4f} origin_auc {:.4f} test_auc {:.4f}'
                  .format(epoch, cost.item(), per_auc, val_auc, test_auc))
    return model

def test(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        hidden = None
        for time, snapshot in enumerate(data):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            z,hidden = model.encode(snapshot.x.cuda(), snapshot.edge_index.cuda(),hidden)
            out = model.decode(z, edge_label_index.cuda()).view(-1).sigmoid()
            score += roc_auc_score(edge_label[:-node_num].cpu().numpy(), out[:-node_num].cpu().numpy())
    model.train()
    return score/(time+1)

def test5(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        hidden1 = None
        hidden2 = None
        for time, snapshot in enumerate(data):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            z,hidden1,hidden2 = model.encode(snapshot.x.cuda(), snapshot.edge_index.cuda(),hidden1, hidden2)
            out = model.decode(z, edge_label_index.cuda()).view(-1).sigmoid()
            score += roc_auc_score(edge_label[:-node_num].cpu().numpy(), out[:-node_num].cpu().numpy())
    model.train()
    return score/(time+1)

def test2(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        for time, snapshot in enumerate(data):
            node_num = snapshot.x.shape[0]
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            z = model.encode(snapshot.x.cuda(), snapshot.edge_index.cuda())
            out = model.decode(z, edge_label_index.cuda()).view(-1).sigmoid()
            score += roc_auc_score(edge_label[:-node_num].cpu().numpy(), out[:-node_num].cpu().numpy())
    model.train()
    return score/(time+1)

def test3(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        hidden = None
        for time, snapshot in enumerate(data):
            graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y)
            node_num = graph.x.shape[0]
            graph.x = F.normalize(graph.x, p=2, dim=1)
            graph.edge_label = graph.edge_attr
            graph.edge_label_index = graph.edge_index
            #snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            #snapshot.edge_label = snapshot.edge_attr
            #snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(graph)
            graph.x = graph.x.cuda()
            graph.edge_index = edge_label_index.cuda()
            # print(train_graphs[time].edge_index.shape)
            graph.edge_attr = edge_label.cuda()
            #graph.y = graph.y.cuda()
            z, hidden, coe1, coe2 = model.encode(graph, hidden)
            out = model.decode(z.squeeze(), edge_label_index.cuda()).view(-1).sigmoid()
            score += roc_auc_score(edge_label[:-node_num].cpu().numpy(), out[:-node_num].cpu().numpy())
    model.train()
    return score/(time+1)


def test4(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        hidden = None
        for time, snapshot in enumerate(data):
            node_num = snapshot.x.shape[0]
            graph = torch_geometric.data.data.Data(x=snapshot.x.cuda(), edge_index=snapshot.edge_index.cuda(), edge_attr=snapshot.edge_attr.cuda(), y=snapshot.y.cuda())
            graph.x = F.normalize(graph.x, p=2, dim=1)
            graph.edge_label = graph.edge_attr
            graph.edge_label_index = graph.edge_index
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            cycle_edge_index = [torch.arange(graph.x.shape[0]).tolist(), torch.arange(graph.x.shape[0]).tolist()]
            train_edge_label =  torch.cat([edge_label_index, torch.tensor(cycle_edge_index)],dim=-1,)
            train_edge_label_index = torch.cat([edge_label, torch.ones(graph.x.shape[0])], dim=0)
            graph.edge_index = train_edge_label.cuda()
            # print(train_graphs[time].edge_index.shape)
            graph.edge_attr = train_edge_label_index.cuda()
            z, hidden = model.encode(graph, hidden)
            out = model.decode(z, train_edge_label_index.cuda()).view(-1).sigmoid()
            score += roc_auc_score(train_edge_label[:-node_num].cpu().numpy(), out[:-node_num].cpu().numpy())
    model.train()
    return score/(time+1)

def data_preprossing(dataset):
    graphs = []
    for time, snapshot in enumerate(dataset):
        #edge_index = snapshot.edge_index
        #edge_attr = snapshot.edge_attr
        #edge_index_0 = torch.cat((edge_index[0], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        #edge_index_1 = torch.cat((edge_index[1], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        #new_edge_index = torch.tensor([edge_index_0,edge_index_1])
        #new_edge_attr = torch.cat((edge_attr, torch.ones(snapshot.x.shape[0])), dim=0)
        graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y)
        graphs.append(graph)
    return graphs


def objection_function3(model, dataset, device):
    accuracy = 0
    total_time = 0
    graphs, y = dataset
    y_hat = model.to(device)(dataset)
    y = y.numpy()
    y_labels = np.argmax(y, axis=2)
    accuracy = torch.eq(y_hat, y_labels).sum()/y_labels.shape[0]/y_labels.shape[1]
    print('The accuracy result of rgcn is ' + str(accuracy))

#model = train()

#DBLP5
#GCN2
#epoch 299 train_loss 0.39463621 val_auc 0.9957 test_auc 0.9927
#GCN2_LSTM
#epoch 299 train_loss 0.38928372 val_auc 0.9943 test_auc 0.9923
#GAT2
#epoch 299 train_loss 0.37790093 val_auc 0.9971 test_auc 0.9858
#GAT2_GRU
#epoch 299 train_loss 0.37912297 val_auc 0.9969 test_auc 0.9862
#GAT2_GRU(new)
#epoch 299 train_loss 0.37386113 val_auc 0.9982 test_auc 0.9835
#DBLP3
#GCN2
#epoch 299 train_loss 0.41991705 val_auc 0.9632 test_auc 0.9669
#GCN2_LSTM
#epoch 299 train_loss 0.42514834 val_auc 0.9686 test_auc 0.9692
#GAT2
#epoch 299 train_loss 0.40581116 val_auc 0.9750 test_auc 0.9684
#GAT2_GRU
#epoch 299 train_loss 0.40089986 val_auc 0.9746 test_auc 0.9633

#GAT2_GRU(new)
#epoch 299 train_loss 0.38128600 val_auc 0.9909 test_auc 0.9841
