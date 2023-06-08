from Models.GNN.models_link import GCN2,GAT2,GCN2_LSTM,GAT2_GRU
from Models.GNN.models_node import GCN2_attention,GCN2_GAT,GAT
from Models.RGCN.models import DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
from dataloader import DBLPLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import negative_sampling,to_dense_adj
import torch.nn.functional as F
def negative_sample(data, force_undirected=True):
    # 从训练集中采样与正边相同数量的负边
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

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
def train(GCN_type, train_loader, test_loader, origin_train_loader):
    features = torch.tensor(train_loader.features).shape[2]
    num_classes = torch.tensor(train_loader.targets).shape[2]
    #train_loader, test_loader = temporal_signal_split(dataset, 0.6)
    if GCN_type =='GAT2_GRU':
        model = GAT2_GRU(node_features=features, second_features=64, num_classes=32)
    else:
        return 0
    if torch.cuda.is_available():
        model = model.cuda()
    #criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    best_model = None
    best_val_auc = 0
    final_test_auc = 0
    min_epochs=10
    val_auc = test(model, origin_train_loader)
    test_auc = test(model, test_loader)
    print('epoch -1  val_auc {:.4f} test_auc {:.4f}'
              .format(val_auc, test_auc))
    model.train()
    for epoch in tqdm(range(300)):
        hidden = None
        cost = 0
        for time, snapshot in enumerate(train_loader):
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            out, hidden= model(snapshot.x.cuda(), snapshot.edge_index.cuda(), edge_label_index.cuda(),hidden)
            out = out.view(-1)
            cost += criterion(out.cuda(), edge_label.cuda())
        optimizer.zero_grad(0)
        cost = cost/(time+1)
        cost.backward()
        optimizer.step()
            # validation
        val_auc = test(model, origin_train_loader)
        test_auc = test(model, test_loader)
        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        if (epoch+1)%20==0:
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} test_auc {:.4f}'
                  .format(epoch, cost.item(), val_auc, test_auc))
    return model


def test(model, data):
    model.eval()
    score = 0
    with torch.no_grad():
        hidden = None
        for time, snapshot in enumerate(data):
            snapshot.x = F.normalize(snapshot.x, p=2, dim=1)
            snapshot.edge_label = snapshot.edge_attr
            snapshot.edge_label_index = snapshot.edge_index
            edge_label, edge_label_index = negative_sample(snapshot)
            z,hidden = model.encode(snapshot.x.cuda(), snapshot.edge_index.cuda(),hidden)
            out = model.decode(z, edge_label_index.cuda()).view(-1).sigmoid()
            model.train()
            score += roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())
    return score/(time+1)


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
