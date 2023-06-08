import torch
from torch_geometric_temporal.nn.recurrent import DCRNN,EvolveGCNO,TGCN,A3TGCN
import torch.nn.functional as F

class DCRNN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(DCRNN_RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, second_features, 1)  ##need to reset
        self.linear = torch.nn.Linear(second_features, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)

class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, num_classes)
        #self.recurrent2 = EvolveGCNO(num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
       #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)

class TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(TGCN_RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, second_features)
        #self.recurrent2 = TGCN(32, num_classes)
        self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)

class A3TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(A3TGCN_RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, second_features, 1)
        self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0],x.shape[1],1), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1)