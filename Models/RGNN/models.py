#two-layer GCN Model
import torch
from torch.nn import Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn.parameter import Parameter
class GCN2(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCN2, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, second_features)
        self.dropout = Dropout(p=0.5)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)


    def forward(self, x, edge_index, edge_weight):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

#three-layer GCN Model
class GCN3(torch.nn.Module):
    def __init__(self, node_features, second_features, third_features, num_classes):
        super(GCN3, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, third_features)
        self.conv_layer3 = GCNConv(third_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer3(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

class ModelfreeGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(ModelfreeGCN, self).__init__()
        self.conv_layer = GCNConv(in_channels=node_features,
                                  out_channels=node_features,
                                  improved=False,
                                  cached=False,
                                  normalize=True,
                                  add_self_loops=True,
                                  bias=False)
        '''
        self.conv_layer2 = GCNConv(in_channels=node_features,
                                  out_channels=node_features,
                                  improved=False,
                                  cached=False,
                                  normalize=True,
                                  add_self_loops=True,
                                  bias=False)
        '''
    def forward(self, x, edge_index, edge_weight):
        Weight = self.conv_layer.lin.weight
        self.conv_layer.lin.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]).cuda())
        #Weight2 = self.conv_layer2.lin.weight
        #self.conv_layer2.lin.weight = Parameter(torch.eye(Weight2.shape[0], Weight2.shape[1]))
        h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        #h = self.conv_layer2(h, edge_index, edge_weight)

        #Weight = self.conv_layer.weight
        #self.conv_layer.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]))
        #h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return h

'''
m = ModelfreeGCN(100)
m = m.cuda()
x= torch.rand(100,100).cuda()
edge_index = torch.LongTensor([[0,1,2,3],[0,1,2,4]]).cuda()
edge_attr = torch.tensor([1.,1.,1.,1.]).cuda()
b= m(x,edge_index,edge_attr)
print(b)
'''