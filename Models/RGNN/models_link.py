import torch
from torch_geometric.nn import GCNConv,GATConv
from torch.nn import LSTM,GRU
import torch.nn.functional as F
from Models.layers import StructuralAttentionLayer, TemporalAttentionLayer, TemporalAttentionLayer3
import numpy as np
import torch_geometric
from torch_geometric.utils import negative_sampling,to_dense_adj
from torch_scatter import scatter
from torch_geometric_temporal.nn.recurrent import DCRNN,EvolveGCNO,TGCN,A3TGCN,GCLSTM

class GCN2(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCN2, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, second_features)

        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)

    def encode(self, x, edge_index):
        x = self.conv_layer(x, edge_index).relu()
        std = torch.std(x, dim=0)
        mean = torch.mean(x, dim=0)
        x= x+torch.normal(mean=mean, std=std)
        x = self.conv_layer2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
class GCN2_LSTM(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCN2_LSTM, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.recurrent_layer1 = LSTM(input_size=node_features, hidden_size=node_features, num_layers=1)
        self.conv_layer = GCNConv(node_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)

    def encode(self, x, edge_index):
        h, _ = self.recurrent_layer1(x[None, :, :])
        # print(h.shape)
        h = h.squeeze()
        h = self.conv_layer(h, edge_index).relu()
        h = self.conv_layer2(h, edge_index)
        return h

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

class GAT2(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GAT2, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GATConv(node_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GATConv(second_features, num_classes)

    def encode(self, x, edge_index):
        x = self.conv_layer(x, edge_index).relu()
        x = self.conv_layer2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

class GAT2_GRU(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GAT2_GRU, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GATConv(node_features, second_features)
        self.recurrent_layer1 = GRU(input_size=second_features, hidden_size=second_features, num_layers=1)
        self.recurrent_layer2 = LSTM(input_size=second_features, hidden_size=second_features, num_layers=1)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GATConv(second_features, num_classes)

    def encode(self, x, edge_index,hidden):
        h = self.conv_layer(x, edge_index).relu()
        h, hidden = self.recurrent_layer1(h[None, :, :],hidden)
        # print(h.shape)
        h = h.squeeze()
        h = self.conv_layer2(h, edge_index)
        return h,hidden

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index, hidden=None):
        z,hidden = self.encode(x, edge_index,hidden)
        z = z.relu()
        return self.decode(z, edge_label_index),hidden

class DCRNN_RGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(DCRNN_RGCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer1 = GCNConv(node_features, node_features)
        #self.conv_layer = GATConv(node_features, second_features)
        self.recurrent_layer1 = DCRNN(node_features, second_features, 1)
        #self.linear = torch.nn.Linear(32, num_classes)
        #self.conv_layer2 = GATConv(node_features, node_features)
        self.linear = torch.nn.Linear(second_features, num_classes)

    def encode(self, x, edge_index, hidden):
        h = self.conv_layer1(x, edge_index)
        h = self.recurrent_layer1(h, edge_index,None, hidden)
        hidden = h
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1), hidden

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index, hidden=None):
        z,hidden = self.encode(x, edge_index,hidden)
        return self.decode(z, edge_label_index),hidden

class GCLSTM_RGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCLSTM_RGCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        #self.conv_layer = GATConv(node_features, second_features)
        self.conv_layer1 = GCNConv(node_features, node_features)
        self.recurrent_layer1 = GCLSTM(node_features, second_features, 1)
        #self.linear = torch.nn.Linear(32, num_classes)
        #self.conv_layer2 = GATConv(second_features, num_classes)
        self.linear = torch.nn.Linear(second_features, num_classes)
    def encode(self, x, edge_index, hidden1=None,hidden2=None):
        h = self.conv_layer1(x, edge_index)
        hidden1,hidden2 = self.recurrent_layer1(h, edge_index,None, hidden1, hidden2)
        h = F.relu(hidden1)
        h = self.linear(h)
        return F.softmax(h, dim=1), hidden1, hidden2
    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def decode_all(self,z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, edge_label_index, hidden1=None,hidden2=None):
        z,hidden,hidden2 = self.encode(x, edge_index,hidden1,hidden2)
        return self.decode(z, edge_label_index),hidden1, hidden2

class Robust_RGNN(torch.nn.Module):
    def __init__(self, args, device, criterion, num_features, time_length, num_classes):
        super(Robust_RGNN, self).__init__()
        self.args = args
        self.num_time_steps = time_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.structural_attn1, self.structural_attn2, self.recurrent_layer1,self.temporal_attn, self.linear, self.dropout = self.build_model(num_features,num_classes)
        self.device = device
        self.criterion = criterion
        #torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
        #, self.temporal_attn
    def build_model(self, num_features, num_classes):
        input_dim = self.num_features
        structural_attention_layers = torch.nn.Sequential()
        # num of structure layer is two
        structural_attn1 = StructuralAttentionLayer(input_dim=input_dim,
                                                        output_dim=self.structural_layer_config[0],
                                                        n_heads=self.structural_head_config[0],
                                                        attn_drop=self.spatial_drop,
                                                        ffd_drop=self.spatial_drop,
                                                        residual=self.args.residual)
        structural_attn2 = StructuralAttentionLayer(input_dim=self.structural_layer_config[0],
                                                    output_dim=self.structural_layer_config[1],
                                                    n_heads=self.structural_head_config[1],
                                                    attn_drop=self.spatial_drop,
                                                    ffd_drop=self.spatial_drop,
                                                    residual=self.args.residual)
        recurrent_layer1 = LSTM(input_size=self.structural_layer_config[1], hidden_size=self.temporal_layer_config[0], num_layers=1)

        temporal_attn = TemporalAttentionLayer3(input_dim=self.temporal_layer_config[0],
                                               n_heads=self.temporal_head_config[0],
                                               num_time_steps=self.num_time_steps,
                                               attn_drop=self.temporal_drop,
                                               residual=self.args.residual,type='edge')


        linear =  torch.nn.Linear(in_features=self.temporal_layer_config[0], out_features=4)
        dropout = torch.nn.Dropout(p=0.7)




        return structural_attn1, structural_attn2, recurrent_layer1, temporal_attn, linear, dropout

    def forward2(self, graphs, edge_label_indexes, hidden = None):
        #用于存储spatial-temporal输出，spatial为双重GAT, temporal为LSTM
        st_out = []
        out = None
        #hidden = None
        coe1 = []
        coe2 = []
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            z, hidden, c1, c2 = self.encode(graph, hidden)
            z = self.linear(z)
            out1 = self.decode(z, edge_label_indexes[t])
            st_out.append(out1)
            coe1.append(c1)
            coe2.append(c2)

        #temporal_out = self.temporal_attn(out).transpose(0,1)
            #这作预测
            #st_out.append(out3)
        #temporal_out = self.temporal_attn(st_out)

        '''
        structural_outputs = [h[:,None,:] for h in st_out]
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1)  # [N, T, F]
        #temporal_output = self.temporal_attn(structural_outputs_padded)
        '''
        return st_out, coe1, coe2

    def forward(self, graphs, edge_label_indexes, hidden = None):
        #用于存储spatial-temporal输出，spatial为双重GAT, temporal为LSTM
        st_out = []
        time = len(graphs)
        out = None
        #hidden = None
        coe1 = []
        coe2 = []
        for t in range(time):
            graph = graphs[t].cuda()
            z, hidden, c1, c2 = self.encode(graph, hidden)
            if out==None:
                out= z.transpose(0,1)
            else:
                out = torch.cat((out,z.transpose(0,1)),dim=1)
            coe1.append(c1)
            coe2.append(c2)
        out2 = self.temporal_attn(out).transpose(0,1)
        out2 = self.dropout(out2)
        #out2 = self.linear(out2)#only for reddit
        #
        for t in range(time):
            st_out.append(self.decode(out2[t].squeeze(), edge_label_indexes[t]))
        #temporal_out = self.temporal_attn(out).transpose(0,1)
            #这作预测
            #st_out.append(out3)
        #temporal_out = self.temporal_attn(st_out)

        '''
        structural_outputs = [h[:,None,:] for h in st_out]
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1)  # [N, T, F]
        #temporal_output = self.temporal_attn(structural_outputs_padded)
        '''
        return st_out, coe1, coe2
    def encode(self, graph, hidden):
        out1, c1 = self.structural_attn1(graph)
        out2, c2 = self.structural_attn2(out1)
        out2.x = out2.x.relu()
        out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
        return out3, hidden, c1, c2
        #return out2.x[None, :, :], hidden, c1,c2

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        # src为起点的节点特征集，dst为终点的特征集
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def get_total_loss(self, graphs, edge_label_indexes, edge_labels, degrees, cen_var):
        mu1 = 0.1#0.1
        mu2 = 0.2#0.2
        loss, coe1, coe2 = self.get_loss(graphs, edge_label_indexes, edge_labels)
        loss += mu1*self.get_spatial_loss(graphs, coe1, coe2, degrees)
        loss += mu2*self.get_temporal_loss(graphs, coe1, coe2, cen_var)
        return loss


    def get_loss(self, graphs, edge_label_indexes, edge_labels):  #需要修改
        # run gnn
        final_emb, coe1, coe2 = self.forward(graphs, edge_label_indexes)  # [N, T, F]
        self.graph_loss = 0
        for t in range(len(graphs)):
            #y = graphs[t].y
            node_num = graphs[t].x.shape[0]
            #y = y.numpy()
            #y = torch.argmax(y, dim=1)
            #y = np.argmax(y, axis=1)
            #labels = torch.from_numpy(y).long().cuda()
            emb_t = final_emb[t].squeeze()  # [N, F]
            #y_hat = F.softmax(emb_t, dim=1)
            #self.graph_loss += self.criterion(emb_t[:-node_num].cuda(), edge_labels[t][:-node_num].cuda())
            self.graph_loss += self.criterion(emb_t[:-node_num].cuda(), edge_labels[t][:-node_num].cuda())
        return self.graph_loss/self.num_time_steps, coe1, coe2

    def get_spatial_loss(self, graphs, coe1, coe2, degrees):
        self.graph_spatial_loss = 0
        for t in range(len(graphs)):
            c1 = coe1[t]
            c2 = coe2[t]
            node_num = graphs[t].x.shape[0]
            # graph = to_networkx(graphs[t])
            # x = graphs[t].x
            edge_index = graphs[t].edge_index
            # x_origin = cp.deepcopy(x)
            # edge_weight = graphs[t].edge_attr.reshape(-1, 1)
            # node_num = x.shape[0]
            alph1 = scatter(c1, edge_index[0], dim=0, reduce="sum")
            alph1 = torch.norm(alph1, p=2, dim=1)
            alph2 = scatter(c2, edge_index[0], dim=0, reduce="sum")
            alph2 = torch.norm(alph2, p=2, dim=1)
            sim = torch.cosine_similarity(graphs[t].x[edge_index[0]], graphs[t].x[edge_index[1]])  # +1
            # sim = F.pairwise_distance(graphs[t].x[edge_index[0]][:-node_num],graphs[t].x[edge_index[1][:-node_num]], p=2)
            sim1 = scatter(sim, edge_index[1], dim=0, reduce="mean")
            # self.graph_spatial_loss += torch.mean((alph1 + alph2) / 2 - degrees[t])
            self.graph_spatial_loss += torch.mean((alph1 + alph2) / 2 - sim1 / (degrees[t] + 1))
        return self.graph_spatial_loss / len(graphs)

    def get_temporal_loss(self, graphs, coe1, coe2, cen_var):
        self.graph_temporal_loss = 0
        for t in range(len(graphs)):
            c1 = coe1[t]
            c2 = coe2[t]
            edge_index = graphs[t].edge_index
            # x_origin = cp.deepcopy(x)
            # edge_weight = graphs[t].edge_attr.reshape(-1, 1)
            # node_num = x.shape[0]
            alph1 = scatter(c1, edge_index[0], dim=0, reduce="sum")
            alph1 = torch.norm(alph1, p=2, dim=1)
            alph2 = scatter(c2, edge_index[0], dim=0, reduce="sum")
            alph2 = torch.norm(alph2, p=2, dim=1)
            self.graph_temporal_loss += torch.mean(cen_var[t] * torch.mean((alph1 + alph2) / 2))
        return self.graph_temporal_loss / len(graphs)

