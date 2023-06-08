import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
#from torch_geometric.nn import GATConv
from Models.RGNN.gat_conv import GATConv
from torch.nn import LSTM,GRU
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from random import gauss
from Models.layers import StructuralAttentionLayer, TemporalAttentionLayer, TemporalAttentionLayer2, TemporalAttentionLayer3
import numpy as np
import torch_geometric
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric_temporal.nn.recurrent import DCRNN,EvolveGCNO,TGCN,A3TGCN,GCLSTM




class GCN2_attention(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes,periods):
        super(GCN2_attention, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.periods = periods
        #self.gat_layer = GATConv(node_features,node_features)
        self.conv_layer = GCNConv(node_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)
        self._attention = torch.empty(self.periods)
        torch.nn.init.uniform_(self._attention)
        #self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        #h = self.encoder(x, edge_index,edge_weight)
        #data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        #batch = Batch.from_data_list(data_list)
        #h = self.gat_layer(batch.x, batch.edge_index, batch.edge_attr)
        #h = self.conv_layer(x, edge_index, edge_weight)
        h0 = x.view(x.shape[0],x.shape[1],1)
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self.conv_layer(h0[:, :, period], edge_index, edge_weight)
        #h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        h = F.relu(H_accum)
        #print(h.shape)
        #print(h.shape)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

class GCN2_GAT(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCN2_GAT, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.gat_layer = GATConv(node_features,node_features,heads=8)
        self.conv_layer = GCNConv(node_features, second_features)
        #self.gat = GATConv(second_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)
        #self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        #h = self.encoder(x, edge_index,edge_weight)
        #data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        #batch = Batch.from_data_list(data_list)
        h = self.gat_layer(x, edge_index)
        #h = self.conv_layer(x, edge_index, edge_weight)
        h = self.conv_layer(x, edge_index, edge_weight)
        #h0 = h.view(1, h.shape[0], h.shape[1])
        #h = self.gat(h0, edge_index, edge_weight)
        #h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        h = F.relu(h)
        #print(h.shape)
        #print(h.shape)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

class GCN2_GAT(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCN2_GAT, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.gat_layer = GATConv(node_features,node_features,heads=8)
        self.conv_layer = GCNConv(node_features, second_features)
        #self.gat = GATConv(second_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(second_features, num_classes)
        #self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight):
        #h = self.encoder(x, edge_index,edge_weight)
        #data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        #batch = Batch.from_data_list(data_list)
        h = self.gat_layer(x, edge_index)
        #h = self.conv_layer(x, edge_index, edge_weight)
        h = self.conv_layer(x, edge_index, edge_weight)
        #h0 = h.view(1, h.shape[0], h.shape[1])
        #h = self.gat(h0, edge_index, edge_weight)
        #h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        h = F.relu(h)
        #print(h.shape)
        #print(h.shape)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GAT, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.heads = 8
        self.gat_layer1 = GATConv(node_features, second_features, heads=self.heads, dropout=0.6)
        self.gat_layer2 = GATConv(second_features, second_features, heads=self.heads, dropout=0.6)
        self.recurrent_layer1 = LSTM(input_size=second_features * self.heads, hidden_size=second_features, num_layers=1)
        #self.temporal_attention = TemporalAttentionLayer()
        self.recurrent_layer2 = GRU(input_size=second_features * self.heads, hidden_size=second_features, num_layers=1)


        #self.gat = GATConv(second_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        #self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight,hidden1=None,hidden2=None):
        #h = self.encoder(x, edge_index,edge_weight)
        #data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        #batch = Batch.from_data_list(data_list)
        #
        #H_accum = 0
        #probs = torch.nn.functional.softmax(self.attention, dim=0)
        h = self.gat_layer1(x, edge_index)
        h, hidden2 = self.recurrent_layer1(h[None, :, :],hidden2)
        #print(h.shape)
        h = h.squeeze()
        #print(h.shape)
        #print(h.shape)
        #h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.gat_layer2(h, edge_index)
        #h0 = h.view(1, h.shape[0], h.shape[1])
        #h = self.gat(h0, edge_index, edge_weight)
        #h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        #print(h.shape)
        #print(h.shape)
        #h = self.linear(h)
        return F.softmax(h, dim=1), hidden1, hidden2


class Evolve_GAT(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(Evolve_GAT, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.heads = 8
        self.recurrent_layer1 = LSTM(input_size=node_features, hidden_size=node_features, num_layers=1)
        self.gat_layer1 = GATConv(node_features, second_features, heads=self.heads, dropout=0.6)
        self.recurrent_layer2 = GRU(input_size=second_features * self.heads, hidden_size=second_features, num_layers=1)
        self.gat_layer2 = GATConv(second_features, num_classes, heads=1, concat=False, dropout=0.6)

        # self.gat = GATConv(second_features, second_features)
        # self.linear = torch.nn.Linear(32, num_classes)
        # self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight, hidden1=None, hidden2=None):
        # h = self.encoder(x, edge_index,edge_weight)
        # data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        # batch = Batch.from_data_list(data_list)
        #
        h, hidden1 = self.recurrent_layer1(x[None, :, :], hidden1)
        # print(h.shape)
        h = h.squeeze()
        h = self.gat_layer1(h, edge_index)
        h, hidden2 = self.recurrent_layer2(h[None, :, :], hidden2)
        # print(h.shape)
        h = h.squeeze()
        # print(h.shape)
        # print(h.shape)
        # h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.gat_layer2(h, edge_index)
        # h0 = h.view(1, h.shape[0], h.shape[1])
        # h = self.gat(h0, edge_index, edge_weight)
        # h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        # print(h.shape)
        # print(h.shape)
        # h = self.linear(h)
        return F.softmax(h, dim=1), hidden1, hidden2

class GCLSTM_RGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(GCLSTM_RGCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes

        #self.recurrent_layer1 = DCRNN(node_features, second_features, 1)
        #self.recurrent_layer1 = TGCN(node_features, second_features)
        self.conv_layer1 = GCNConv(node_features, node_features)
        self.recurrent_layer1 = GCLSTM(node_features, second_features, 1)
        #self.recurrent_layer1 = A3TGCN(node_features, second_features, 1)
        self.linear = torch.nn.Linear(second_features, num_classes)
        # self.gat = GATConv(second_features, second_features)
        # self.linear = torch.nn.Linear(32, num_classes)
        # self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight, hidden1=None, hidden2 = None):
        # h = self.encoder(x, edge_index,edge_weight)
        # data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        # batch = Batch.from_data_list(data_list)
        #
        h = self.conv_layer1(x, edge_index, edge_weight)
        hidden1, hidden2 = self.recurrent_layer1(h, edge_index, edge_weight, hidden1, hidden2)
        # print(h.shape)
        #h = hidden1.squeeze()
        # print(h.shape)
        # print(h.shape)
        # h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(hidden1)
        h = self.linear(h)
        # h0 = h.view(1, h.shape[0], h.shape[1])
        # h = self.gat(h0, edge_index, edge_weight)
        # h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        # print(h.shape)
        # print(h.shape)
        # h = self.linear(h)
        return F.softmax(h, dim=1), hidden1, hidden2

class DCRNN_RGCN(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(DCRNN_RGCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        #self.recurrent_layer1 = DCRNN(node_features, second_features, 1)
        #self.recurrent_layer1 = TGCN(node_features, second_features)
        #self.gcn = GCNConv(node_features, node_features)
        self.recurrent_layer1 = DCRNN(node_features, second_features, 1)
        self.conv_layer1 = GCNConv(node_features, node_features)
        #self.recurrent_layer1 = A3TGCN(node_features, second_features, 1)
        self.linear = torch.nn.Linear(second_features, num_classes)
        #self.gat = GATConv(second_features, second_features)
        # self.linear = torch.nn.Linear(32, num_classes)
        # self.linear = torch.nn.Linear(second_features, num_classes)

    def forward(self, x, edge_index, edge_weight, hidden1=None, hidden2 = None):
        # h = self.encoder(x, edge_index,edge_weight)
        # data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        # batch = Batch.from_data_list(data_list)
        #
        h = self.conv_layer1(x, edge_index, edge_weight)
        #h = self.gcn(x,edge_index,edge_weight)
        h = self.recurrent_layer1(h, edge_index, edge_weight, hidden1)
        hidden1 = h
        # print(h.shape)
        #h = hidden1.squeeze()
        # print(h.shape)
        # print(h.shape)
        # h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        # h0 = h.view(1, h.shape[0], h.shape[1])
        # h = self.gat(h0, edge_index, edge_weight)
        # h = [Data(x=h_, edge_index=edge_index,edge_attr=edge_weight) for h_ in h]
        # print(h.shape)
        # print(h.shape)
        # h = self.linear(h)
        return F.softmax(h, dim=1), hidden1, hidden2

class New_GAT_LSTM(torch.nn.Module):
    def __init__(self, node_features, second_features, num_classes):
        super(New_GAT_LSTM, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.heads = 8
        self.gat_layer1 = GATConv(node_features,second_features,heads=self.heads, dropout=0.6)
        self.gat_layer2 = GATConv(second_features*self.heads, second_features,heads=1,concat=False, dropout=0.6)
        self.recurrent_layer1 = LSTM(input_size=node_features, hidden_size=num_classes, num_layers=1)

        #self.gat = GATConv(second_features, second_features)
        #self.linear = torch.nn.Linear(32, num_classes)
        #self.linear = torch.nn.Linear(second_features, num_classes)

    def graph_embedding(self,x,edge_index, edge_weight):
        h = self.gat_layer1(x, edge_index)
        h = F.relu(h)
        h = self.gat_layer2(h, edge_index)
        return h

    def forward(self, data):
        #h = self.encoder(x, edge_index,edge_weight)
        #data_list = [Data(x=x_, edge_index=edge_index,edge_attr=edge_weight) for x_ in x]
        #batch = Batch.from_data_list(data_list)
        #
        h = None
        for time, snapshot in enumerate(data):
            if time ==0:
                h = self.graph_embedding(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda())[None,:,:]

            else:
                h = torch.cat((h, self.graph_embedding(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda())[None,:,:]),dim=0)
        print(h.shape)
        h,_ = self.recurrent_layer1(h)
        return F.softmax(h, dim=1)



class Robust_RGNN(torch.nn.Module):
    def __init__(self, args, device, num_features, time_length, num_classes):
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
        self.structural_attn1, self.structural_attn2, self.recurrent_layer1,self.temporal_attn, self.linear = self.build_model(num_features,num_classes)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
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

        #temporal_attn = torch.nn.Linear(in_features=self.temporal_layer_config[0], out_features=self.num_classes)

        temporal_attn = TemporalAttentionLayer3(input_dim=self.temporal_layer_config[0],
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual,type='node')
        linear = torch.nn.Linear(in_features=self.temporal_layer_config[0], out_features=self.num_classes)



        return structural_attn1, structural_attn2, recurrent_layer1, temporal_attn, linear

    def forward(self, graphs):
        #用于存储spatial-temporal输出，spatial为双重GAT, temporal为LSTM

        st_out = None
        coe1 = []
        coe2 = []
        hidden = None
        '''
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out2, c2 = self.structural_attn2(out1)
            out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
            st_out.append(self.temporal_attn(out3))
            coe1.append(c1)
            coe2.append(c2)
            #st_out.append(out3)
        '''
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out2, c2 = self.structural_attn2(out1)
            out2.x = out2.x.relu()
            out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
            #st_out.append(self.temporal_attn(out3))
            coe1.append(c1)
            coe2.append(c2)
            if st_out == None:
                st_out = out3.transpose(0,1)
            else:
                st_out = torch.cat((st_out,out3.transpose(0,1)), dim=1)
            #st_out:[T,N,F]
        temporal_out = self.temporal_attn(st_out)
        #return st_out, coe1, coe2
        return temporal_out.transpose(0,1), coe1, coe2
        #return self.linear(st_out.transpose(0, 1)), coe1, coe2



    def forward2(self, graphs):
        #用于存储spatial-temporal输出，spatial为双重GAT, temporal为LSTM
        st_out = []
        coe1 = []
        coe2 = []
        hidden = None
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out2, c2 = self.structural_attn2(out1)
            out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
            st_out.append(self.temporal_attn(out3).squeeze())
            coe1.append(c1)
            coe2.append(c2)
            #st_out.append(out3)
        #temporal_out = self.temporal_attn(st_out)
        return st_out, coe1, coe2


    def get_total_loss(self,graphs, degrees, cen_var):
        mu1 = 0.2 #0-1  0.1
        mu2 = 0.2 #0-1  0.2
        loss, coe1, coe2= self.get_loss(graphs)
        loss += mu1*self.get_spatial_loss(graphs, coe1, coe2, degrees)
        loss += mu2*self.get_temporal_loss(graphs, coe1, coe2, cen_var)
        return loss

    def get_loss(self, graphs):  #需要修改
        # run gnn
        final_emb, coe1, coe2 = self.forward(graphs)  # [N, T, F]
        self.graph_loss = 0
        for t in range(len(graphs)):
            y = graphs[t].y
            #y = y.numpy()
            y = torch.argmax(y, dim=1)
            #y = np.argmax(y, axis=1)
            #labels = torch.from_numpy(y).long().cuda()
            emb_t = final_emb[t]  # [N, F]
            y_hat = F.softmax(emb_t, dim=1)
            self.graph_loss += self.criterion(y_hat.cuda(), y)
        return self.graph_loss/len(graphs), coe1, coe2

    def get_spatial_loss(self, graphs, coe1, coe2, degrees):
        self.graph_spatial_loss = 0
        for t in range(len(graphs)):
            c1 = coe1[t]
            c2 = coe2[t]
            node_num = graphs[t].x.shape[0]
            #graph = to_networkx(graphs[t])
            #x = graphs[t].x
            edge_index = graphs[t].edge_index
            # x_origin = cp.deepcopy(x)
            #edge_weight = graphs[t].edge_attr.reshape(-1, 1)
            #node_num = x.shape[0]
            alph1 = scatter(c1, edge_index[0], dim=0, reduce="sum")
            alph1 = torch.norm(alph1, p=2, dim=1)
            alph2 = scatter(c2, edge_index[0], dim=0, reduce="sum")
            alph2 = torch.norm(alph2, p=2, dim=1)
            sim = torch.cosine_similarity(graphs[t].x[edge_index[0]],graphs[t].x[edge_index[1]])#+1
            #sim = F.pairwise_distance(graphs[t].x[edge_index[0]][:-node_num],graphs[t].x[edge_index[1][:-node_num]], p=2)
            sim1 = scatter(sim, edge_index[1], dim=0, reduce="mean")
            #self.graph_spatial_loss += torch.mean((alph1 + alph2) / 2 - degrees[t])
            self.graph_spatial_loss += torch.mean((alph1+alph2)/2-sim1/(degrees[t]+1))
        return self.graph_spatial_loss/len(graphs)

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
        return self.graph_temporal_loss/len(graphs)
        #structural层的输出










