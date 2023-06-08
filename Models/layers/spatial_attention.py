import torch
import torch.nn as nn
import torch.nn.functional as F
import copy as cp
from torch_geometric.utils import softmax
from torch_scatter import scatter
import torch_geometric
import copy
from Models.RGNN.models import ModelfreeGCN

class StructuralAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(StructuralAttentionLayer, self).__init__()
        self.input_dim = input_dim

        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()
        self.beta = 0.5

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.sim_weight_linear = torch.nn.Parameter(
            torch.Tensor(n_heads, self.input_dim, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.model_free = ModelfreeGCN(input_dim)

        #self.coefficient = None

        self.xavier_init()

    def forward(self, graph):
        graph = torch_geometric.data.data.Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y)
        #graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        #x_origin = cp.deepcopy(x)
        edge_weight = graph.edge_attr.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C)  # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]]  # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]
        # dropout
        #if self.training:
        coefficients = self.selective_sampling(coefficients, graph)  # 4912,8
        coefficients = self.attn_drop(coefficients)
        x = self.ffd_drop(x)

        #self.coefficient = coefficients #############################这个在不同时间会被替换
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]
        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            edge_set = list(set(edge_index[1].tolist()))
            out = out + self.lin_residual(graph.x)
        graph.x = out
        return graph, coefficients

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.sim_weight_linear)

    def selective_sampling(self, coefficients, graph): #选择性采样  sim
        coefficients = coefficients * torch.exp(self.beta * self.similarity(graph))
        coefficients = softmax(coefficients, graph.edge_index[1])
        return coefficients

    def similarity(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        #x_l = torch.matmul(self.model_free(x[edge_index[0]], edge_index, edge_attr).detach(), self.sim_weight_linear)
        #x_r = torch.matmul(self.model_free(x[edge_index[1]], edge_index, edge_attr).detach(), self.sim_weight_linear)
        x_l = torch.matmul(x[edge_index[0]], self.sim_weight_linear)
        x_r = torch.matmul(x[edge_index[1]], self.sim_weight_linear)
        cos = nn.CosineSimilarity(dim=2)
        return cos(x_l, x_r).transpose(0,1)