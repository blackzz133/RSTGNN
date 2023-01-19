import torch
import torch.nn as nn
import torch.nn.functional as F
import copy as cp

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy
class TemporalAttentionLayer3(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps, #已经修改为time_span
                 attn_drop,
                 residual,type):
        super(TemporalAttentionLayer3, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        self.temp_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp1_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp2_weights = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim, input_dim))
        self.drop = nn.Dropout(0.8)
        self.type = type
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        #N节点，T时间，F特征
        # 1: Add position embeddings to input
        # [N,T]
        #assert inputs.shape[1] >= self.num_time_steps
        #H_temp1 = []
        #H_temp2 = []
        time = inputs.shape[1] #(N.T,D)

        #inputs = inputs.transpose(0,1) #(N.T,D)
        H_temp1 = torch.matmul(inputs, self.temp1_weights)
        H_temp2 = inputs - inputs
        #H_temp2 = H_temp2 - H_temp2

        for t in range(0,time):
            for t2 in range(max(0,t-self.num_time_steps+1),t+1):
                H_temp2[:, t, :] += torch.matmul(inputs[:,t2,:],self.temp2_weights[t2])
            H_temp2[:, t, :] /= min(t+1, self.num_time_steps)
        alph = torch.bmm(H_temp1, H_temp2.transpose(1,2))
        H_temp = torch.matmul(inputs,self.temp_weights)
        outputs = torch.matmul(alph, H_temp)
        if self.training:
            outputs = self.drop(outputs)
        if self.type=='node':
            return outputs  # if node
        if self.type =='edge':
            return self.feedforward(outputs) + inputs  # if edge


    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.temp1_weights)
        nn.init.xavier_uniform_(self.temp2_weights)
        nn.init.xavier_uniform_(self.temp_weights)



