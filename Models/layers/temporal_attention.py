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
        # define weights
        self.temp_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp1_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp2_weights = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim, input_dim))
        self.drop = nn.Dropout(0.8)
        #self.position_embeddings = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim))
        #self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.type = type

        # ff
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
            #return outputs
            #return outputs+inputs  #No attack  DBLP
            #return self.lin(outputs+inputs) #under attack reddit
            return self.feedforward(outputs+inputs)  # if edge DBLP


    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs+inputs

    def xavier_init(self):
        #nn.init.xavier_uniform_(self.position_embeddings)
        #nn.init.xavier_uniform_(self.Q_embedding_weights)
        #nn.init.xavier_uniform_(self.K_embedding_weights)
        #nn.init.xavier_uniform_(self.V_embedding_weights)
        nn.init.xavier_uniform_(self.temp1_weights)
        nn.init.xavier_uniform_(self.temp2_weights)
        nn.init.xavier_uniform_(self.temp_weights)




class TemporalAttentionLayer2(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps, #已经修改为time_span
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer2, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
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
        time_steps = inputs.shape[1]
        position_inputs = torch.arange(0, time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        # [N,T,F]
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        #h is the number of heads
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)





class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)