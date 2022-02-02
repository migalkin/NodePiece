#  taken from https://github.com/omri1348/LRGA/blob/master/ogb/examples/linkproppred/collab/graph_global_attention_layer.py

import torch
import torch.nn as nn


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    tmp_ones = torch.ones((V_T.shape[1], 1))
    if torch.cuda.is_available():
        tmp_ones = tmp_ones.to(torch.device('cuda'))
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1/norm_factor


def weight_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant(layer.bias.data, 0)
    return


class LowRankAttention(nn.Module):
    def __init__(self, k, d, dropout):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d, 4*k), nn.ReLU())
        self.activation = nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        tmp = self.w(x)  # B x D
        U = tmp[:, :self.k]  # B x K
        V = tmp[:, self.k:2*self.k]  # B x K
        Z = tmp[:, 2*self.k:3*self.k]  # B x K
        T = tmp[:, 3*self.k:]  # B x K
        V_T = torch.t(V)  # K x B
        # normalization
        D = joint_normalize2(U, V_T)  # scalar
        res = torch.mm(U, torch.mm(V_T, Z))  # (B x K) mm (K x K) -> B x K
        res = torch.cat((res*D, T), dim=1)  # B x 2K
        return self.dropout(res)
