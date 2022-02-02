import torch
import numpy as np

from typing import Dict
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.utils_gcn import weight_init
from .gnn_layer import StarEConvLayer
from .lrga_model import LowRankAttention
from torch_geometric.data import Data

class StarE_PyG_Encoder(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 device: torch.device,
                 num_rel: int,
                 layer_config: dict,
                 num_layers: int = 2,
                 use_lrga: bool = False,
                 lrga_k: int = 50,
                 lrga_drop: float = 0.1,
                 hid_drop: float = 0.1,
                 drop1: float = 0.1,
                 triple_mode: bool = True,
                 residual: bool = False,
                 jk: bool = False,
                 ):
        super(StarE_PyG_Encoder, self).__init__()
        self.act = torch.relu  # was tanh before

        self.layer_config = layer_config

        self.emb_dim = emb_dim
        self.gcn_dim = emb_dim
        self.hid_drop = hid_drop
        self.drop1 = nn.Dropout(drop1)
        self.triple_mode = triple_mode
        self.num_rel = num_rel

        self.device = device
        self.residual = residual
        self.jk = jk

        self.num_layers = num_layers


        """
            LRGA params
        """
        self.use_lrga = use_lrga
        self.lrga_k = lrga_k
        self.lrga_drop = lrga_drop


        self.convs = nn.ModuleList()
        if self.use_lrga:
            self.attention = nn.ModuleList()
            self.dim_reduction = nn.ModuleList()

        # populating manually first and last layers, otherwise in a loop
        self.convs.append(StarEConvLayer(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, config=layer_config))
        if self.use_lrga:
            self.attention.append(LowRankAttention(self.lrga_k, self.emb_dim, self.lrga_drop))
            self.dim_reduction.append(nn.Sequential(nn.Linear(2 * self.lrga_k + self.gcn_dim + self.emb_dim, self.gcn_dim)))
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.gcn_dim) for _ in range(self.num_layers - 1)])

        for _ in range(self.num_layers - 1):
            self.convs.append(StarEConvLayer(self.gcn_dim, self.gcn_dim, self.num_rel, act=self.act, config=layer_config))
            if self.use_lrga:
                self.attention.append(LowRankAttention(self.lrga_k, self.gcn_dim, self.lrga_drop))
                self.dim_reduction.append(nn.Sequential(nn.Linear(2 * (self.lrga_k + self.gcn_dim), self.gcn_dim)))

        # self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))
        if self.jk:
            self.linear = nn.Sequential(
                nn.Linear(self.gcn_dim * self.num_layers, self.gcn_dim),
                nn.ReLU(),
                nn.Linear(self.gcn_dim, self.gcn_dim)
            )

    def reset_parameters(self):

        #torch.nn.init.constant_(self.bias.data, 0)
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_lrga:
            for att in self.attention:
                att.apply(weight_init)
            for dim_r in self.dim_reduction:
                dim_r.apply(weight_init)
            for bnorm in self.bns:
                bnorm.reset_parameters()

        if self.jk:
            for module in self.linear:
                if module is self:
                    continue
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()


    def forward_base(self, graph):

        x, edge_index, edge_type, r = graph['x'], graph['edge_index'], graph['edge_type'], graph['rels']

        # Add reverse stuff
        # reverse_index = torch.zeros_like(edge_index)
        # reverse_index[1, :] = edge_index[0, :]
        # reverse_index[0, :] = edge_index[1, :]
        # rev_edge_type = edge_type + self.num_rel
        #
        # edge_index = torch.cat([edge_index, reverse_index], dim=1)
        # edge_type = torch.cat([edge_type, rev_edge_type], dim=0)

        if not self.triple_mode:
            raise NotImplementedError

        if self.jk:
            outputs = []

        for i, conv in enumerate(self.convs[:-1]):
            x_local, r = conv(x=x, edge_index=edge_index, edge_type=edge_type, rel_embed=r, quals=None)
            x_local = self.drop1(x_local)
            if self.use_lrga:
                x_global = self.attention[i](x)
                x = self.dim_reduction[i](torch.cat((x_global, x_local, x), dim=1))
                x = F.relu(x)
                x = self.bns[i](x)
            else:
                if self.residual:
                    x = x + x_local
                else:
                    x = x_local
            if self.jk:
                outputs.append(x)

        # last layer
        x_local, r = self.convs[-1](x=x, edge_index=edge_index, edge_type=edge_type, rel_embed=r, quals=None)
        x_local = self.drop1(x_local)
        if self.use_lrga:
            x_global = self.attention[-1](x)
            x = self.dim_reduction[-1](torch.cat((x_global, x_local, x), dim=1))
        else:
            if self.residual:
                x = x + x_local
            else:
                x = x_local

        if self.jk:
            outputs.append(x)
            x = self.linear(torch.cat(outputs, dim=-1))


        return x, r