import torch
import numpy as np

from typing import Dict
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.utils_gcn import get_param, weight_init
from .gnn_layer import StarEConvLayer
from .lrga_model import LowRankAttention
from utils.nodepiece_tokenizer import NodePiece_Tokenizer
from utils.nodepiece_encoder import NodePieceEncoder
from torch_geometric.data import Data


class StarE_PyG_Encoder(nn.Module):
    def __init__(self, config: dict, tokenizer: NodePiece_Tokenizer = None, graph: Data = None):
        super(StarE_PyG_Encoder, self).__init__()
        self.act = torch.relu  # was tanh before
        self.model_nm = config['MODEL_NAME']
        self.config = config

        self.emb_dim = config['EMBEDDING_DIM']
        self.num_rel = config['NUM_RELATIONS']
        self.num_ent = config['NUM_ENTITIES']
        self.gcn_dim = config['STAREARGS']['GCN_DIM']
        self.hid_drop = config['STAREARGS']['HID_DROP']
        # self.bias = config['STAREARGS']['BIAS']
        self.triple_mode = config['STATEMENT_LEN'] == 3
        self.qual_mode = config['STAREARGS']['QUAL_REPR']

        self.device = config['DEVICE']

        self.num_layers = config['STAREARGS']['LAYERS']

        # self.gcn_dim = self.emb_dim if self.n_layer == 1 else self.gcn_dim

        """
            LRGA params
        """
        self.use_lrga = config['STAREARGS']['LRGA']
        self.lrga_k = config['STAREARGS']['LRGA_K']
        self.lrga_drop = config['STAREARGS']['LRGA_DROP']

        # if self.model_nm.endswith('transe'):
        #     self.init_rel = get_param((self.num_rel, self.emb_dim))
        # elif config['STAREARGS']['OPN'] == 'rotate' or config['STAREARGS']['QUAL_OPN'] == 'rotate':
        #     phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
        #     self.init_rel = nn.Parameter(torch.cat([
        #         torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
        #         torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
        #     ], dim=0))
        # else:
        #     self.init_rel = get_param((self.num_rel * 2, self.emb_dim))
        self.init_rel = nn.Embedding(self.num_rel * 2 + 1, self.emb_dim, padding_idx=self.num_rel * 2)

        self.init_rel.to(self.device)

        self.tokenizer = tokenizer
        if not config['USE_FEATURES']:
            if self.tokenizer is None:
                self.entity_embeddings = get_param((self.num_ent, self.emb_dim))
            else:
                self.embedder = NodePieceEncoder(config, tokenizer, rel_embs=self.init_rel, graph=graph)


        self.feature_reduction = nn.Linear(config['FEATURE_DIM'], self.emb_dim)

        self.convs = nn.ModuleList()
        if self.use_lrga:
            self.attention = nn.ModuleList()
            self.dim_reduction = nn.ModuleList()

        # populating manually first and last layers, otherwise in a loop
        self.convs.append(StarEConvLayer(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, config=config))
        if self.use_lrga:
            self.attention.append(LowRankAttention(self.lrga_k, self.emb_dim, self.lrga_drop))
            self.dim_reduction.append(nn.Sequential(nn.Linear(2 * self.lrga_k + self.gcn_dim + self.emb_dim, self.gcn_dim)))
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.gcn_dim) for _ in range(self.num_layers - 1)])

        for _ in range(self.num_layers - 1):
            self.convs.append(StarEConvLayer(self.gcn_dim, self.gcn_dim, self.num_rel, act=self.act, config=config))
            if self.use_lrga:
                self.attention.append(LowRankAttention(self.lrga_k, self.gcn_dim, self.lrga_drop))
                self.dim_reduction.append(nn.Sequential(nn.Linear(2 * (self.lrga_k + self.gcn_dim), self.gcn_dim)))

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def reset_parameters(self):
        if self.config['STAREARGS']['OPN'] == 'rotate' or self.config['STAREARGS']['QUAL_OPN'] == 'rotate':
            # phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
            # self.init_rel = nn.Parameter(torch.cat([
            #     torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
            #     torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            # ], dim=0))
            phases = 2 * np.pi * torch.rand(self.num_rel * 2, self.emb_dim // 2, device=self.device)
            relations = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1).detach()
            assert torch.allclose(torch.norm(relations, p=2, dim=-1), phases.new_ones(size=(1, 1)))
            self.init_rel.weight.data[:-1] = relations.view(self.num_rel * 2, self.emb_dim)
            self.init_rel.weight.data[-1] = torch.zeros(self.emb_dim)
        else:
            torch.nn.init.xavier_normal_(self.init_rel.weight.data)
        self.feature_reduction.apply(weight_init)
        torch.nn.init.constant_(self.bias.data, 0)
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_lrga:
            for att in self.attention:
                att.apply(weight_init)
            for dim_r in self.dim_reduction:
                dim_r.apply(weight_init)
            for bnorm in self.bns:
                bnorm.reset_parameters()
        if not self.config['USE_FEATURES']:
            if self.tokenizer is None:
                torch.nn.init.xavier_normal_(self.entity_embeddings.data)
            else:
                self.embedder.reset_parameters()


    def post_parameter_update(self):
        rel = self.init_rel.weight.data.view(self.num_rel * 2 + 1, self.emb_dim // 2, 2)
        rel = F.normalize(rel, p=2, dim=-1)
        self.init_rel.weight.data = rel.view(self.num_rel * 2 + 1, self.emb_dim)


    def forward_base(self, graph, drop1, drop2):

        x, edge_index, edge_type, quals = graph['x'], graph['edge_index'], graph['edge_type'], graph['quals']

        # Add reverse stuff
        reverse_index = torch.zeros_like(edge_index)
        reverse_index[1, :] = edge_index[0, :]
        reverse_index[0, :] = edge_index[1, :]
        rev_edge_type = edge_type + self.num_rel

        edge_index = torch.cat([edge_index, reverse_index], dim=1)
        edge_type = torch.cat([edge_type, rev_edge_type], dim=0)

        if not self.triple_mode:
            quals = torch.cat([quals, quals], dim=1)

        r = self.init_rel.weight if not self.model_nm.endswith('transe') \
            else torch.cat([self.init_rel.weight, -self.init_rel.weight], dim=0)

        if self.config['USE_FEATURES']:
            x = self.feature_reduction(x)   # TODO find a way to perform attention without dim reduction beforehand
        else:
            if self.tokenizer is None:
                x = self.entity_embeddings
            else:
                x = self.embedder.get_all_representations()


        for i, conv in enumerate(self.convs[:-1]):
            x_local, r = conv(x=x, edge_index=edge_index, edge_type=edge_type, rel_embed=r, quals=quals)
            x_local = drop1(x_local)
            if self.use_lrga:
                x_global = self.attention[i](x)
                x = self.dim_reduction[i](torch.cat((x_global, x_local, x), dim=1))
                x = F.relu(x)
                x = self.bns[i](x)
            else:
                x = x_local

        # last layer
        x_local, r = self.convs[-1](x=x, edge_index=edge_index, edge_type=edge_type, rel_embed=r, quals=quals)
        x_local = drop2(x_local)
        if self.use_lrga:
            x_global = self.attention[-1](x)
            x = self.dim_reduction[-1](torch.cat((x_global, x_local, x), dim=1))
        else:
            x = x_local

        return x, r