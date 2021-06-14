import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from models.gnn_encoder import StarE_PyG_Encoder
from utils.utils_gcn import get_param, weight_init
from utils.nodepiece_tokenizer import NodePiece_Tokenizer
from torch_geometric.data import Data


class StarE_PyG_NC(StarE_PyG_Encoder):
    """
        A model that is supposed to work with pyg-like data objects and be inductive
    """
    def __init__(self, config: dict, tokenizer: NodePiece_Tokenizer = None, graph: Data = None):
        super(self.__class__, self).__init__(config, tokenizer=tokenizer, graph=graph)

        self.model_name = 'StarE_PyG'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.hidden_dim = config['STAREARGS']['GCN_DIM']
        self.num_classes = config['NUM_CLASSES']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        self.to_classes = nn.Linear(self.hidden_dim, self.num_classes)

    def reset_parameters(self):
        super(StarE_PyG_NC, self).reset_parameters()
        self.to_classes.apply(weight_init)

    def forward(self, graph, train_mask):
        '''
        :param graph: pyg data object
        :param train_mask: nodes for classification
        :return: class probabilities (logits)
        '''
        all_ent, rels = self.forward_base(graph, self.hidden_drop, self.feature_drop)
        nodes = torch.index_select(all_ent, 0, train_mask)

        nodes = self.hidden_drop2(nodes)
        probs = self.to_classes(nodes)
        return probs
