import torch.nn as nn
import torch
from utils.utils_gcn import get_param, weight_init
from utils.nodepiece_encoder import NodePieceEncoder


class MLP(nn.Module):
    """
    Vanilla MLP based only on node features
    """
    def __init__(self, initial_features, config):
        super().__init__()

        self.input_dim = initial_features.shape[1]
        self.hidden = config['STAREARGS']['GCN_DIM']
        self.num_layers = config['STAREARGS']['LAYERS']
        self.num_classes = config['NUM_CLASSES']
        self.dropout = config['STAREARGS']['GCN_DROP']
        self.use_features = config['USE_FEATURES']
        self.device = config['DEVICE']

        if not config['USE_FEATURES']:
            self.entity_embeddings = get_param((config['NUM_ENTITIES'], config['EMBEDDING_DIM']))
        else:
            self.node_features = torch.cat([
                torch.zeros((1, initial_features.shape[1]), device=self.device),
                torch.tensor(initial_features, dtype=torch.float, device=self.device)], dim=0)

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.ReLU(), nn.Dropout(self.dropout)),
            *[nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout)) for _ in range(self.num_layers-2)],
            nn.Sequential(nn.Linear(self.hidden, self.num_classes), nn.Dropout(self.dropout))
        ])

    def forward(self, train_mask):
        if not self.use_features:
            x = self.entity_embeddings
        else:
            x = self.node_features

        for layer in self.layers:
            x = layer(x)

        probs = torch.index_select(x, 0, train_mask)

        return probs


class MLP_PyG(nn.Module):
    """
    Vanilla MLP based only on node features - for PyG setup
    """
    def __init__(self, config, tokenizer, graph):
        super().__init__()

        #self.input_dim = config['FEATURE_DIM']
        self.hidden = config['STAREARGS']['GCN_DIM']
        self.num_layers = config['STAREARGS']['LAYERS']
        self.num_classes = config['NUM_CLASSES']
        self.dropout = config['STAREARGS']['GCN_DROP']
        self.use_features = config['USE_FEATURES']
        self.device = config['DEVICE']

        self.tokenizer = tokenizer

        if not config['USE_FEATURES']:
            self.input_dim = config['EMBEDDING_DIM']
            if self.tokenizer is None:
                self.entity_embeddings = get_param((config['NUM_ENTITIES'], self.input_dim))
            else:
                self.init_rel = nn.Embedding(config['NUM_RELATIONS'] * 2 + 1, self.input_dim)
                self.embedder = NodePieceEncoder(config, tokenizer, rel_embs=self.init_rel, graph=graph)
        else:
            self.input_dim = config['FEATURE_DIM']

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.ReLU(), nn.Dropout(self.dropout)),
            *[nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout)) for _ in range(self.num_layers-2)],
            nn.Sequential(nn.Linear(self.hidden, self.num_classes), nn.Dropout(self.dropout))
        ])

    def reset_parameters(self):
        for layer in self.layers:
            layer.apply(weight_init)
        if not self.use_features:
            if self.tokenizer is None:
                torch.nn.init.xavier_normal_(self.entity_embeddings.data)
            else:
                torch.nn.init.xavier_normal_(self.init_rel.weight.data)
                self.embedder.reset_parameters()

    def forward(self, graph, train_mask):

        if not self.use_features:
            if self.tokenizer is None:
                x = self.entity_embeddings
            else:
                x = self.embedder.get_all_representations()
        else:
            x = graph['x']

        for layer in self.layers:
            x = layer(x)

        probs = torch.index_select(x, 0, train_mask)

        return probs
