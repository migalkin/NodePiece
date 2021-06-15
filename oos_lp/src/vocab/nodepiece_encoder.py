import torch
import numpy as np
import random
import torch.nn.functional as F

from torch import nn
from vocab.nodepiece_tokenizer import NodePiece_Tokenizer
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from tqdm import tqdm
from collections import defaultdict
from typing import Optional
from common.dataset import Dataset
from argparse import Namespace


class NodePieceEncoder(nn.Module):
    """
    Pretty much the same thing as in link prediction / nodepiece_rotate
    """
    def __init__(self, config: Namespace, tokenizer: NodePiece_Tokenizer, rel_embs: nn.Embedding, graph: Dataset, device: torch.device):

        super(NodePieceEncoder, self).__init__()

        self.tokenizer = tokenizer
        self.pooler = config.pooler
        self.policy = "sum"
        self.nearest = config.nearest_ancs
        self.sample_rels = config.sample_rels
        self.random_hashes = config.random_hashes

        self.subbatch = config.subbatch
        self.embedding_dim = config.emb_dim
        self.real_embedding_dim = self.embedding_dim // 2

        self.max_seq_len = config.max_path_len
        self.sample_paths = config.sample_size
        self.use_distances = config.anc_dist
        self.hid_dim = config.t_hidden
        self.drop_prob = config.t_drop
        self.num_heads = config.t_heads
        self.num_layers = config.t_layers
        self.num_entities = graph.init_num_ent
        self.num_relations = len(graph.rel2id)
        self.device = device

        self.no_anc = config.no_anc


        if self.pooler == "cat":
            self.set_enc = nn.Sequential(
                nn.Linear(self.embedding_dim * (self.sample_paths + self.sample_rels), self.embedding_dim * 2), nn.Dropout(self.drop_prob),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            ) if not self.ablate_anchors else nn.Sequential(
                nn.Linear(self.embedding_dim * self.sample_rels, self.embedding_dim * 2), nn.Dropout(self.drop_prob), nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            )
        elif self.pooler == "trf":
            encoder_layer = TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hid_dim,
                dropout=self.drop_prob,
            )
            self.set_enc = TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)



        self.tokenizer.token2id[self.tokenizer.NOTHING_TOKEN] = len(tokenizer.token2id)
        self.anchor_embeddings = nn.Embedding(len(tokenizer.token2id), embedding_dim=self.embedding_dim,
                                              padding_idx=self.tokenizer.token2id[tokenizer.PADDING_TOKEN])
        self.relation_embeddings = rel_embs
        self.dist_emb = nn.Embedding(self.max_seq_len, embedding_dim=self.embedding_dim)

        if self.random_hashes == 0:
            if not self.nearest:
                # subsample paths, need to align them with distances
                sampled_paths = {
                    entity: random.sample(paths, k=min(self.sample_paths, len(paths)))
                    for entity, paths in self.tokenizer.vocab.items()
                }
            elif self.nearest:
                # sort paths by length first and take K of them
                prev_max_len = max(len(path) for k, v in self.tokenizer.vocab.items() for path in v)
                sampled_paths = {
                    entity: sorted(paths, key=lambda x: len(x))[:min(self.sample_paths, len(paths))]
                    for entity, paths in self.tokenizer.vocab.items()
                }
                self.max_seq_len = max(len(path) for k, v in sampled_paths.items() for path in v)
                print(
                    f"Changed max seq len from {prev_max_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

            hashes = [
                [self.tokenizer.token2id[path[0]] for path in paths] + [
                    self.tokenizer.token2id[tokenizer.PADDING_TOKEN]] * (self.sample_paths - len(paths))
                for entity, paths in sampled_paths.items()
            ]
            distances = [
                [len(path) - 1 for path in paths] + [0] * (self.sample_paths - len(paths))
                for entity, paths in sampled_paths.items()
            ]


            self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
            self.distances = torch.tensor(distances, dtype=torch.long, device=self.device)


        else:
            # in this case, we bypass distances and won't use relations in the encoder
            self.anchor_embeddings = nn.Embedding(self.random_hashes, embedding_dim=self.embedding_dim)
            hashes = [
                random.sample(list(range(self.random_hashes)), self.sample_paths)
                for i in range(self.num_entities)
            ]

            self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
            self.distances = torch.zeros((self.num_entities, self.sample_paths), dtype=torch.long,
                                         device=self.device)



        if self.sample_rels > 0:
            pad_idx = self.num_relations
            e2r = defaultdict(set)
            for row in graph.train_data:
                e2r[int(row[0])].add(int(row[1]))

            len_stats = [len(v) for k, v in e2r.items()]
            print(
                f"Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(sorted(len_stats), 66)}, max: {max(len_stats)} ")
            unique_1hop_relations = [
                random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (
                            self.sample_rels - min(len(e2r[i]), self.sample_rels))
                for i in range(self.num_entities)
            ]
            self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.device)

    def reset_parameters(self):

        if self.pooler != "avg":
            for module in self.set_enc.modules():
                if module is self:
                    continue
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            if self.pooler == "mlp":
                for module in self.set_dec.modules():
                    if module is self:
                        continue
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()


        torch.nn.init.xavier_uniform_(self.anchor_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.dist_emb.weight)

        if self.random_hashes == 0:
            with torch.no_grad():
                self.anchor_embeddings.weight[self.tokenizer.token2id[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.embedding_dim)
                self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)


    def pool_anchors(self, anc_embs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        input shape: (bs, num_anchors, emb_dim)
        output shape: (bs, emb_dim)
        """

        if self.pooler == "cat":
            anc_embs = anc_embs.view(anc_embs.shape[0], -1)
            pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
        elif self.pooler == "trf":
            pooled = self.set_enc(anc_embs.transpose(1, 0))  # output shape: (seq_len, bs, dim)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)

        return pooled

    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:

        hashes, dists = self.hashes[entities], self.distances[entities]

        anc_embs = self.anchor_embeddings(hashes)
        mask = None

        if self.use_distances:
            dist_embs = self.dist_emb(dists)
            anc_embs += dist_embs

        if self.no_anc:
            anc_embs = torch.tensor([], device=self.device)

        if self.sample_rels > 0:
            rels = self.unique_1hop_relations[entities]  # (bs, rel_sample_size)
            rels = self.relation_embeddings(rels)   # (bs, rel_sample_size, dim)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)  # (bs, dim)

        return anc_embs


    def get_all_representations(self):

        temp_embs = torch.zeros((len(self.hashes), self.embedding_dim), dtype=torch.float, device=self.device)

        vocab_keys = list(range(len(self.hashes)))
        for i in tqdm(range(0, len(self.hashes), self.subbatch)):
            entities = torch.tensor(vocab_keys[i: i+self.subbatch], dtype=torch.long, device=self.device)
            embs = self.encode_by_index(entities)
            temp_embs[i: i + self.subbatch, :] = embs

        return temp_embs


    def encode_by_hash(self, hashes: torch.LongTensor, distances: torch.LongTensor, rels: np.ndarray):
        anc_embs = self.anchor_embeddings(hashes.unsqueeze(0))
        mask = None
        pad_idx = self.num_relations

        if self.use_distances:
            dist_embs = self.dist_emb(distances)
            anc_embs += dist_embs

        if self.sample_rels > 0:
            if len(rels) < self.sample_rels:
                rels = torch.cat(
                    [torch.tensor(rels, dtype=torch.long, device=self.device),
                     torch.tensor([pad_idx]*(self.sample_rels-len(rels)), dtype=torch.long, device=self.device)
                     ])
            elif len(rels) > self.sample_rels:
                rels = torch.tensor(np.random.permutation(rels)[:self.sample_rels], dtype=torch.long, device=self.device)
            else:
                rels = torch.tensor(rels, dtype=torch.long, device=self.device)

            rels = self.relation_embeddings(rels.unsqueeze(0))   # (bs, rel_sample_size, dim)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)  # (bs, dim)

        return anc_embs