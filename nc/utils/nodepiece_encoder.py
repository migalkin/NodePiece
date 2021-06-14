import torch
import numpy as np
import random
import torch.nn.functional as F

from torch import nn
from utils.nodepiece_tokenizer import NodePiece_Tokenizer
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from tqdm import tqdm
from collections import defaultdict
from typing import Optional
from torch_geometric.data import Data


class NodePieceEncoder(nn.Module):
    def __init__(self, config: dict, tokenizer: NodePiece_Tokenizer, rel_embs: nn.Embedding, graph: Data):

        super(NodePieceEncoder, self).__init__()

        self.tokenizer = tokenizer
        self.pooler = config['POOLER']
        self.policy = "sum"
        self.use_rels = False
        self.nearest = config['NEAREST']
        self.use_neighbor_rels = False
        self.sample_rels = config['SAMPLE_RELS']
        self.graph = graph

        if not self.use_rels:
            self.policy = "sum"

        self.random_hashes = config['RANDOM_HASHES']

        self.subbatch = config['SUBBATCH']
        self.embedding_dim = config['EMBEDDING_DIM']
        self.real_embedding_dim = self.embedding_dim // 2

        self.max_seq_len = config['MAX_PATH_LEN']
        self.sample_paths = config['MAX_PATHS']
        self.use_distances = config['USE_DISTANCES']
        self.hid_dim = config['T_HIDDEN']
        self.drop_prob = config['T_DROP']
        self.num_heads = config['T_HEADS']
        self.num_layers = config['T_LAYERS']
        self.num_entities = config['NUM_ENTITIES']
        self.num_relations = config['NUM_RELATIONS']
        self.device = config['DEVICE']
        self.no_anc = config['NO_ANC']


        if self.pooler == "mlp":
            self.set_enc = nn.Sequential(
                nn.Linear(self.embedding_dim if self.policy != "cat" else 2 * self.embedding_dim, self.hid_dim), nn.Dropout(self.drop_prob),
                nn.ReLU(),
                nn.Linear(self.hid_dim, self.hid_dim), nn.Dropout(self.drop_prob), nn.ReLU(),
                nn.Linear(self.hid_dim, self.hid_dim),
            )
            self.set_dec = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim), nn.Dropout(self.drop_prob), nn.ReLU(),
                nn.Linear(self.hid_dim, self.hid_dim), nn.Dropout(self.drop_prob), nn.ReLU(),
                nn.Linear(self.hid_dim, self.embedding_dim)
            )
        elif self.pooler == "cat":
            self.set_enc = nn.Sequential(
                nn.Linear(self.embedding_dim * (self.sample_paths + self.sample_rels), self.embedding_dim * 2), nn.Dropout(self.drop_prob),
                nn.ReLU(),
                # nn.Linear(embedding_dim * 4, embedding_dim * 2), nn.Dropout(drop_prob), nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            ) if not self.no_anc else nn.Sequential(
                nn.Linear(self.embedding_dim * self.sample_rels, self.embedding_dim * 2), nn.Dropout(self.drop_prob),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim))
        elif self.pooler == "trf":
            encoder_layer = TransformerEncoderLayer(
                d_model=self.embedding_dim if self.policy != "cat" else 2 * self.embedding_dim,
                nhead=self.num_heads if self.policy != "cat" else 2 * self.num_heads,
                dim_feedforward=self.hid_dim,
                dropout=self.drop_prob,
            )
            self.set_enc = TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)
            if self.policy == "cat":
                self.linear = nn.Linear(2 * self.embedding_dim, self.embedding_dim)




        self.rel_gnn = False
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
            # _PRIMES = [
            #     31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223
            # ]
            # self.num_buckets = self.random_hashes

            # self.anchor_embeddings = nn.Embedding(self.num_buckets * self.num_hashes, embedding_dim=embedding_dim // self.num_hashes)

            # self.hash_projector = nn.Sequential(
            #     nn.Linear(self.embedding_dim, self.embedding_dim),
            #     nn.ReLU(),
            #     nn.Linear(self.embedding_dim, self.embedding_dim)
            # )

            # primes = _PRIMES[:self.num_hashes]
            # hashes = [
            #     [(((i+1) * prime) % self.num_buckets) + k*self.num_buckets for k, prime in enumerate(primes)]
            #     for i in range(triples.num_entities)
            # ]

            self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
            self.distances = torch.zeros((self.num_entities, self.sample_paths), dtype=torch.long,
                                         device=self.device)



        if self.use_neighbor_rels:
            # create a feature matrix where rows are used relations in a 1-hop neighbourhood around each node
            unique_sp = self.triples_factory.mapped_triples[:, [0, 1]].unique(dim=0, return_counts=False)
            self.relation_features = torch.zeros((self.num_entities, self.num_relations * 2),
                                                 dtype=torch.float, requires_grad=False,
                                                 device=self.device)  # features matrix
            self.relation_features[unique_sp[:, 0], unique_sp[:, 1]] = 1.0  # counts.float().to(self.device)
            # self.relation_features = torch.nn.functional.normalize(self.relation_features, p=1, dim=1)
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim + self.num_relations, self.hid_dim),
                nn.ReLU(),
                nn.Linear(self.hid_dim, self.embedding_dim)
            )

        if self.sample_rels > 0:
            pad_idx = self.num_relations * 2
            e2r = defaultdict(set)
            edge_index = self.graph.edge_index
            edge_type = self.graph.edge_type
            for i, src_node in enumerate(edge_index[0]):
                e2r[src_node.item()].add(edge_type[i].item())
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


        if self.use_neighbor_rels:
            for module in self.projection.modules():
                if module is self:
                    continue
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        # if self.random_hashes != 0:
        #     for module in self.hash_projector.modules():
        #         if module is self:
        #             continue
        #         if hasattr(module, "reset_parameters"):
        #             module.reset_parameters()

        torch.nn.init.xavier_uniform_(self.anchor_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.dist_emb.weight)
        if self.use_rels == "joint":
            torch.nn.init.xavier_uniform_(self.node_types.weight)

        if self.random_hashes == 0:
            with torch.no_grad():
                self.anchor_embeddings.weight[self.tokenizer.token2id[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.embedding_dim)
                self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)
            # if self.use_rels == "trf":
            #     self.rel_pos.weight[0] = torch.zeros(self.embedding_dim)

        # phases randomly between 0 and 2 pi
        # phases = 2 * np.pi * torch.rand(self.num_relations, self.real_embedding_dim, device=self.device)
        # relations = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1).detach()
        # assert torch.allclose(torch.norm(relations, p=2, dim=-1), phases.new_ones(size=(1, 1)))
        # self.relation_embeddings.weight.data[:-1] = relations.view(self.num_relations, self.embedding_dim)
        # self.relation_embeddings.weight.data[-1] = torch.zeros(self.embedding_dim)

    def pool_anchors(self, anc_embs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        input shape: (bs, num_anchors, emb_dim)
        output shape: (bs, emb_dim)
        """

        if self.pooler == "set":
            pooled = self.set_enc(anc_embs)
        elif self.pooler == "cat":
            anc_embs = anc_embs.view(anc_embs.shape[0], -1)
            pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
        elif self.pooler == "trf" or self.pooler == "moe":
            if self.use_rels != "joint":
                pooled = self.set_enc(anc_embs.transpose(1, 0))  # output shape: (seq_len, bs, dim)
            else:
                pooled = self.set_enc(anc_embs.transpose(1, 0), src_key_padding_mask=mask)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)
            if self.policy == "cat":
                pooled = self.linear(pooled)
        elif self.pooler == "perc":
            pooled = self.set_enc(anc_embs)
        elif self.pooler == "mlp":
            pooled = self.set_dec(self.set_enc(anc_embs).mean(-2))
        else:
            pooled = anc_embs.mean(dim=1)

        return pooled

    def encode_rels(self, rel_hashes: torch.LongTensor, weights: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        # input: (bs, num_anchors, max_seq_len)
        bs, num_paths, seq_len = rel_hashes.shape
        rel_hashes = rel_hashes.view(bs * num_paths, seq_len)
        if weights is not None:
            weights = weights.view(bs * num_paths, seq_len)
        pad_mask = rel_hashes != self.triples_factory.num_relations
        rel_hashes = self.relation_embeddings(rel_hashes)  # (bs*num_paths, seq_len, hid_dim)
        if self.use_rels == "lstm" or self.use_rels == "gru":
            rel_hashes, _ = self.rel_enc(rel_hashes)  # (bs, seq_len, hid_dim)
            rel_hashes = self.rel_proj(rel_hashes[:, -1, :])  # (bs, emb_dim)
        elif self.use_rels == "mlp":
            accumulator = torch.zeros((rel_hashes.shape[0], rel_hashes.shape[-1]), dtype=torch.float, device=rel_hashes.device)
            enc, dec = self.rel_enc[0], self.rel_enc[1]
            for i in range(seq_len-1):
                pair = rel_hashes[:, i:i+2].view(-1, 2 * self.embedding_dim)  # (bs*num_anc, 2 * 100)
                pair = enc(pair)
                accumulator += pair
            rel_hashes = dec(accumulator)
        elif self.use_rels == "avg":
            if weights is None:
                rel_hashes = (rel_hashes * pad_mask.float().unsqueeze(-1)).sum(-2) / pad_mask.float().sum(-1).clamp_min(1.0).unsqueeze(-1)
            else:
                rel_hashes = (rel_hashes * weights.unsqueeze(-1)).sum(-2) / pad_mask.float().sum(-1).clamp_min(1.0).unsqueeze(-1)
        elif self.use_rels == "avg+":
            pos = torch.arange(seq_len, dtype=torch.long, device=rel_hashes.device).repeat(bs * num_paths, 1)
            pos = self.rel_pos(pos)
            rel_hashes = torch.cat([rel_hashes, pos], dim=-1)
            rel_hashes = self.rel_enc(rel_hashes)
            rel_hashes = (rel_hashes * pad_mask.float().unsqueeze(-1)).sum(-2) / pad_mask.float().sum(-1).clamp_min(1.0).unsqueeze(-1)
        elif self.use_rels == "trf":
            temp = torch.zeros((rel_hashes.shape[1], rel_hashes.shape[0], rel_hashes.shape[2]), dtype=torch.float, device=rel_hashes.device)
            positions = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
            positions = self.rel_pos(positions)
            nnz_paths = rel_hashes[pad_mask.sum(1) > 0]
            nnz_paths += positions
            nnz_paths = self.rel_enc(nnz_paths.transpose(1, 0), src_key_padding_mask=~pad_mask[pad_mask.sum(1) > 0])  # (seq_len, bs, dim)
            nnz_paths[torch.isnan(nnz_paths)] = 1.0  # for numerical stability of empty paths with NOTHING tokens
            nnz_paths[torch.isinf(nnz_paths)] = 1.0  # for numerical stability of empty paths with NOTHING tokens
            temp[:, pad_mask.sum(1) > 0, :] = nnz_paths
            rel_hashes = temp
            rel_hashes = (rel_hashes * pad_mask.t().float().unsqueeze(-1)).sum(0) / pad_mask.float().sum(1).clamp_min(1.0).unsqueeze(-1)
            #rel_hashes = rel_hashes.mean(0)
        elif self.use_rels == "int":
            # replace padding 0's with 1's to prevent nans in the rotate computation
            rel_hashes[~pad_mask] = 1.0
            start = rel_hashes[:, 0, :]
            for i in range(1, seq_len):
                target = rel_hashes[:, i, :]
                interaction = self.pairwise_interaction_function(start.view(-1, self.real_embedding_dim, 2), target.view(-1, self.real_embedding_dim, 2))
                start = interaction
            rel_hashes = interaction


        return rel_hashes.view(bs, num_paths, self.embedding_dim)


    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:

        hashes, dists = self.hashes[entities], self.distances[entities]

        anc_embs = self.anchor_embeddings(hashes)
        mask = None

        if self.use_distances:
            dist_embs = self.dist_emb(dists)
            anc_embs += dist_embs

        if self.no_anc:
            anc_embs = torch.tensor([], device=self.device)

        if self.use_rels:
            rel_hashes = self.rel_hash[entities]  # (bs, num_relations)
            path_weights = self.path_weights[entities] if self.use_mc else None
            # if self.rel_gnn:
            #     self.relation_embeddings.weight.data = self.gnn_encoder(self.relation_embeddings.weight, self.edge_index)
            if self.use_rels != "joint":
                path_embs = self.encode_rels(rel_hashes, path_weights)
                if self.policy != "cat":
                    anc_embs += path_embs
                else:
                    anc_embs = torch.cat([anc_embs, path_embs], dim=-1)
            else:
                path_embs = self.relation_embeddings(rel_hashes)
                anc_embs = torch.cat([anc_embs, path_embs], dim=1)
                node_types = torch.cat([
                    torch.zeros_like(hashes, dtype=torch.long, device=self.device),
                    torch.ones_like(rel_hashes, dtype=torch.long, device=self.device)
                ], dim=1)
                node_type_embs = self.node_types(node_types)
                anc_embs += node_type_embs
                mask = rel_hashes == self.num_relations
                mask = torch.cat([
                    torch.zeros_like(hashes, dtype=torch.bool, device=self.device),
                    mask.to(self.device)
                ], dim=1).to(self.device)

        #set_input[mask] = anc_embs.view(-1, 1)
        if self.sample_rels > 0:
            rels = self.unique_1hop_relations[entities]  # (bs, rel_sample_size)
            rels = self.relation_embeddings(rels)   # (bs, rel_sample_size, dim)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)  # (bs, dim)

        if self.use_neighbor_rels:
            rels_one_hot = self.relation_features[entities]
            anc_embs = torch.cat([anc_embs, rels_one_hot], dim=-1)  # (dim + num_relations)
            anc_embs = self.projection(anc_embs)    # (bs, dim)
        return anc_embs


    def get_all_representations(self):

        if self.subbatch != 0:
            temp_embs = torch.zeros((len(self.hashes), self.embedding_dim), dtype=torch.float, device=self.device)

            vocab_keys = list(range(len(self.hashes)))
            for i in tqdm(range(0, len(self.hashes), self.subbatch)):
                entities = torch.tensor(vocab_keys[i: i+self.subbatch], dtype=torch.long, device=self.device)
                embs = self.encode_by_index(entities)
                temp_embs[i: i + self.subbatch, :] = embs

            return temp_embs

        else:
            anc_embs = self.anchor_embeddings(self.hashes)
            mask = None
            if self.use_distances:
                dist_embs = self.dist_emb(self.distances)
                anc_embs += dist_embs
            if self.use_rels:

                if self.use_rels != "joint":
                    path_embs = self.encode_rels(self.rel_hash)
                    if self.policy != "cat":
                        anc_embs += path_embs
                    else:
                        anc_embs = torch.cat([anc_embs, path_embs], dim=-1)
                else:
                    path_embs = self.relation_embeddings(self.rel_hash)
                    anc_embs = torch.cat([anc_embs, path_embs], dim=1)
                    node_types = torch.cat([
                        torch.zeros_like(self.hashes, dtype=torch.long, device=self.device),
                        torch.ones_like(self.rel_hash, dtype=torch.long, device=self.device)
                    ], dim=1)
                    node_type_embs = self.node_types(node_types)
                    anc_embs += node_type_embs
                    mask = self.rel_hash == self.num_relations
                    mask = torch.cat([
                        torch.zeros_like(self.hashes, dtype=torch.bool, device=self.device),
                        mask.to(self.device)
                    ], dim=1).to(self.device)


            res = self.pool_anchors(anc_embs, mask=mask)
            return res