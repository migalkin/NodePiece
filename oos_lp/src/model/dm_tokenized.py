import torch
import numpy as np

from torch import nn

from vocab.nodepiece_tokenizer import NodePiece_Tokenizer
from vocab.nodepiece_encoder import NodePieceEncoder
from common.dataset import Dataset

class TokenizedDistMult(nn.Module):

    def __init__(self,
                 args: dict,
                 device: torch.device,
                 dataset: Dataset,
                 tokenizer: NodePiece_Tokenizer):
        super().__init__()

        self.rel_embs = nn.Embedding(len(dataset.rel2id) + 1, args.emb_dim, padding_idx=len(dataset.rel2id))
        torch.nn.init.xavier_uniform_(self.rel_embs.weight[:-1])

        self.embedder = NodePieceEncoder(args, tokenizer, rel_embs=self.rel_embs, graph=dataset, device=device)
        self.embedder.reset_parameters()
        self.device = device

        self.num_direct_rels = len(dataset.rel2id) // 2


    def forward(self, triples, mask):

        # don't do anythign with the mask, just compute embs and scoring function

        subs = self.embedder.encode_by_index(triples[:, 0])
        rels = self.rel_embs(triples[:, 1])
        objs = self.embedder.encode_by_index(triples[: , 2])

        score = torch.sum(subs * rels * objs, dim=-1)

        return score, None

    # for backporting with the eval code
    def get_rel_embs(self, rels: torch.LongTensor):
        return self.rel_embs(rels)

    def get_ent_embs(self, entities: torch.LongTensor):
        if self.all is None:
            self.all = self.embedder.get_all_representations()
        return self.all[entities]

    def reset(self):
        self.all = None

    def cal_score(self, obs_ents, new_ents, rels):
        scores = (obs_ents * new_ents) * rels
        scores = torch.sum(scores, dim=1)
        return scores

    def get_ent_exc_ids(self, obs_triples, new_ent):
        heads = np.where(obs_triples[:, 0] != new_ent)[0]
        h1 = np.zeros((len(heads), 1))
        tails = np.where(obs_triples[:, 2] != new_ent)[0]
        t1 = np.zeros((len(tails), 1))
        t1.fill(2)
        heads = np.expand_dims(heads, axis=1)
        heads = np.concatenate((heads, h1), axis=1)
        tails = np.expand_dims(tails, axis=1)
        tails = np.concatenate((tails, t1), axis=1)
        obs_ent = np.concatenate((heads, tails))
        obs_ent = obs_ent[np.argsort(obs_ent[:, 0])].astype(int)
        obs_ent_ids = obs_triples[obs_ent[:, 0], obs_ent[:, 1]]
        return obs_ent_ids

    def prune_tokens(self, temp_hashes: torch.LongTensor, temp_dist: torch.LongTensor):
        # this function selects self.sample_paths number of UNIQUE and NEAREST anchors from the list of anchors and their distances
        nothing_token = self.embedder.tokenizer.token2id[self.embedder.tokenizer.NOTHING_TOKEN]
        unique_ancs, unique_dists = [], []
        for anchor, dist in zip(temp_hashes, temp_dist):
            if anchor.item() not in unique_ancs and anchor.item() != nothing_token:
                unique_ancs.append(anchor.item())
                unique_dists.append(dist.item())
            else:
                continue

        # in case we stuck with the disconnected node w/o anchors, add only NOTHING tokens
        if len(unique_ancs) < self.embedder.sample_paths:
            unique_ancs += [nothing_token] * (self.embedder.sample_paths - len(unique_ancs))
            unique_dists += [0] * (self.embedder.sample_paths - len(unique_dists))

        return unique_ancs, unique_dists

    def find_embedding(self, ent_id: int, observed_triples: np.ndarray):
        # tokenizing unseen nodes using the nodepiece vocab and relational context

        all_relations = np.unique(observed_triples[:, 1])

        # build a relational context - get only outgoing edges and inverses of incoming
        outgoing_rels = observed_triples[observed_triples[:, 0] == ent_id][:, 1]
        incoming_rels = np.unique(np.array(list(set(list(all_relations)) - set(list(outgoing_rels)))))
        outgoing_rels = np.unique(np.array(outgoing_rels))
        if len(incoming_rels) > 0:
            incoming_inv = incoming_rels + self.num_direct_rels
            relational_context = np.concatenate([outgoing_rels, incoming_inv])
        else:
            relational_context = outgoing_rels  # unique for sorting as we do in pre-processing in training

        # first, get hashes (and anchor distances) of seen nodes in the neighborhood
        seen_nodes = torch.tensor(self.get_ent_exc_ids(observed_triples, ent_id), dtype=torch.long, device=self.device)

        hashes = self.embedder.hashes[seen_nodes].flatten()
        distances = self.embedder.distances[seen_nodes].flatten()

        # get topK closest anchors from the given hashes
        topk_idx = torch.argsort(distances, descending=False)

        temp_hashes = hashes[topk_idx]
        temp_dist = distances[topk_idx]

        unique_ancs, unique_dists = self.prune_tokens(temp_hashes, temp_dist)

        # build a full hash of the unseen node using nodepiece anchors, distances, and relational context
        top_hashes = torch.tensor(unique_ancs[:self.embedder.sample_paths], dtype=torch.long, device=self.device)
        top_distances = torch.tensor(unique_dists[:self.embedder.sample_paths], dtype=torch.long, device=self.device)
        top_distances = top_distances + 1  # as we are 1 hop away

        # return the encoded hash
        return self.embedder.encode_by_hash(top_hashes, top_distances, relational_context)

