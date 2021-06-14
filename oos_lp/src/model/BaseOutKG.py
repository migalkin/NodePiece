# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn


class BaseOutKG(nn.Module):
    def __init__(self, dataset, args, device):
        super(BaseOutKG, self).__init__()
        self.dataset = dataset
        self.args = args
        self.emb_method = self.args.emb_method
        self.emb_dim = self.args.emb_dim
        self.device = device
        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def forward(self, triples, mask):
        new_ent_embs = torch.zeros(mask.shape[0], self.emb_dim).to(self.device)
        obs_ent_embs = torch.zeros(mask.shape[0], self.emb_dim).to(self.device)

        # get embeddings for unobserved entities (entity mask = 0 or 2)
        deep_mask = mask[mask != 1]
        if len(deep_mask) > 0:
            obs_ents_mask = torch.add(torch.mul(deep_mask, -1), 2)
            self.obs_ents_id = triples[mask != 1, obs_ents_mask]
            obs_ent_embs[mask != 1] = self.get_ent_embs(self.obs_ents_id)
            new_ent_embs[mask != 1] = self.get_new_ent_embs(
                triples[mask != 1], deep_mask
            )

        # get embeddings for observed entities (entity mask = 1)
        obs_ent_embs[mask == 1] = self.get_ent_embs(triples[mask == 1, 0])
        new_ent_embs[mask == 1] = self.get_ent_embs(triples[mask == 1, 2])

        rel_embs = self.get_rel_embs(triples[:, 1])
        scores = self.cal_score(obs_ent_embs, new_ent_embs, rel_embs)
        return scores, new_ent_embs

    def cal_score(self, obs_ents, new_ents, rels):
        scores = (obs_ents * new_ents) * rels
        scores = torch.sum(scores, dim=1)
        return scores

    def get_ent_embs(self, ent_id):
        raise NotImplementedError

    def get_new_ent_embs(self, triples, mask):
        raise NotImplementedError

    def get_rel_embs(self, rel_id):
        raise NotImplementedError

    def l2_loss(self):
        raise NotImplementedError

    def find_embedding(self, new_ent, obs_triples):
        raise NotImplementedError
