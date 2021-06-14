# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from torch import nn

from model.BaseOutKG import BaseOutKG


class DisMultOutKG(BaseOutKG):
    def build_model(self):
        self.num_ent = self.dataset.init_num_ent
        self.num_rel = self.dataset.num_rel()

        self.ent_embs = nn.Embedding(
            self.num_ent + 1, self.emb_dim, padding_idx=self.num_ent
        ).to(self.device)
        self.rel_embs = nn.Embedding(
            self.num_rel + 1, self.emb_dim, padding_idx=self.num_rel
        ).to(self.device)
        nn.init.xavier_uniform_(self.ent_embs.weight[:-1])
        nn.init.xavier_uniform_(self.rel_embs.weight[:-1])

    def get_ent_embs(self, ent_id):
        # lookup the embedding
        return self.ent_embs(ent_id)

    def get_rel_embs(self, rel_id):
        return self.rel_embs(rel_id)

    def get_new_ent_embs(self, triples, mask):
        # find the embedding for the unobserved entities
        new_ents_id = triples[torch.arange(triples.shape[0], dtype=torch.long), mask]
        ent_neighbors = self.dataset.adj_list[new_ents_id].to(self.device)
        neighbors_ent_embs = self.get_ent_embs(ent_neighbors[:, :, 0].view(-1)).view(
            ent_neighbors.shape[0], ent_neighbors.shape[1], self.emb_dim
        )
        neighbors_rel_embs = self.get_rel_embs(ent_neighbors[:, :, 1].view(-1)).view(
            ent_neighbors.shape[0], ent_neighbors.shape[1], self.emb_dim
        )
        ent_mask = (ent_neighbors[:, :, 0] != self.obs_ents_id.unsqueeze(1)) * (
                ent_neighbors[:, :, 0] != self.num_ent
        )
        neighbors_ent_embs = neighbors_ent_embs * ent_mask.unsqueeze(2).expand(
            neighbors_ent_embs.shape
        ).to(torch.float)
        neighbors_rel_embs = neighbors_rel_embs * ent_mask.unsqueeze(2).expand(
            neighbors_rel_embs.shape
        ).to(torch.float)
        return self.infer_emb(neighbors_ent_embs, neighbors_rel_embs, mask=ent_mask)

    def l2_loss(self):
        emb_reg = (torch.norm(self.ent_embs.weight, p=2) ** 2) + (
                torch.norm(self.rel_embs.weight, p=2) ** 2
        )
        return emb_reg / 2

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

    def find_embedding(self, new_ent, obs_triples):
        obs_ent_ids = self.get_ent_exc_ids(obs_triples, new_ent)
        obs_rel_ids = obs_triples[:, 1]
        obs_ent_embs = self.get_ent_embs(
            torch.tensor(obs_ent_ids, dtype=torch.long).to(self.device)
        ).unsqueeze(0)
        obs_rel_embs = self.get_rel_embs(
            torch.tensor(obs_rel_ids, dtype=torch.long).to(self.device)
        ).unsqueeze(0)
        return self.infer_emb(obs_ent_embs, obs_rel_embs).squeeze(0)

    def infer_emb(self, ent_emb, rel_emb, mask=None):
        # infers embedding of unobserved entities based on the emb_method (aggregation function)
        if self.emb_method == "ERAverage":
            return self.find_neighbors_avg(ent_emb, rel_emb=rel_emb, mask=mask)
        elif self.emb_method == "Average":
            return self.find_neighbors_avg(ent_emb, mask=mask)
        elif self.emb_method == "LS":
            labels = torch.ones(ent_emb.shape[0], ent_emb.shape[1]).to(self.device)
            if mask is not None:
                labels = labels * mask.to(torch.float)

            A = nn.functional.normalize(ent_emb * rel_emb, dim=2)
            A_t = A.permute(0, 2, 1)
            c = torch.inverse(
                torch.bmm(A_t, A)
                + self.args.reg_ls
                * (
                    torch.eye(A.shape[2])
                        .unsqueeze(0)
                        .expand(A.shape[0], A.shape[2], A.shape[2])
                ).to(self.device)
            )
            return torch.bmm(torch.bmm(c, A_t), labels.unsqueeze(2)).squeeze()

        elif self.emb_method == "LS_unnorm":
            labels = torch.ones(ent_emb.shape[0], ent_emb.shape[1]).to(self.device)
            if mask is not None:
                labels = labels * mask.to(torch.float)

            A = ent_emb * rel_emb
            A_t = A.permute(0, 2, 1)
            c = torch.inverse(
                torch.bmm(A_t, A)
                + self.args.reg_ls
                * (
                    torch.eye(A.shape[2])
                        .unsqueeze(0)
                        .expand(A.shape[0], A.shape[2], A.shape[2])
                ).to(self.device)
            )
            return torch.bmm(torch.bmm(c, A_t), labels.unsqueeze(2)).squeeze()

    def find_neighbors_avg(self, ent_emb, rel_emb=None, mask=None):
        if mask is None:
            if rel_emb is None:
                return torch.mean(ent_emb, dim=1)
            else:
                return torch.mean(ent_emb * rel_emb, dim=1)
        else:
            mask_sum = torch.sum(mask, dim=1)
            mask_sum[mask_sum == 0] = 1.0
            if rel_emb is None:
                return torch.sum(ent_emb, dim=1) / mask_sum.unsqueeze(1).to(torch.float)
            else:
                return torch.sum(ent_emb * rel_emb, dim=1) / mask_sum.unsqueeze(1).to(
                    torch.float
                )
