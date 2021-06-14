# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from tqdm import tqdm

from common.measure import Measure


class OutKGTester:
    def __init__(self, dataset):
        self.dataset = dataset
        self.measure = Measure()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.unseen_ent = self.get_unseen_entities()
        self.new_node_init = None

    def test(self, model_path, valid_or_test):
        if type(model_path) == "tuple" or type(model_path) == "str":
            self.model = torch.load(model_path)  # removed module
        else:
            self.model = model_path  # removed module

        self.model.to(self.device)  # for bypassing torch DP
        self.model.eval()
        if hasattr(self.model, "reset"):
            self.model.reset()

        for i, new_ent in enumerate(tqdm(self.dataset.val_test_data[valid_or_test].keys())):
            self.new_node_init = None
            ent_triples = self.get_ent_triples(new_ent, valid_or_test)
            for j in range(len(ent_triples)):
                obs_triples = np.delete(ent_triples, [j], axis=0)
                target_triple = ent_triples[j]
                new_ent_id = self.dataset.init_num_ent
                self.predict(target_triple, new_ent_id, obs_triples)

        self.measure.normalize()
        self.measure.print_()
        return self.measure.mrr

    def predict(self, target_triple, new_ent, obs_triples):
        head_or_tail = "tail" if target_triple[0] == new_ent else "head"
        queries_ent, rel_id = self.create_queries(
            target_triple, head_or_tail, obs_triples
        )
        rel_emb = self.model.get_rel_embs(
            torch.tensor(rel_id, dtype=torch.long).to(self.device)
        )
        new_ent_emb = self.model.find_embedding(new_ent, obs_triples)
        queries_ent_emb = self.model.get_ent_embs(
            torch.tensor(queries_ent, dtype=torch.long).to(self.device)
        )
        scores = self.model.cal_score(queries_ent_emb, new_ent_emb, rel_emb)
        rank = self.get_rank(scores)
        self.measure.update(rank)

    def filter_entities(self, obs_triples, rel_id, head_or_tail):
        idx = np.where(obs_triples[:, 1] == rel_id)[0]
        if head_or_tail == "head":
            return obs_triples[idx, 0]
        else:
            return obs_triples[idx, 2]

    def create_queries(self, target_triple, head_or_tail, obs_triples):
        head_id, rel_id, tail_id = target_triple
        ent_list = list(range(self.dataset.init_num_ent))
        filtered_ent_list = self.filter_entities(
            obs_triples, rel_id, head_or_tail
        ).tolist()
        ent_list = set(ent_list) - set(filtered_ent_list)
        if head_or_tail == "head":
            ent_list = list(ent_list - set([head_id]))
            return [head_id] + ent_list, rel_id
        elif head_or_tail == "tail":
            ent_list = list(ent_list - set([tail_id]))
            return [tail_id] + ent_list, rel_id

    def get_rank(self, sim_scores):
        # assuming the test fact is the first one
        equals = ((sim_scores == sim_scores[0]).sum() - 1) // 2
        higher = (sim_scores > sim_scores[0]).sum()
        rank = (equals + higher + 1.0).to(torch.float)
        return rank

    def get_unseen_entities(self):
        valid_ent = set(self.dataset.val_test_data["valid"].keys())
        test_ent = set(self.dataset.val_test_data["test"].keys())
        return valid_ent.union(test_ent)

    def get_ent_triples(self, ent, valid_or_test):
        triples = np.asarray(self.dataset.val_test_data[valid_or_test][ent])
        h = np.where(triples[:, 0] == ent)
        triples[h, 0] = self.dataset.init_num_ent
        t = np.where(triples[:, 2] == ent)
        triples[t, 2] = self.dataset.init_num_ent
        return triples.astype(int)
