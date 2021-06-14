# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Bahare Fatemi.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is from https://github.com/baharefatemi/SimplE by Bahare Fatemi
####################################################################################
import json
import math
import random

import numpy as np
import torch

from utils import random_new_ent_mask


class Dataset:
    def __init__(self, dataset_name, cons_masking=False, mask_prob=0.5, tokenize=False):
        self.dataset_name = dataset_name
        self.data_path = "../datasets/" + dataset_name + "/processed/"
        self.ent2id = {}
        self.rel2id = {}
        self.train_data = self.read_text(self.data_path + "train.txt")
        self.init_num_ent = self.num_ent()

        if not tokenize:
            self.adj_list, self.ent_freq, self.rel_freq = self.dataset_stat(self.train_data)
            self.max_degree = max(map(len, self.adj_list))
        self.val_test_data = {
            "valid": self.read_json(self.data_path + "valid.json"),
            "test": self.read_json(self.data_path + "test.json"),
        }
        self.batch_index = 0
        self.cons_masking = cons_masking
        self.mask_prob = mask_prob

    def read_text(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")))
        return triples

    def dataset_stat(self, triples):
        adj_list = [[] for _ in range(self.num_ent())]
        ent_freq = torch.zeros(self.num_ent())
        rel_freq = torch.zeros(self.num_rel())
        for t in triples:
            t = t.astype(int)
            adj_list[t[0]].append([t[2], t[1], 0])
            adj_list[t[2]].append([t[0], t[1], 1])
            ent_freq[t[0]] += 1
            ent_freq[t[2]] += 1
            rel_freq[t[1]] += 1
        max_degree = max(map(len, adj_list))
        for item in adj_list:
            item.extend(
                [[self.num_ent(), self.num_rel(), 0]] * (max_degree - len(item))
            )
        return (
            torch.tensor(adj_list),
            ent_freq / triples.shape[0],
            rel_freq / triples.shape[0],
        )

    def read_json(self, file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)

        for k, v in data.items():
            triples = []
            for t in v:
                if t[0] == k:
                    triples_ids = [t[0], self.get_rel_id(t[1]), self.get_ent_id(t[2])]
                else:
                    triples_ids = [self.get_ent_id(t[0]), self.get_rel_id(t[1]), t[2]]
                triples.append(triples_ids)
            data[k] = triples
        return data

    def triple2ids(self, triple):
        return [
            self.get_ent_id(triple[0]),
            self.get_rel_id(triple[1]),
            self.get_ent_id(triple[2]),
        ]

    def get_ent_id(self, ent, add_ent=True):
        if not ent in self.ent2id:
            if add_ent:
                self.ent2id[ent] = len(self.ent2id)
            else:
                return self.init_num_ent
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.init_num_ent - 1)
        while rand_ent == ent:
            rand_ent = random.randint(0, self.init_num_ent - 1)
        return rand_ent

    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.train_data):
            batch = self.train_data[self.batch_index: self.batch_index + batch_size]
            self.batch_index += batch_size
        else:
            batch = self.train_data[self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype(
            "int"
        )  # appending the +1 label

    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0])  # flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2])  # flipping tail
        neg_batch[:, -1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        labels = torch.tensor(batch[:, 3], dtype=torch.float, device=device)
        triples = batch[:, 0:3]

        if self.cons_masking:
            pos_new_ent_mask = random_new_ent_mask(pos_batch[:, 0:3], self.mask_prob)
            neg_new_ent_mask = np.repeat(np.copy(pos_new_ent_mask), neg_ratio, axis=0)
            new_ent_mask = np.append(pos_new_ent_mask, neg_new_ent_mask, axis=0)
            new_ent_mask = torch.tensor(new_ent_mask, dtype=torch.long).to(device)
        else:
            new_ent_mask = torch.tensor(
                random_new_ent_mask(triples, self.mask_prob), dtype=torch.long
            ).to(device)

        return (
            torch.tensor(triples, dtype=torch.long, device=device),
            labels,
            new_ent_mask,
        )

    def was_last_batch(self):
        return self.batch_index == 0

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.train_data)) / batch_size))

    def num_batch_simulated(self, simulate_batch_size):
        return int(math.ceil(float(len(self.train_data)) / simulate_batch_size))

    def generate_neg_obs(self, obs_triples, new_ent, neg_ratio):
        neg_triples = np.repeat(np.copy(obs_triples), neg_ratio, axis=0)
        for i in range(len(neg_triples)):
            if neg_triples[i, 0] == new_ent:
                # new entity is in head -> flip tail
                neg_triples[i, 2] = self.rand_ent_except(neg_triples[i, 2])
            else:
                # new entity is in tail -> flip head
                neg_triples[i, 0] = self.rand_ent_except(neg_triples[i, 0])
        return neg_triples
