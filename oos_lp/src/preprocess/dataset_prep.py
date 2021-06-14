# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import os
import random

import numpy as np

"""
Code for dataset preparation. 
"""


class DatasetPreprocess:
    def __init__(self, dataset_name, smpl_ratio=0.2, spl_ratio=0.5):
        self.data_path = "datasets/" + dataset_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.all_triples = self.read_all()
        self.smpl_ratio = smpl_ratio
        self.spl_ratio = spl_ratio
        self.old_ent = []
        self.new_ent = []
        self.test_triples = []

    def read_all(self):
        all_lines = []
        for spl in ["train", "valid", "test"]:
            file_path = self.data_path + spl + ".txt"
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    all_lines.append(line)
        triples = np.zeros((len(all_lines), 3), dtype=int)
        for i, line in enumerate(all_lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")))
        return triples

    def make_dataset(self):
        self.single_triple_ent()
        self.split_entities()
        self.separate_triples()
        self.find_dangling_ent()
        self.find_dangling_rel()
        self.explore_split_dataset()
        self.constraint_check()
        self.save_dataset()

    def single_triple_ent(self):
        """
        find those entities that are participated in only one triple 
        add them to self.old_ent
        """
        ent_triple_count = np.zeros(self.num_ent())
        for i in range(self.num_ent()):
            ent_triple_count[i] = np.sum(self.all_triples[:, 0] == i) + np.sum(
                self.all_triples[:, 2] == i
            )
        single_triple_ent = np.where(ent_triple_count == 1)[0]
        self.old_ent.extend(list(single_triple_ent))

    def split_entities(self):
        all_ent = set(range(self.num_ent()))
        all_ent = all_ent - set(self.old_ent)
        self.new_ent = random.sample(list(all_ent), int(len(all_ent) * self.smpl_ratio))
        self.old_ent.extend(list(all_ent - set(self.new_ent)))

    def save_dataset(self):
        # save train
        new_dir = self.data_path + "processed/"
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        print("Saving old triples")
        old_triples_seq = [self.ids2triple(t) for t in self.old_triples]
        out_f = open(new_dir + "train.txt", "w")
        out_f.writelines(old_triples_seq)
        out_f.close()

        # split new triples to [val, test] and save
        print("Saving new triples")
        with open(new_dir + "valid.json", "w") as json_file:
            json.dump(self.valid_dict, json_file)
        with open(new_dir + "test.json", "w") as json_file:
            json.dump(self.test_dict, json_file)

    def constraint_check(self):
        old_ent = np.union1d(self.old_triples[:, 0], self.old_triples[:, 2])
        new_ent = set(self.test_dict.keys()).union(set(self.valid_dict.keys()))
        all_ent = len(old_ent) + len(new_ent)
        removed_ent = self.num_ent() - all_ent
        print("New entity ratio: ", len(new_ent) / self.num_ent())
        print(
            "Number of deleted entities: {}, ratio: {}".format(
                removed_ent, removed_ent / self.num_ent()
            )
        )

        total_triples = len(self.old_triples) + len(self.test_triples)
        removed_triples = len(self.all_triples) - total_triples
        print(
            "Number of deleted triples: {}, ratio: {}".format(
                removed_triples, removed_triples / len(self.all_triples)
            )
        )

        print(
            "[Train] #entities: {}, #triples: {}".format(
                len(self.old_ent), len(self.old_triples)
            )
        )
        print(
            "[Valid/Test] #entities: {}, #triples: {}".format(
                len(new_ent), len(self.test_triples)
            )
        )

    def separate_triples(self):
        self.new_triples, new_ids = self.get_ent_triples(
            self.new_ent, self.all_triples, return_ids=True
        )
        mask = np.ones(len(self.all_triples), dtype=bool)
        mask[new_ids] = False
        self.old_triples = self.all_triples[mask]

    def find_dangling_ent(self):
        old_triples_ent = np.union1d(self.old_triples[:, 0], self.old_triples[:, 2])
        self.dang_ent = list(set(self.old_ent) - set(old_triples_ent))

    def find_dangling_rel(self):
        old_triples_rel = set(self.old_triples[:, 1])
        rel_ids = list(self.rel2id.values())
        self.dang_rel = list(set(rel_ids) - old_triples_rel)

    def explore_split_dataset(self):
        new_ent_dict = {}
        for new_e in self.new_ent:
            ent_triples = self.get_ent_triples([new_e], self.new_triples)
            # remove those triples that contain dangle entity
            _, ids = self.get_ent_triples(self.dang_ent, ent_triples, return_ids=True)
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]
            # remove those triples that contain dangle relations
            ids = np.nonzero(np.in1d(ent_triples[:, 1], self.dang_rel))
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]
            # remove those triples that contain other new_ent
            other_new_ent = list(set(self.new_ent) - set([new_e]))
            _, ids = self.get_ent_triples(other_new_ent, ent_triples, return_ids=True)
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]

            # remove reflexive triples
            ref_ids = np.where(ent_triples[:, 0] == ent_triples[:, 2])
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ref_ids] = False
            ent_triples = ent_triples[mask]

            if len(ent_triples) >= 2:
                new_ent_dict[self.get_ent_str(new_e)] = [
                    [
                        self.get_ent_str(t[0]),
                        self.get_rel_str(t[1]),
                        self.get_ent_str(t[2]),
                    ]
                    for t in ent_triples
                ]
                self.test_triples.extend(ent_triples.tolist())

        new_keys = list(new_ent_dict.keys())
        valid_ent = random.sample(new_keys, int(len(new_keys) * self.spl_ratio))
        test_ent = list(set(new_keys) - set(valid_ent))
        self.valid_dict = {k: new_ent_dict[k] for k in valid_ent}
        self.test_dict = {k: new_ent_dict[k] for k in test_ent}

    def get_ent_triples(self, e_ids, triples, return_ids=False):
        h_ids = np.nonzero(np.in1d(triples[:, 0], e_ids))
        t_ids = np.nonzero(np.in1d(triples[:, 2], e_ids))
        triple_ids = np.union1d(h_ids, t_ids)
        if return_ids:
            return triples[triple_ids], triple_ids
        return triples[triple_ids]

    def smpl_new_ent(self):
        all_keys = self.ent2id.keys()
        new_keys = random.sample(all_keys, int(self.num_ent() * self.smpl_ratio))
        old_keys = set(all_keys) - set(new_keys)
        self.new_ent2id = {k: self.ent2id[k] for k in new_keys}
        self.old_ent2id = {k: self.ent2id[k] for k in old_keys}
        self.new_ent = list(self.new_ent2id.values())
        self.old_ent = list(self.old_ent2id.values())

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def triple2ids(self, triple):
        return [
            self.get_ent_id(triple[0]),
            self.get_rel_id(triple[1]),
            self.get_ent_id(triple[2]),
        ]

    def ids2triple(self, ids):
        return "{0}\t{1}\t{2}\n".format(
            self.get_ent_str(ids[0]), self.get_rel_str(ids[1]), self.get_ent_str(ids[2])
        )

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def get_ent_str(self, e_id):
        for key, value in self.ent2id.items():
            if value == e_id:
                return key

    def get_rel_str(self, r_id):
        for key, value in self.rel2id.items():
            if value == r_id:
                return key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset", default="YAGO3-10", type=str, help="wordnet dataset"
    )
    parser.add_argument(
        "-smpl_ratio", default=0.2, type=float, help="new entities ratio"
    )
    parser.add_argument(
        "-spl_ratio", default=0.5, type=float, help="new dataset split ratio"
    )
    args = parser.parse_args()

    print("sample ratio: ", args.smpl_ratio)
    dataset_prep = DatasetPreprocess(
        args.dataset, smpl_ratio=args.smpl_ratio, spl_ratio=args.spl_ratio
    )
    print("saving datasets...", args.dataset)
    dataset_prep.make_dataset()

    print("done!")
