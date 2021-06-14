from pykeen.sampling import NegativeSampler
import pdb
import os
import torch
import numpy as np
from collections import defaultdict
import pickle


def get_true_subject_and_object_per_graph(triples):
    true_head = defaultdict(lambda: defaultdict(list))
    true_tail = defaultdict(lambda: defaultdict(list))
    true_relations = defaultdict(lambda: defaultdict(list))
    for head, relation, tail in triples:
        head, relation, tail = head.item(), relation.item(), tail.item()
        true_tail[head][relation].append(tail)
        true_head[tail][relation].append(head)
        true_relations[head][tail].append(relation)

    # this is correct
    for head in true_tail:
        for relation in true_tail[head]:
            true_tail[head][relation] = np.array(true_tail[head][relation])

    for tail in true_head:
        for relation in true_head[tail]:
            true_head[tail][relation] = np.array(true_head[tail][relation])

    for head in true_relations:
        for tail in true_relations[head]:
            true_relations[head][tail] = np.array(true_relations[head][tail])

    return dict(true_head), dict(true_tail), dict(true_relations)


class FilteredNegativeSampler(NegativeSampler):
    def __init__(self, triples_factory, num_negs_per_pos=None, dataset_name='fb15k237'):
        super(FilteredNegativeSampler, self).__init__(triples_factory, num_negs_per_pos)
        true_head_path = os.path.join("cached_input", dataset_name, "true_heads.pt")
        true_tail_path = os.path.join("cached_input", dataset_name, "true_tails.pt")
        true_relations_path = os.path.join("cached_input", dataset_name, "true_relations.pt")
        if os.path.exists(true_head_path) and os.path.exists(true_tail_path) and os.path.exists(true_relations_path):
            with open(true_head_path, 'rb') as handle:
                self.true_head = pickle.load(handle)
            with open(true_tail_path, 'rb') as handle:
                self.true_tail = pickle.load(handle)
            with open(true_relations_path, 'rb') as handle:
                self.true_relations = pickle.load(handle)
        else:
            os.makedirs(os.path.join("cached_input", dataset_name))
            self.true_head, self.true_tail, self.true_relations = \
                get_true_subject_and_object_per_graph(triples_factory.mapped_triples)
            with open(true_head_path, 'wb') as handle:
                pickle.dump(self.true_head, handle)
            with open(true_tail_path, 'wb') as handle:
                pickle.dump(self.true_tail, handle)
            with open(true_relations_path, 'wb') as handle:
                pickle.dump(self.true_relations, handle)

    def sample_entities(self, h, r, t, corrput_head=True):
        true_entities = self.true_head[t][r] if corrput_head else self.true_tail[h][r]
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.num_negs_per_pos:
            negative_sample = np.random.choice(self.num_entities - 1, size=self.num_negs_per_pos)
            mask = np.in1d(
                negative_sample,
                true_entities,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:self.num_negs_per_pos]

    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """Generate negative samples from the positive batch."""
        half = positive_batch.shape[0] // 2
        negative_batch_entity = positive_batch.repeat(self.num_negs_per_pos, 1).view(-1, positive_batch.shape[0], 3)

        # Sample random entities as replacement
        for i in range(len(positive_batch)):
            h, r, t = positive_batch[i]
            h, r, t = h.item(), r.item(), t.item()

            if i < half:
                negative_head = torch.from_numpy(self.sample_entities(h, r, t, corrput_head=True))
                negative_batch_entity[:, i, 0] = negative_head
            else:
                negative_tail = torch.from_numpy(self.sample_entities(h, r, t, corrput_head=False))
                negative_batch_entity[:, i, 2] = negative_tail
            # pdb.set_trace()

        return negative_batch_entity.view(-1, 3)


class RelationalNegativeSampler(FilteredNegativeSampler):

    def __init__(self, triples_factory, num_negs_per_pos=None, num_negs_per_pos_rel=None, dataset_name=None):
        super(RelationalNegativeSampler, self).__init__(triples_factory, num_negs_per_pos, dataset_name)
        self.num_negs_per_pos_rel = num_negs_per_pos_rel if num_negs_per_pos_rel is not None else 1

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of entities to sample from."""
        return self.triples_factory.num_relations

    def sample_relations(self, h, t):
        true_relations = self.true_relations[h][t]
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.num_negs_per_pos_rel:
            negative_sample = np.random.choice(self.num_relations - 1, size=self.num_negs_per_pos_rel)
            mask = np.in1d(
                negative_sample,
                true_relations,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        return np.concatenate(negative_sample_list)[:self.num_negs_per_pos_rel]

    def sample(self, positive_batch):
        bsz = positive_batch.shape[0]
        negative_entities = super().sample(positive_batch)
        negative_relations = positive_batch.repeat(self.num_negs_per_pos_rel, 1).view(-1, bsz, 3)

        for i in range(len(positive_batch)):
            h, r, t = positive_batch[i]
            h, r, t = h.item(), r.item(), t.item()
            negative_relation = torch.from_numpy(self.sample_relations(h, t))
            negative_relations[:, i, 1] = negative_relation

        return torch.cat([negative_entities, negative_relations.view(-1, 3)])
