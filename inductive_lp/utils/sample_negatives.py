import torch

import numpy as np
from pykeen.datasets import TriplesFactory
from collections import defaultdict


def sample_negatives(valid_triples: TriplesFactory, all_pos: TriplesFactory, num_samples: int = 50):

    val_triples = valid_triples.mapped_triples
    all_pos_triples = all_pos.mapped_triples
    num_entities = all_pos.num_entities

    head_samples, tail_samples = [[] for _ in range(len(val_triples))], [[] for _ in range(len(val_triples))]

    head_index = defaultdict(list)
    tail_index = defaultdict(list)
    for triple in all_pos_triples:
        h,r,t = triple[0].item(), triple[1].item(), triple[2].item()
        tail_index[(h, r)].append(t)
        head_index[(r, t)].append(h)

    for i, row in enumerate(val_triples):
        head, rel, tail = row[0].item(), row[1].item(), row[2].item()

        head_samples[i].append(head)
        while len(head_samples[i]) < num_samples:

            neg_head = np.random.choice(num_entities)

            if neg_head != head and neg_head not in head_index[(rel, tail)]:
                head_samples[i].append(neg_head)

        tail_samples[i].append(tail)

        while len(tail_samples[i]) < num_samples:

            neg_tail = np.random.choice(num_entities)
            if neg_tail != tail and neg_tail not in tail_index[(head, rel)]:
                tail_samples[i].append(neg_tail)

    head_samples = torch.tensor(head_samples, dtype=torch.long)
    tail_samples = torch.tensor(tail_samples, dtype=torch.long)

    return head_samples, tail_samples

