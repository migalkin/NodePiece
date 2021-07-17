from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from ogb_wikikg2.dataloader import TestDataset
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

from torch.nn import TransformerEncoderLayer, TransformerEncoder, GRU

from ogb.linkproppred import Evaluator


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 tokenizer, pooler, use_rels, rel_policy, sample_paths,
                 trf_layers, trf_heads, trf_hidden, drop, use_distances, max_seq_len,
                 sample_rels, triples, ablate_anchors, device,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.pooler = pooler
        self.use_rels = use_rels
        self.policy = rel_policy
        self.sample_paths = sample_paths
        self.use_distances = use_distances
        self.max_seq_len = max_seq_len
        self.sample_rels = sample_rels
        self.drop = drop
        self.triples = triples
        self.ablate_anchors = ablate_anchors
        self.device = device


        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        # anchors hashing mechanism

        self.set_enc = nn.Sequential(
            nn.Linear(self.entity_dim * (self.sample_paths + self.sample_rels), self.entity_dim * 2), nn.Dropout(drop), nn.ReLU(),
            nn.Linear(self.entity_dim * 2, self.entity_dim)
        ) if not self.ablate_anchors else nn.Sequential(
            nn.Linear(self.entity_dim * sample_rels, self.entity_dim * 2), nn.Dropout(drop), nn.ReLU(),
            nn.Linear(self.entity_dim * 2, self.entity_dim)
        )

        # init
        for module in self.set_enc.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.anchor_embeddings = nn.Embedding(num_embeddings=len(tokenizer.token2id)+1, embedding_dim=self.entity_dim)
        nn.init.uniform_(
            tensor=self.anchor_embeddings.weight,  # .weight for Embedding
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # back to normal relation embs, +1 for the padding relation
        self.relation_embedding = nn.Embedding(num_embeddings=nrelation, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )



        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'AutoSF']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')

        self.evaluator = evaluator

        print("Creating hashes")

        hashes = [
            [tokenizer.token2id[token] for token in vals['ancs'][:min(self.sample_paths, len(vals['ancs']))]] + [
                tokenizer.token2id[tokenizer.PADDING_TOKEN]] * (self.sample_paths - len(vals['ancs']))
            for entity, vals in tokenizer.vocab.items()
        ]
        distances = [
            [d for d in vals['dists'][:min(self.sample_paths, len(vals['dists']))]] + [0] * (
                        self.sample_paths - len(vals['dists']))
            for entity, vals in tokenizer.vocab.items()
        ]

        self.max_seq_len = max([d for row in distances for d in row])
        print(
            f"Changed max seq len from {max_seq_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

        self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
        self.distances = torch.tensor(distances, dtype=torch.long, device=self.device)

        print("Creating relational context")
        if self.sample_rels > 0:
            pad_idx = nrelation - 1
            e2r = defaultdict(set)
            for i in tqdm(range(len(self.triples['head']))):
                e2r[self.triples['head'][i]].add(self.triples['relation'][i])

            len_stats = [len(v) for k,v in e2r.items()]
            print(f"Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")
            unique_1hop_relations = [
                random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (self.sample_rels-min(len(e2r[i]), self.sample_rels))
                for i in range(self.nentity)
            ]
            self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.device)


        # distance integer to denote path lengths
        self.dist_emb = nn.Embedding(self.max_seq_len + 1, embedding_dim=self.entity_dim)
        torch.nn.init.xavier_uniform_(self.dist_emb.weight)

        with torch.no_grad():
            self.anchor_embeddings.weight[tokenizer.token2id[tokenizer.PADDING_TOKEN]] = torch.zeros(self.entity_dim)
            self.relation_embedding.weight.data[-1] = torch.zeros(self.relation_dim)
            self.dist_emb.weight[0] = torch.zeros(self.entity_dim)


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
            pooled = self.set_enc(anc_embs.transpose(1, 0))  # output shape: (seq_len, bs, dim)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)
            if self.policy == "cat":
                pooled = self.linear(pooled)

        return pooled


    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:

        hashes, dists = self.hashes[entities], self.distances[entities]

        #anc_embs = torch.index_select(self.anchor_embeddings, dim=0, index=hashes)
        anc_embs = self.anchor_embeddings(hashes)
        mask = None

        if self.use_distances:
            dist_embs = self.dist_emb(dists)
            anc_embs += dist_embs

        if self.sample_rels > 0:
            rels = self.unique_1hop_relations[entities]  # (bs, rel_sample_size)
            #rels = torch.index_select(self.relation_embedding, dim=0, index=rels)   # (bs, rel_sample_size, dim)
            rels = self.relation_embedding(rels)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)
        return anc_embs


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = self.encode_by_index(sample[:, 0]).unsqueeze(1)

            relation = self.relation_embedding(sample[:, 1]).unsqueeze(1)
            # relation = torch.index_select(
            #     self.relation_embedding,
            #     dim=0,
            #     index=sample[:, 1]
            # ).unsqueeze(1)

            tail = self.encode_by_index(sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = self.encode_by_index(head_part.view(-1)).view(batch_size, negative_sample_size, -1)

            relation = self.relation_embedding(tail_part[:, 1]).unsqueeze(1)
            # relation = torch.index_select(
            #     self.relation_embedding,
            #     dim=0,
            #     index=tail_part[:, 1]
            # ).unsqueeze(1)

            tail = self.encode_by_index(tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = self.encode_by_index(head_part[:, 0]).unsqueeze(1)

            relation = self.relation_embedding(head_part[:, 1]).unsqueeze(1)
            # relation = torch.index_select(
            #     self.relation_embedding,
            #     dim=0,
            #     index=head_part[:, 1]
            # ).unsqueeze(1)

            tail = self.encode_by_index(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'AutoSF': self.AutoSF,
            'PairRE': self.PairRE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def AutoSF(self, head, relation, tail, mode):

        if mode == 'head-batch':
            rs = torch.chunk(relation, 4, dim=-1)
            ts = torch.chunk(tail, 4, dim=-1)
            rt0 = rs[0] * ts[0]
            rt1 = rs[1] * ts[1] + rs[2] * ts[3]
            rt2 = rs[0] * ts[2] + rs[2] * ts[3]
            rt3 = -rs[1] * ts[1] + rs[3] * ts[2]
            rts = torch.cat([rt0, rt1, rt2, rt3], dim=-1)
            score = torch.sum(head * rts, dim=-1)

        else:
            hs = torch.chunk(head, 4, dim=-1)
            rs = torch.chunk(relation, 4, dim=-1)
            hr0 = hs[0] * rs[0]
            hr1 = hs[1] * rs[1] - hs[3] * rs[1]
            hr2 = hs[2] * rs[0] + hs[3] * rs[3]
            hr3 = hs[1] * rs[2] + hs[2] * rs[2]
            hrs = torch.cat([hr0, hr1, hr2, hr3], dim=-1)
            score = torch.sum(hrs * tail, dim=-1)

        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                args,
                'head-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                args,
                'tail-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0],
                                                          'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics