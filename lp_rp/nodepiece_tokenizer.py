import torch
import numpy as np
import pickle
from igraph import Graph

from pykeen.triples import TriplesFactory
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import random



class NodePiece_Tokenizer:

    """
        Tokenizer for KGs: will select topK anchor nodes according to certain strategies;
            and encode other nodes as 'words' as hashes comprised of relations and anchor nodes
    """

    def __init__(self,
                 triples: TriplesFactory,
                 dataset_name: str,
                 num_anchors: int,
                 anchor_strategy: Dict[str, float],
                 limit_shortest: int = 0,
                 limit_random: int = 0,
                 add_identity: bool = True,
                 mode: str = "path",
                 ) -> None:
        super().__init__()

        self.triples_factory = triples  # original triples of the training graph, assuming inverses are added
        self.dataset_name = dataset_name
        self.num_anchors = num_anchors  # total N anchor nodes
        self.anchor_strategy = anchor_strategy  # ratios of strategies for sampling anchor nodes
        self.num_paths = num_anchors  # aux variable
        self.sp_limit = limit_shortest  # keep only K nearest anchor nodes per entity
        self.rand_limit = limit_random  # keep only K random anchor nodes per entity

        if self.sp_limit * self.rand_limit != 0:
            raise Exception("-sp_limit and -rand_limit are mutually exclusive")

        self.add_identity = add_identity  # anchor nodes will have their own indices with distance 0  in their hashes
        self.tkn_mode = mode  # only "path" mode is implemented atm

        # auxiliary tokens for the vocabulary
        self.NOTHING_TOKEN = -99  # means the node is not reachable from any of anchor nodes
        self.CLS_TOKEN = -1
        self.MASK_TOKEN = -10
        self.PADDING_TOKEN = -100
        self.SEP_TOKEN = -2

        # well, betwenness is disabled
        self.AVAILABLE_STRATEGIES = set(["degree", "betweenness", "pagerank", "random"])

        assert sum(self.anchor_strategy.values()) == 1.0, "Ratios of strategies should sum up to one"
        assert set(self.anchor_strategy.keys()).issubset(self.AVAILABLE_STRATEGIES)

        # load or create the vocabulary
        self.top_entities, self.other_entities, self.vocab = self.tokenize_kg()

        # numerical indices for entities and relations
        self.token2id = {t: i for i, t in enumerate(self.top_entities)}
        self.rel2token = {t: i + len(self.top_entities) for i, t in
                          enumerate(list(self.triples_factory.relation_to_id.values()))}
        self.vocab_size = len(self.token2id) + len(self.rel2token)

        # although we don't use paths, we count their lengths for anchor distances
        self.max_seq_len = max([len(path) for k, v in self.vocab.items() for path in v])

        if self.add_identity:
            # add identity for anchor nodes as the first / closest node
            if self.tkn_mode != "bfs":
                for anchor in self.top_entities[:-4]:  # last 4 are always service tokens
                    self.vocab[anchor] = [[anchor]] + self.vocab[anchor][:-1]
            else:
                for anchor in self.top_entities[:-4]:
                    self.vocab[anchor]['ancs'] = [anchor] + self.vocab[anchor]['ancs'][:-1]
                    self.vocab[anchor]['dists'][0] = 0



    def tokenize_kg(self):

        # creating a filename
        strategy_encoding = f"d{self.anchor_strategy['degree']}_b{self.anchor_strategy['betweenness']}_p{self.anchor_strategy['pagerank']}_r{self.anchor_strategy['random']}"

        filename = f"data/{self.dataset_name}_{self.num_anchors}_anchors_{self.num_paths}_paths_{strategy_encoding}_pykeen"
        if self.sp_limit > 0:
            filename += f"_{self.sp_limit}sp"  # for separating vocabs with limited mined shortest paths
        if self.rand_limit > 0:
            filename += f"_{self.rand_limit}rand"
        if self.tkn_mode == "bfs":
            filename += "_bfs"
        filename += ".pkl"
        self.model_name = filename.split('.pkl')[0]
        path = Path(filename)
        if path.is_file():
            anchors, non_anchors, vocab = pickle.load(open(path, "rb"))
            return anchors, non_anchors, vocab

        if type(self.triples_factory.mapped_triples) == torch.Tensor:
            src, tgt, rels = self.triples_factory.mapped_triples[:, 0].numpy(), self.triples_factory.mapped_triples[:, 2].numpy(), self.triples_factory.mapped_triples[:, 1].numpy()
        else:
            print("Input triples are expected to be in the torch.Tensor format")
            raise NotImplementedError

        # create an input object for iGraph - edge list with relation types, and then create a graph
        edgelist = [[s, t] for s, t, r in zip(src, tgt, rels)]
        graph = Graph(n=self.triples_factory.num_entities, edges=edgelist, edge_attrs={'relation': list(rels)}, directed=True)

        # sampling anchor nodes
        anchors = []
        for strategy, ratio in self.anchor_strategy.items():
            if ratio <= 0.0:
                continue
            topK = int(np.ceil(ratio * self.num_anchors))
            print(f"Computing the {strategy} nodes")
            if strategy == "degree":
                top_nodes = sorted([(i, n) for i, n in enumerate(graph.degree())], key=lambda x: x[1], reverse=True)
            elif strategy == "betweenness":
                # This is O(V^3) - disabled
                raise NotImplementedError("Betweenness is disabled due to computational costs")
            elif strategy == "pagerank":
                top_nodes = sorted([(i, n) for i, n in enumerate(graph.personalized_pagerank())], key=lambda x: x[1], reverse=True)
            elif strategy == "random":
                top_nodes = [(int(k), 1) for k in np.random.permutation(np.arange(self.triples_factory.num_entities))]

            selected_nodes = [node for node, d in top_nodes if node not in anchors][:topK]

            anchors.extend(selected_nodes)
            print(f"Added {len(selected_nodes)} nodes under the {strategy} strategy")

        # now mine the anchors per node
        vocab = self.create_all_paths(graph, anchors)
        top_entities = anchors + [self.CLS_TOKEN] + [self.MASK_TOKEN] + [self.PADDING_TOKEN] + [self.SEP_TOKEN]
        non_core_entities = [i for i in range(self.triples_factory.num_entities) if i not in anchors]

        pickle.dump((top_entities, non_core_entities, vocab), open(filename, "wb"))
        print("Vocabularized and saved!")

        return top_entities, non_core_entities, vocab


    def create_all_paths(self, graph: Graph, top_entities: List = None) -> Dict[int, List]:

        vocab = {}
        if self.rand_limit == 0:
            print(f"Computing the entity vocabulary - paths, retaining {self.sp_limit if self.sp_limit >0 else self.num_paths} shortest paths per node")
        else:
            print(f"Computing the entity vocabulary - paths, retaining {self.rand_limit} random paths per node")

        if self.tkn_mode:
            anc_set = set(top_entities)

        # single-threaded mining is found to be as fast as multi-processing + igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(range(self.triples_factory.num_entities)):
            if self.tkn_mode == "path":
                paths = graph.get_shortest_paths(v=i, to=top_entities, output="epath", mode='in')
                if len(paths[0]) > 0:
                    relation_paths = [[graph.es[path[-1]].source] + [graph.es[k]['relation'] for k in path[::-1]] for path in paths if len(path) > 0]
                else:
                    # if NO anchor can be reached from the node - encode with a special NOTHING_TOKEN
                    relation_paths = [[self.NOTHING_TOKEN] for _ in range(self.num_paths)]
                if self.sp_limit > 0:
                    relation_paths = sorted(relation_paths, key=lambda x: len(x))[:self.sp_limit]
                if self.rand_limit > 0:
                    random.shuffle(relation_paths)
                    relation_paths = relation_paths[:self.rand_limit]
                vocab[i] = relation_paths
            elif self.tkn_mode == "bfs":
                # overall limit of anchors per node
                limit = self.sp_limit if self.sp_limit != 0 else (self.rand_limit if self.rand_limit != 0 else self.num_paths)
                nearest_ancs, anc_dists = [], []
                hop = 1
                while len(nearest_ancs) < limit:
                    neigbs = graph.neighborhood(vertices=i, order=hop, mode="in", mindist=hop)  # get k-hop neighbors
                    ancs = list(set(neigbs).intersection(anc_set).difference(set(nearest_ancs)))  # find anchors in this neighborhood
                    nearest_ancs.extend(ancs)  # update the list of anchors
                    anc_dists.extend([hop for _ in range(len(ancs))])  # update the list of anchor distances
                    hop += 1
                    if hop >= 50:  # hardcoded constant for a disconnected node
                        nearest_ancs.extend([self.NOTHING_TOKEN for _ in range(limit - len(nearest_ancs))])
                        anc_dists.extend([0 for _ in range(limit - len(anc_dists))])
                        break
                vocab[i] = {'ancs': nearest_ancs[:limit], 'dists': anc_dists[:limit]}  # update the vocabulary
            else:
                raise NotImplementedError

        return vocab
