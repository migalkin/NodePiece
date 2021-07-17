import torch
import numpy as np
import pickle
import multiprocessing as mp
import networkx as nx
from functools import partial
from igraph import Graph
from torch_geometric.data.cluster import ClusterData
from torch_geometric.data import Data

from torch import nn
from pykeen.triples import TriplesFactory
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
import random

class NodePiece_OGB:

    """
        NodePiece for OGB: uses a parallel batch mode for anchor mining
    """

    def __init__(self,
                 triples: TriplesFactory,
                 dataset_name: str,
                 num_anchors: int,
                 anchor_strategy: Dict[str, float],
                 num_paths: int,
                 limit_shortest: int = 0,
                 limit_random: int = 0,
                 add_identity: bool = False,
                 mode: str = "bfs",
                 tkn_batch: int = 100,
                 inv: bool = False,
                 dir: str = "in",
                 partition: int = 1,
                 cpus: int = 1,
                 ) -> None:
        super().__init__()

        self.triples_factory = triples
        self.dataset_name = dataset_name
        self.num_anchors = num_anchors
        self.anchor_strategy = anchor_strategy
        self.num_paths = num_paths
        self.sp_limit = limit_shortest
        self.rand_limit = limit_random
        self.partition = partition
        self.cpus = cpus

        if self.sp_limit * self.rand_limit != 0:
            raise Exception("-sp_limit and -rand_limit are mutually exclusive")

        self.add_identity = add_identity
        self.tkn_mode = mode

        self.NOTHING_TOKEN = -99
        self.CLS_TOKEN = -1
        self.MASK_TOKEN = -10
        self.PADDING_TOKEN = -100
        self.SEP_TOKEN = -2

        self.batch_size = tkn_batch
        self.use_inv = inv
        self.dir = dir


        self.AVAILABLE_STRATEGIES = set(["degree", "betweenness", "pagerank", "random"])

        assert sum(self.anchor_strategy.values()) == 1.0, "Ratios of strategies should sum up to one"
        assert set(self.anchor_strategy.keys()).issubset(self.AVAILABLE_STRATEGIES)

        self.top_entities, self.other_entities, self.vocab = self.tokenize_kg()

        self.token2id = {t: i for i, t in enumerate(self.top_entities)}
        self.rel2token = {t: i + len(self.top_entities) for i, t in
                          enumerate(list(self.triples_factory.relation_to_id.values()))}
        self.vocab_size = len(self.token2id) + len(self.rel2token)


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

        strategy_encoding = f"d{self.anchor_strategy['degree']}_b{self.anchor_strategy['betweenness']}_p{self.anchor_strategy['pagerank']}_r{self.anchor_strategy['random']}"

        filename = f"data/{self.dataset_name}_{self.num_anchors}_anchors_{self.num_paths}_paths_{strategy_encoding}_pykeen"
        if self.sp_limit > 0:
            filename += f"_{self.sp_limit}sp"  # for separating vocabs with limited mined shortest paths
        if self.rand_limit > 0:
            filename += f"_{self.rand_limit}rand"
        if self.tkn_mode == "bfs":
            filename += "_bfs"
        if self.partition > 1:
            filename += f"_metis{self.partition}"
        filename += ".pkl"
        self.model_name = filename.split('.pkl')[0]
        path = Path(filename)
        if path.is_file():
            anchors, non_anchors, vocab = pickle.load(open(path, "rb"))
            return anchors, non_anchors, vocab

        if type(self.triples_factory.mapped_triples) == torch.Tensor:
            src, tgt, rels = self.triples_factory.mapped_triples[:, 0].numpy(), self.triples_factory.mapped_triples[:, 2].numpy(), self.triples_factory.mapped_triples[:, 1].numpy()
        else:
            # dummy triple factory for OGB
            src, tgt, rels = self.triples_factory.mapped_triples['head'], self.triples_factory.mapped_triples['tail'], self.triples_factory.mapped_triples['relation']
        edgelist = [[s, t] for s, t, r in zip(src, tgt, rels)]
        graph = Graph(n=self.triples_factory.num_entities, edges=edgelist, edge_attrs={'relation': list(rels)}, directed=True)


        anchors = []
        for strategy, ratio in self.anchor_strategy.items():
            if ratio <= 0.0:
                continue
            topK = int(np.ceil(ratio * self.num_anchors))
            print(f"Computing the {strategy} nodes")
            if strategy == "degree":
                # top_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True) # OLD NetworkX
                top_nodes = sorted([(i, n) for i, n in enumerate(graph.degree())], key=lambda x: x[1], reverse=True)
            elif strategy == "betweenness":
                raise NotImplementedError("Betweenness is disabled due to computational costs")
            elif strategy == "pagerank":
                #top_nodes = sorted(nx.pagerank(nx.DiGraph(graph)).items(), key=lambda x: x[1], reverse=True)
                top_nodes = sorted([(i, n) for i, n in enumerate(graph.personalized_pagerank())], key=lambda x: x[1], reverse=True)
            elif strategy == "random":
                top_nodes = [(int(k), 1) for k in np.random.permutation(np.arange(self.triples_factory.num_entities))]

            # slow version
            # selected_nodes = [node for node, d in top_nodes if node not in anchors][:topK]

            # faster version
            tops = {k: v for k, v in top_nodes}
            # remove ancs
            for a in anchors:
                tops.pop(a, None)
            selected_nodes = [k for k, v in tops.items()][:topK]  # dict is ordered so the sorted order is preserved

            anchors.extend(selected_nodes)
            print(f"Added {len(selected_nodes)} nodes under the {strategy} strategy")

        vocab = self.create_all_paths(graph, anchors) if self.partition == 1 else self.mine_parallel(anchors) # self.mine_partitions(anchors)
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
            top_np = np.array(top_entities)
            anc_set = set(top_entities)

        # igraph_mode = "in" if self.use_inv else "all"

        for i in tqdm(range(0, self.triples_factory.num_entities, self.batch_size)):

            batch_verts = list(range(0, self.triples_factory.num_entities))[i: i+self.batch_size]

            limit = self.sp_limit if self.sp_limit != 0 else (self.rand_limit if self.rand_limit != 0 else self.num_paths)
            nearest_ancs, anc_dists = [[] for _ in range(len(batch_verts))], [[] for _ in range(len(batch_verts))]
            hop = 1
            while any([len(lst) < limit for lst in nearest_ancs]):
                tgt_idx = [k for k in range(len(batch_verts)) if len(nearest_ancs[k]) < limit]
                verts = [batch_verts[idx] for idx in tgt_idx]
                neigbs_list = graph.neighborhood(vertices=verts, order=hop, mode=self.dir, mindist=hop)  # list of lists
                ancs = [list(set(neigbs).intersection(anc_set).difference(set(nearest_ancs[id]))) for id, neigbs in enumerate(neigbs_list)]
                # updating anchor lists
                for local_idx, global_idx in enumerate(tgt_idx):
                    nearest_ancs[global_idx].extend(ancs[local_idx])
                    anc_dists[global_idx].extend([hop for _ in range(len(ancs[local_idx]))])
                # nearest_ancs.extend(ancs)
                # anc_dists.extend([hop for _ in range(len(ancs))])
                hop += 1
                if hop >= 50:  # hardcoded constant for a disconnected node
                    for idx in tgt_idx:
                        nearest_ancs[idx].extend([self.NOTHING_TOKEN for _ in range(limit - len(nearest_ancs[idx]))])
                        anc_dists[idx].extend([0 for _ in range(limit - len(anc_dists[idx]))])
                        break

            # update the vocab
            for idx, v in enumerate(batch_verts):
                vocab[v] = {'ancs': nearest_ancs[idx][:limit], 'dists': anc_dists[idx][:limit]}


        return vocab


    def mine_partitions(self, anchors) -> Dict[int, List]:
        # let's try splitting the graph into connected components with METIS and run mining on them

        src, tgt = self.triples_factory.mapped_triples['head'], self.triples_factory.mapped_triples['tail']
        edgelist = [[s, t] for s, t in zip(src, tgt)]
        pyg_graph = Data(edge_index=torch.tensor(edgelist).T, num_nodes=self.triples_factory.num_entities)
        print(f"Using METIS to partition the graph into {self.partition} partitions")
        clusters = ClusterData(pyg_graph, num_parts=self.partition)

        vocab = {}

        # now find anchors in each cluster and tokenize clusters one by one
        for cluster_id in range(len(clusters)):
            print(f"Processing cluster {cluster_id}")
            start = int(clusters.partptr[cluster_id])
            end = int(clusters.partptr[cluster_id + 1])
            cluster_nodes = clusters.perm[start: end]
            cluster_anchors = list(set(cluster_nodes.cpu().numpy()).intersection(set(anchors)))

            # map global anchor IDs to local ids [they start from 0 to num_nodes in cluster]
            local_anchor_ids = []
            for anc_idx, anc in enumerate(cluster_anchors):
                local_id = (cluster_nodes == anc).nonzero(as_tuple=False)
                local_anchor_ids.append(local_id.item())

            node_mapping = {i: n.item() for i, n in enumerate(cluster_nodes)}
            anchor_mapping = {loc_id: glob_id for loc_id, glob_id in zip(local_anchor_ids, cluster_anchors)}
            anchor_mapping[-99] = -99

            cluster = clusters[cluster_id]
            cl_vocab = self.bfs_cluster(cluster, local_anchor_ids)

            # re-map back to global ids
            cl_vocab = {node_mapping[k]: {
                'ancs': [anchor_mapping[a] for a in v['ancs']],
                'dists': v['dists']
            } for k, v in cl_vocab.items()}
            vocab.update(cl_vocab)

        # sort vocab by key - need it to be in the ascending order 0 - n
        vocab = dict(sorted(vocab.items()))
        return vocab


    def bfs_cluster(self, cluster: Data, anchors: List, tqdm_pos=None) -> Dict[int, List]:

        num_nodes = cluster.edge_index.max() + 1
        edge_index = cluster.edge_index
        anc_set = set(anchors)
        vocab = {}

        edgelist = [[s.item(), t.item()] for s, t in zip(edge_index[0], edge_index[1])]
        graph = Graph(n=num_nodes, edges=edgelist, directed=False)

        # for i in tqdm(range(num_nodes)):
        #     limit = self.sp_limit if self.sp_limit != 0 else (self.rand_limit if self.rand_limit != 0 else self.num_paths)
        #     nearest_ancs, anc_dists = [], []
        #     hop = 1
        #     while len(nearest_ancs) < limit:
        #         neigbs = graph.neighborhood(vertices=i, order=hop, mode="all", mindist=hop)
        #         ancs = list(set(neigbs).intersection(anc_set).difference(set(nearest_ancs)))
        #         nearest_ancs.extend(ancs)
        #         anc_dists.extend([hop for _ in range(len(ancs))])
        #         hop += 1
        #         if hop >= 50:  # hardcoded constant for a disconnected node
        #             nearest_ancs.extend([self.NOTHING_TOKEN for _ in range(limit - len(nearest_ancs))])
        #             anc_dists.extend([0 for _ in range(limit - len(anc_dists))])
        #             break
        #     vocab[i] = {'ancs': nearest_ancs[:limit], 'dists': anc_dists[:limit]}

        ## BATCH version
        for i in tqdm(range(0, num_nodes, self.batch_size), position=tqdm_pos):

            batch_verts = list(range(0, num_nodes))[i: i+self.batch_size]

            limit = self.sp_limit if self.sp_limit != 0 else (self.rand_limit if self.rand_limit != 0 else self.num_paths)
            nearest_ancs, anc_dists = [[] for _ in range(len(batch_verts))], [[] for _ in range(len(batch_verts))]
            hop = 1
            while any([len(lst) < limit for lst in nearest_ancs]):
                tgt_idx = [k for k in range(len(batch_verts)) if len(nearest_ancs[k]) < limit]
                verts = [batch_verts[idx] for idx in tgt_idx]
                neigbs_list = graph.neighborhood(vertices=verts, order=hop, mode=self.dir, mindist=hop)  # list of lists
                ancs = [list(set(neigbs).intersection(anc_set).difference(set(nearest_ancs[id]))) for id, neigbs in enumerate(neigbs_list)]
                # updating anchor lists
                for local_idx, global_idx in enumerate(tgt_idx):
                    nearest_ancs[global_idx].extend(ancs[local_idx])
                    anc_dists[global_idx].extend([hop for _ in range(len(ancs[local_idx]))])

                hop += 1
                if hop >= 50:  # hardcoded constant for a disconnected node
                    for idx in tgt_idx:
                        nearest_ancs[idx].extend([self.NOTHING_TOKEN for _ in range(limit - len(nearest_ancs[idx]))])
                        anc_dists[idx].extend([0 for _ in range(limit - len(anc_dists[idx]))])
                        break

            # update the vocab
            for idx, v in enumerate(batch_verts):
                vocab[v] = {'ancs': nearest_ancs[idx][:limit], 'dists': anc_dists[idx][:limit]}

        return vocab

    def mine_parallel(self, anchors):
        from tqdm.contrib.concurrent import process_map
        # let's try splitting the graph into connected components with METIS and run mining on them
        src, tgt = self.triples_factory.mapped_triples['head'], self.triples_factory.mapped_triples['tail']
        edgelist = [[s, t] for s, t in zip(src, tgt)]
        pyg_graph = Data(edge_index=torch.tensor(edgelist).T, num_nodes=self.triples_factory.num_entities)
        print(f"Using METIS to partition the graph into {self.partition} partitions")
        clusters = ClusterData(pyg_graph, num_parts=self.partition)

        vocab = {}
        data_points = []

        # now find anchors in each cluster and tokenize clusters one by one
        for cluster_id in range(len(clusters)):
            start = int(clusters.partptr[cluster_id])
            end = int(clusters.partptr[cluster_id + 1])
            cluster_nodes = clusters.perm[start: end]
            cluster_anchors = list(set(cluster_nodes.cpu().numpy()).intersection(set(anchors)))

            # map global anchor IDs to local ids [they start from 0 to num_nodes in cluster]
            local_anchor_ids = []
            for anc_idx, anc in enumerate(cluster_anchors):
                local_id = (cluster_nodes == anc).nonzero(as_tuple=False)
                local_anchor_ids.append(local_id.item())

            node_mapping = {i: n.item() for i, n in enumerate(cluster_nodes)}
            anchor_mapping = {loc_id: glob_id for loc_id, glob_id in zip(local_anchor_ids, cluster_anchors)}
            anchor_mapping[-99] = -99

            #cluster = clusters[cluster_id]

            data_points.append({
                'clusters': clusters,
                'node_mapping': node_mapping,
                'anchor_mapping': anchor_mapping,
                'local_anchor_id': local_anchor_ids,
                'tqdm_pos': cluster_id + 1,  # we have an outer loop that starts with 0
            })

        all_batches = process_map(self.mining_subp, data_points, max_workers=self.cpus)
        for d in all_batches:
            vocab.update(d)

        # sort vocab by key - need it to be in the ascending order 0 - n
        vocab = dict(sorted(vocab.items()))
        return vocab

    def mining_subp(self, data_point):
        clusters, node_mapping, anchor_mapping, local_anchor_ids, tqdm_pos = data_point.values()
        cluster = clusters[tqdm_pos - 1]
        cl_vocab = self.bfs_cluster(cluster, local_anchor_ids, tqdm_pos)

        # re-map back to global ids
        cl_vocab = {node_mapping[k]: {
            'ancs': [anchor_mapping[a] for a in v['ancs']],
            'dists': v['dists']
        } for k, v in cl_vocab.items()}

        return cl_vocab
