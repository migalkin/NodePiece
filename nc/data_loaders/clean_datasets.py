from pathlib import Path
from typing import Dict
from collections import defaultdict
import random
import pickle
import numpy as np
import re
import json
import torch

from torch_geometric.data import Data, DataLoader


def to_sparse_graph(edges, subtype, entoid, prtoid, maxlen):

    edge_index, edge_type = np.zeros((2, len(edges)), dtype='int64'), np.zeros((len(edges)), dtype='int64')
    qualifier_rel = []
    qualifier_ent = []
    qualifier_edge = []
    quals = None

    for i, st in enumerate(edges):
        edge_index[:, i] = [entoid[st[0]], entoid[st[2]]]
        edge_type[i] = prtoid[st[1]]

        if subtype == 'statements':
            qual_rel = np.array([prtoid[r] for r in st[3::2]])[
                       :(maxlen - 3) // 2]  # cut to the max allowed qualifiers per statement
            qual_ent = np.array([entoid[e] for e in st[4::2]])[
                       :(maxlen - 3) // 2]  # cut to the max allowed qualifiers per statement
            for j in range(qual_ent.shape[0]):
                qualifier_rel.append(qual_rel[j])
                qualifier_ent.append(qual_ent[j])
                qualifier_edge.append(i)

    if subtype == 'statements':
        quals = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)

    return edge_index, edge_type, quals


def load_clean_pyg(name, subtype, task, inductive="transductive", ind_v=None, maxlen=43, permute=False) -> Dict:
    """
    :param name: dataset name wd50k/wd50k_33/wd50k_66/wd50k_100
    :param subtype: triples/statements
    :param task: so/full predict entities at sub/obj positions (for triples/statements) or all nodes incl quals
    :param inductive: whether to load transductive dataset (one graph for train/val/test) or inductive
    :param ind_v: v1 / v2 for the inductive dataset
    :param maxlen: max statement length
    :return: train/valid/test splits for the wd50k datasets suitable for loading into TORCH GEOMETRIC dataset
    no reciprocal edges (as will be added in the gnn layer), create directly the edge index
    """
    assert name in ['wd50k', 'wd50k_100', 'wd50k_33', 'wd50k_66'], "Incorrect dataset"
    assert subtype in ["triples", "statements"], "Incorrect subtype: triples/statements"
    assert inductive in ["transductive", "inductive"], "Incorrect ds type: only transductive and inductive accepted"
    if inductive == "inductive":
        assert ind_v in ["v1", "v2"], "Only v1 and v2 are allowed versions for the inductive task"

    if inductive == "transductive":
        DIRNAME = Path(f'./data/clean/{name}/{subtype}')
        train_edges = [line.strip("\n").split(",") for line in open(DIRNAME / 'nc_edges.txt', 'r').readlines()]
        print(f"Transductive: With quals: {len([t for t in train_edges if len(t)>3])} / {len(train_edges)}, Ratio: {round((len([t for t in train_edges if len(t)>3]) / len(train_edges)),2)}")
    else:
        DIRNAME = Path(f'./data/clean/{name}/inductive/nc/{subtype}/{ind_v}')
        train_edges = [line.strip("\n").split(",") for line in open(DIRNAME / 'nc_train_edges.txt', 'r').readlines()]
        val_edges = [line.strip("\n").split(",") for line in open(DIRNAME / 'nc_val_edges.txt', 'r').readlines()]
        test_edges = [line.strip("\n").split(",") for line in open(DIRNAME / 'nc_test_edges.txt', 'r').readlines()]
        print(
            f"Inductive train: With quals: {len([t for t in train_edges if len(t) > 3])} / {len(train_edges)}, Ratio: {round((len([t for t in train_edges if len(t) > 3]) / len(train_edges)), 2)}")
        print(
            f"Inductive val: With quals: {len([t for t in val_edges if len(t) > 3])} / {len(val_edges)}, Ratio: {round((len([t for t in val_edges if len(t) > 3]) / len(val_edges)), 2)}")
        print(
            f"Inductive test: With quals: {len([t for t in test_edges if len(t) > 3])} / {len(test_edges)}, Ratio: {round((len([t for t in test_edges if len(t) > 3]) / len(test_edges)), 2)}")

    statement_entities = [l.strip("\n") for l in open(DIRNAME / 'nc_entities.txt', 'r').readlines()]
    statement_predicates = [l.strip("\n") for l in open(DIRNAME / 'nc_rels.txt', 'r').readlines()]

    if subtype == "triples":
        task = "so"

    with open(DIRNAME / f'nc_train_{task}_labels.json', 'r') as f:
        train_labels = json.load(f)

    with open(DIRNAME / f'nc_val_{task}_labels.json', 'r') as f:
        val_labels = json.load(f)

    with open(DIRNAME / f'nc_test_{task}_labels.json', 'r') as f:
        test_labels = json.load(f)

    # load node features with the total index
    entity_index = {line.strip('\n'): i for i, line in enumerate(open(f'./data/clean/{name}/statements/{name}_entity_index.txt').readlines())}
    total_node_features = np.load(f'./data/clean/{name}/statements/{name}_embs.pkl', allow_pickle=True)

    idx = np.array([entity_index[key] for key in statement_entities], dtype='int32')
    node_features = total_node_features[idx]

    if permute:
        node_features = np.random.permutation(node_features)

    entoid = {pred: i for i, pred in enumerate(statement_entities)}
    prtoid = {pred: i for i, pred in enumerate(statement_predicates)}

    print(f"Total Entities: {len(entoid)}")
    print(f"Total Rels: {len(prtoid)}")

    train_edge_index, train_edge_type, train_quals = to_sparse_graph(train_edges, subtype, entoid, prtoid, maxlen)
    if inductive == "inductive":
        val_edge_index, val_edge_type, val_quals = to_sparse_graph(val_edges, subtype, entoid, prtoid, maxlen)
        test_edge_index, test_edge_type, test_quals = to_sparse_graph(test_edges, subtype, entoid, prtoid, maxlen)


    train_mask = [entoid[e] for e in train_labels]
    val_mask = [entoid[e] for e in val_labels]
    test_mask = [entoid[e] for e in test_labels]

    if inductive == "inductive":
        print(f"Train Ents: {len(train_labels)}, Val Ents: {len(val_labels)}, Test Ents: {len(test_labels)}")

    all_labels = sorted(list(set([
        label for v in list(train_labels.values()) + list(val_labels.values()) + list(test_labels.values()) for label in
        v])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}
    print(f"Total labels: {len(label2id)}")

    train_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in train_labels.items()}
    val_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in val_labels.items()}
    test_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in test_labels.items()}

    train_graph = Data(x=torch.tensor(node_features, dtype=torch.float),
                       edge_index=torch.tensor(train_edge_index, dtype=torch.long),
                       edge_type=torch.tensor(train_edge_type, dtype=torch.long),
                       quals=torch.tensor(train_quals, dtype=torch.long) if train_quals is not None else None, y=train_y)

    # explicit fix for PyG > 2.0 as its new data creation procedure uses setattr(obj, key, value) which does not
    # assign None values, such that quals=None does not lead to creating the `quals` with value None
    # so we explicitly set it to 0 - it won't affect any code execution whatsoever
    if train_quals is None:
        train_graph.quals = 0
    val_graph, test_graph = None, None



    if inductive == "inductive":
        val_graph = Data(x=torch.tensor(node_features, dtype=torch.float),
                         edge_index=torch.tensor(val_edge_index, dtype=torch.long),
                         edge_type=torch.tensor(val_edge_type, dtype=torch.long),
                         quals=torch.tensor(val_quals, dtype=torch.long) if val_quals is not None else None, y=val_y)
        test_graph = Data(x=torch.tensor(node_features, dtype=torch.float),
                          edge_index=torch.tensor(test_edge_index, dtype=torch.long),
                          edge_type=torch.tensor(test_edge_type, dtype=torch.long),
                          quals=torch.tensor(test_quals, dtype=torch.long) if test_quals is not None else None, y=test_y)

    return {"train_graph": train_graph, "val_graph": val_graph, "test_graph": test_graph,
            "train_mask": train_mask, "valid_mask": val_mask, "test_mask": test_mask,
            "train_y": train_y, "val_y": val_y, "test_y": test_y,
            "all_labels": all_labels, "label2id": label2id, "id2label": id2label,
            "n_entities": len(statement_entities), "n_relations": len(statement_predicates),
            "e2id": entoid, "r2id": prtoid}



if __name__ == "__main__":
    data = load_clean_pyg('wd50k', 'statements', 'so', inductive="inductive", ind_v='v2', maxlen=15)
    print("nop")
    #count_stats(load_clean_wd50k("wd50k","statements",43))