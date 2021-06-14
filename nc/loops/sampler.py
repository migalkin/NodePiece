import numpy as np
import torch

from typing import Union

class NodeClSampler:
    """
    The sampler for multi-label node classification tasks. Each node has k labels
    The graph is processed entirely, so there are no batches.
    The output is [n_entities, n_labels]
    """
    def __init__(self, data: Union[np.array, dict], num_labels: int,
                 label2id: dict, lbl_smooth: float = 0.0):
        """

        :param data: data as an array of statements of STATEMENT_LEN, e.g., [0,0,0] or [0,1,0,2,4]
        :param n_entities: total number of entities
        :param lbl_smooth: whether to apply label smoothing used later in the BCE loss
        :param bs: batch size
        :param with_q: whether indexing will consider qualifiers or not, default: FALSE
        """

        self.data = data
        self.lbl_smooth = lbl_smooth
        self.num_labels = num_labels
        self.label2id = label2id

        self.train_mask, self.train_y, self.eval_mask, self.eval_y = self.generate_labels()

    def generate_labels(self):
        train = self.data["train"]  # node_id: [lab1, lab2, lab3]
        eval = self.data["eval"]

        train_y = np.zeros((len(train), self.num_labels), dtype=np.float32)
        eval_y = np.zeros((len(eval), self.num_labels), dtype=np.float32)
        train_mask = np.zeros(len(train), dtype=np.long)
        eval_mask = np.zeros(len(eval), dtype=np.long)

        for i, (node, labels) in enumerate(train.items()):
            lbls = np.zeros((1, self.num_labels))
            for l in labels:
                lbls[0, l] = 1.0
            train_y[i] = lbls
            train_mask[i] = node

        self.pos_weights = self.compute_weights(train_y)

        if self.lbl_smooth != 0.0:
            train_y = (1.0 - self.lbl_smooth)*train_y + (1.0 / len(train))

        for i, (node, labels) in enumerate(eval.items()):
            lbls = np.zeros((1, self.num_labels))
            for l in labels:
                lbls[0, l] = 1.0
            eval_y[i] = lbls
            eval_mask[i] = node

        return train_mask, train_y, eval_mask, eval_y

    def compute_weights(self, data):
        class_counts = data.sum(axis=0)
        pos_weights = np.ones_like(class_counts)
        neg_counts = np.array([len(data) - pos_count for pos_count in class_counts])  # <-- HERE
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)

        return torch.as_tensor(pos_weights, dtype=torch.float)

    def get_data(self):
        return self.train_mask, self.train_y, self.eval_mask, self.eval_y