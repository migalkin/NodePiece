# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

import numpy as np
import torch


def save_model(
        model,
        model_name,
        emb_method,
        dataset_name,
        chkpnt,
        lr,
        reg_lambda,
        neg_ratio,
        emb_dim,
):
    print("Saving the model")
    directory = "logs/" + model_name + "/" + emb_method + "/" + dataset_name + "/"
    os.makedirs(directory, exist_ok=True)
    torch.save(
        model,
        directory
        + str(chkpnt)
        + "_"
        + str(lr)
        + "_"
        + str(reg_lambda)
        + "_"
        + str(neg_ratio)
        + "_"
        + str(emb_dim)
        + ".chkpnt",
    )


def random_new_ent_mask(triples, mask_prob):
    """
    Assign a random mask value to each triplet based on the given probability.
    mask_prob here corresponds to (1-psi) in Algorithm 2 (https://arxiv.org/pdf/2004.13230.pdf)
    if mask = 0, the head of the triplet is considered as an unobserved entity
    if mask = 2, the tail of the triplet is considered as an unobserved entity
    if mask = 1, both head and tail are observed
    """
    probs = [(1 - mask_prob) / 2, mask_prob, (1 - mask_prob) / 2]
    return np.random.choice([0, 1, 2], triples.shape[0], p=probs)
