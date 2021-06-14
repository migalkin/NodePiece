""" Some things needed across the board"""
import torch
import numpy as np
from typing import Optional, List, Union, Dict, Callable, Tuple


KNOWN_DATASETS = ['wd50k']


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def combine(*args: Union[np.ndarray, list, tuple]):
    """
        Used to semi-intelligently combine data splits

        Case A)
            args is a single element, an ndarray. Return as is.
        Case B)
            args are multiple ndarray. Numpy concat them.
        Case C)
            args is a single dict. Return as is.
        Case D)
            args is multiple dicts. Concat individual elements

    :param args: (see above)
    :return: A nd array or a dict
    """

    # Case A, C
    if len(args) == 1 and type(args[0]) is not dict:
        return np.array(args[0])

    if len(args) == 1 and type(args) is dict:
        return args

    # Case B
    if type(args) is tuple and (type(args[0]) is np.ndarray or type(args[0]) is list):
        # Expected shape will be a x n, b x n. Simple concat will do.
        return np.concatenate(args)

    # Case D
    if type(args) is tuple and type(args[0]) is dict:
        keys = args[0].keys()
        combined = {}
        for k in keys:
            combined[k] = np.concatenate([arg[k] for arg in args], dim=-1)
        return combined


def print_results(traces: List):
    """ traces: List of Dicts of
    "loss": train_loss,
    "train_rocauc:": train_rocauc,
    "train_prcauc": train_prcauc,
    "train_ap": train_ap,
    "train_hard_acc": train_hard_acc,
    "valid_rocauc": valid_rocauc,
    "valid_prcauc": valid_prcauc,
    "valid_ap": valid_ap,
    "valid_hard_acc": valid_hard_acc
    """
    for key in ["train_rocauc", "train_prcauc", "train_ap", "train_hard_acc",
                "valid_rocauc", "valid_prcauc", "valid_ap", "valid_hard_acc"]:
        result = np.array([t[key][-1] for t in traces])
        print(f"Avg {key}: {np.mean(result): .3f}, std dev: {np.std(result):.3f}")
