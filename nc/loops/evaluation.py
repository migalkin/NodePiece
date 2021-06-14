import numpy as np

# Local
from utils.utils import *
from utils.utils_mytorch import Timer
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score, accuracy_score


def compute_roc_auc(y_true, y_pred):
    """

    :param y_true: true labels, shape (n_samples, n_classes)
    :param y_pred: predicted values, shape (n_samples, n_classes)
    :return: roc_auc_score
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))
    #score = roc_auc_score(y_true, y_pred)
    return sum(rocauc_list)/len(rocauc_list) if len(rocauc_list) > 0 else 0


def compute_prcauc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    prcauc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            precision, recall, _ = precision_recall_curve(y_true[is_labeled, i], y_pred[is_labeled, i])
            prcauc = auc(recall, precision)
            prcauc_list.append(prcauc)

    return sum(prcauc_list)/len(prcauc_list) if len(prcauc_list) > 0 else 0


def compute_average_precision(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ap_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)

    return sum(ap_list) / len(ap_list) if len(ap_list) > 0 else 0


def hard_accuracy(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy().round()
    return accuracy_score(y_true, y_pred)


def eval_classification(y_true, y_pred):
    rocauc = compute_roc_auc(y_true, y_pred)
    prcauc = compute_prcauc(y_true, y_pred)
    ap = compute_average_precision(y_true, y_pred)
    hard_acc = hard_accuracy(y_true, y_pred)

    return {"rocauc": rocauc, "prcauc": prcauc, "ap": ap, "hard_acc": hard_acc}


if __name__ == "__main__":
    print("smth")
