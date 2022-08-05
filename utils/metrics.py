from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sn
import torch


def calc_acc(y_hat, y):
    acc = np.sum(y == y_hat) / y.size
    return acc


def calc_f1(y_hat, y):
    f1 = f1_score(y, y_hat, average='macro')
    return f1

def calc_confus_matric(y_hat, y):
    confus_matrix = confusion_matrix(y, y_hat)
    return confus_matrix

def plot_matrix(matrix, labels_name=None):
    plt.figure()
    row_sum = matrix.sum(axis=1)
    matrix_per = np.copy(matrix).astype('float')
    for i in range(row_sum.size):
        if row_sum[i] != 0:
            matrix_per[i] = matrix_per[i] / row_sum[i]
    # plt.figure(figsize=(10, 7))
    if labels_name is None:
        labels_name = "auto"
    sn.heatmap(matrix_per, annot=True, fmt='.2f', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # plt.savefig()
    return matrix

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()