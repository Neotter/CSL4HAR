'''
Date: 2022-08-07 16:18:59
LastEditors: MonakiChen
LastEditTime: 2022-08-08 14:08:46
FilePath: \CSL4HAR\trainer\loss_fn.py
'''
import torch
from itertools import chain
import torch.nn.functional as F
import torch.nn as nn

def info_nce_loss(y_pred, tau): 
    y_pred = y_pred.view([y_pred.shape[0],-1])
    # ids is y positive, y_true is label
    ids = torch.arange(0, y_pred.shape[0]).cuda()
    y_true = ids + 1 - ids % 2 * 2 #[1,2,3,4,5,6] -> [2,1,4,3,6,5]
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2) # shape=[batch_size, batch_size]
    # 屏蔽对角矩阵，即自身相等的loss
    similarities = similarities - torch.eye(y_pred.shape[0]).cuda() * 1e12
    similarities = similarities / tau
    return torch.mean(F.cross_entropy(similarities, y_true))

def mse_loss():
    return nn.MSELoss(reduction='none')

def ce_loss():
    return nn.CrossEntropyLoss()