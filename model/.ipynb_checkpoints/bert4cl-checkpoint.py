'''
Date: 2022-04-13 17:05:36
LastEditors: MonakiChen
LastEditTime: 2022-07-25 13:27:21
FilePath: \CODE_V1\model\bert4cl.py
'''
import torch
import torch.nn as nn
from model.bert import BERT
from model.func_loss import mse_loss
import torch.nn.functional as F
from itertools import chain



class BERT4CL(nn.Module):

    def __init__(self, cfg, input_dim, embed_dim, following_model = None):
        super().__init__()
        # self.augmentation = cfg.augmentation
        self.bert = BERT(cfg, input_dim, embed_dim) # encoder
        # pretrain: fix_bert == false
        # classifier train: fix_bert == true
        if cfg.fix_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.output_layer = following_model
        self.pretrain_criterion = nn.MSELoss(reduction='none')
        self.train_criterion = nn.CrossEntropyLoss()
        
    def forward(self, *input, mode):

        if mode == 'pretrain':
            return self.calc_info_nce_loss(*input)
        if mode == 'pretrain_predict':
            return self.pretrain_predict(*input)

        if mode == 'train':
            return self.calc_ce_loss(*input)
        if mode == 'predict':
            return self.predict(*input)

    def pretrain_predict(self, input_seqs):
        h = self.bert(input_seqs)
        if not self.output_layer:
            return h
        # h = h.mean(1) ######
        embedding = self.output_layer(h)
        # if self.augmentation == 'clipping':
        embedding = embedding.mean(1)
        return embedding

    def predict(self, input_seqs, training=False): #, training
        h = self.bert(input_seqs)
        h = self.output_layer(h, training)
        return h

    def calc_ce_loss(self, batch):
        inputs, label = batch
        logits = self.predict(inputs, True)
        loss = self.train_criterion(logits, label) 
        return loss

    def calc_info_nce_loss(self, batch, tau=0.05):
        # for spanmasking
        # if self.augmentation == 'spanmasking' or self.augmentation == 'delwords':
        y_anchor, y_positive_1, y_positive_2 = batch
        y_anchor = y_positive_1
        y_positive = y_positive_2
        input = torch.stack(list(chain.from_iterable(zip(y_anchor, y_positive))))
        y_pred = self.pretrain_predict(input)
        y_pred = y_pred.view([y_pred.shape[0],-1])

        # for clipping and delwords
        # if self.augmentation == 'clipping':
        # y_anchor, y_positive = batch
        # y_anchor_pred = self.pretrain_predict(y_anchor)
        # y_anchor_pred = y_anchor_pred.view([y_anchor_pred.shape[0],-1])
        # y_positive_pred = self.pretrain_predict(y_positive)
        # y_positive_pred = y_positive_pred.view([y_positive_pred.shape[0],-1])
        # y_pred = torch.stack(list(chain.from_iterable(zip(y_anchor_pred, y_positive_pred))))
        # y_pred = y_pred.view([y_pred.shape[0],-1])

        # ids is y positive, y_true is label
        ids = torch.arange(0, y_pred.shape[0]).cuda()
        y_true = ids + 1 - ids % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        # similarities = self.pretrain_criterion(y_pred.unsqueeze(1), y_pred.unsqueeze(0))
        # 屏蔽对角矩阵，即自身相等的loss
        similarities = similarities - torch.eye(y_pred.shape[0]).cuda() * 1e12
        similarities = similarities / tau
        return torch.mean(F.cross_entropy(similarities, y_true))
    
    def load_self(self, model_file, map_location=None):
            state_dict = self.state_dict()
            model_dicts = torch.load(model_file, map_location=map_location).items()
            for k, v in model_dicts:
                if k in state_dict:
                    state_dict.update({k: v})
            self.load_state_dict(state_dict)


