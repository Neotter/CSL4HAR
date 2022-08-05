'''
Date: 2022-04-13 17:05:36
LastEditors: MonakiChen
LastEditTime: 2022-06-08 09:54:58
FilePath: \S3IMU\CODE\model\bert.py
'''
import torch
import torch.nn as nn
from model.bert import BERT
from model.func_loss import mse_loss
import torch.nn.functional as F

from typing import NamedTuple
import json

class LIBERTConfig(NamedTuple):
    "Configuration for LIBERT model"
    input_dim: int = 6 # Raw Data Dimension
    embed_dim: int = 72  # Embedding Dimension
    hidden_dim: int = 72  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    ff_hidden_dim: int = 144  # Factorized embedding parameterization
    n_layers: int = 4  # Numher of libert Hidden Layers
    n_heads: int = 4  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 120  # Maximum Length for Positional Embeddings
    prob_drop_hidden: float = 0.1
    p_drop_attn: float = 0.1
    embed_norm: bool = True # Switch of embedding normalization
    fix_bert: bool = False # Switch of fixing BERT parameters

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class LIBERT(nn.Module):

    def __init__(self, cfg, input_dim, embed_dim, following_model = None):
        super().__init__()
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
            return self.calc_mse_loss(*input)
        if mode == 'pretrain_predict':
            return self.pretrain_predict(*input)
        if mode == 'train':
            return self.calc_ce_loss(*input)
        if mode == 'predict':
            return self.predict(*input)
        if mode == 'eval':
            return self.calc_eva_loss(*input)

    def pretrain_predict(self, input_seqs, masked_pos=None):
        h_masked = self.bert(input_seqs)
        if not self.output_layer:
            return h_masked
        if masked_pos is not None:
            # [128, 18] => [128, 18, 1] => [128, 18, 72]
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            # [128, 120, 72] => [128, 18, 72]
            # gather会把h_masked变成masked_pos的样子
            # masked_pos表示被遮盖的18个点的位置, gather从h_masked这个embedding中找到对应的位置
            # 的embedding用于predict输出
            # 在原始的h_masked中每一行有72列, 而masked_pos从有18被遮盖的点,
            h_masked = torch.gather(h_masked, 1, masked_pos)
        logits_lm = self.output_layer(h_masked)
        return logits_lm

    def predict(self, input_seqs, training=False): #, training
        h = self.bert(input_seqs)
        h = self.output_layer(h, training)
        return h

    def calc_mse_loss(self, batch):
        mask_seqs, masked_pos, seqs = batch 
        seq_recon = self.pretrain_predict(mask_seqs, masked_pos)
        loss_lm = self.pretrain_criterion(seq_recon, seqs) 
        return loss_lm

    def calc_ce_loss(self, batch):
        inputs, label = batch
        logits = self.predict(inputs, True)
        loss = self.train_criterion(logits, label) 
        return loss

    def calc_eva_loss(self, seqs, predict_seqs):
        loss_lm = self.pretrain_criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    def calc_info_nce_loss(self, batch, tau=0.05):
        inputs, label = batch
        y_pred = self.pretrain_predict(inputs)
        ids = torch.arange(0, y_pred.shape[0])
        y_true = ids + 1 - ids % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        # 屏蔽对角矩阵，即自身相等的loss
        similarities = similarities - torch.eye(y_pred.shape[0]) * 1e12
        similarities = similarities / tau
        return torch.mean(F.cross_entropy(similarities, y_true))

    
    def load_self(self, model_file, map_location=None):
            state_dict = self.state_dict()
            model_dicts = torch.load(model_file, map_location=map_location).items()
            for k, v in model_dicts:
                if k in state_dict:
                    state_dict.update({k: v})
            self.load_state_dict(state_dict)



