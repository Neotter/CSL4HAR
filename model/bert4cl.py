'''
Date: 2022-04-13 17:05:36
LastEditors: MonakiChen
LastEditTime: 2022-08-09 16:52:04
FilePath: \CSL4HAR\model\bert4cl.py
'''
import json
from typing import NamedTuple
import torch
import torch.nn as nn
from model.bert import BERT
from model.func_loss import mse_loss
import torch.nn.functional as F
from itertools import chain

class BERT4CLConfig(NamedTuple):
    "Configuration for BERT4CL model"
    input_dim: int = 6 # Raw Data Dimension
    embed_dim: int = 72  # Embedding Dimension
    hidden_dim: int = 72  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    ff_hidden_dim: int = 144  # Factorized embedding parameterization
    n_layers: int = 4  # Numher of BERT4CL Hidden Layers
    n_heads: int = 4  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 120  # Maximum Length for Positional Embeddings
    prob_drop_hidden: float = 0.1
    p_drop_attn: float = 0.1
    embed_norm: bool = True # Switch of embedding normalization
    fix_bert: bool = False # Switch of fixing BERT parameters

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class BERT4CL(nn.Module):

    def __init__(self, cfg, input_dim, embed_dim, loss_fn = None, following_model = None):
        super().__init__()
        # self.augmentation = cfg.augmentation
        self.bert = BERT(cfg, input_dim, embed_dim) # encoder
        # pretrain: fix_bert == false
        # classifier train: fix_bert == true
        if cfg.fix_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.output_layer = following_model

        self.loss_fn = loss_fn
        
    def forward(self, *input, mode):
        if mode == 'pretrain':
            return self.calc_pretrain_loss(*input)
        if mode == 'predict':
            return self.predict(*input)
        
    def predict(self, input_seqs):
        embeddings = self.bert(input_seqs)
        if not self.output_layer: # output embeddings
            return embeddings
        y_pred = self.output_layer(embeddings)
        return y_pred

    def calc_pretrain_loss(self, batch, tau=0.05):
        x_anchor, x_positive_1, x_positive_2 = batch
        x_anchor = x_positive_1
        x_positive = x_positive_2
        x_pair = torch.stack(list(chain.from_iterable(zip(x_anchor, x_positive))))
        y_pred = self.predict(x_pair)

        return self.loss_fn(y_pred,tau)

    
    def load_self(self, model_file, map_location=None):
            state_dict = self.state_dict()
            model_dicts = torch.load(model_file, map_location=map_location).items()
            for k, v in model_dicts:
                if k in state_dict:
                    state_dict.update({k: v})
            self.load_state_dict(state_dict)


