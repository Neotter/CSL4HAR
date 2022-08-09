'''
Date: 2022-08-05 20:13:48
LastEditors: MonakiChen
LastEditTime: 2022-08-09 14:43:07
FilePath: \CSL4HAR\model\decoder_projhead.py
'''
import torch
import torch.nn as nn
from model.func_activ import gelu
from model.layer_norm import LayerNorm

class ProjectHead(nn.Module):
    def __init__(self, cfg, input_dim, output_dim, meanPooling=False):
        super().__init__()
        hidden_dim = cfg.encoder_cfg.ff_hidden_dim

        self.first_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm_layer = nn.LayerNorm(hidden_dim)
        self.actv = gelu
        self.final_layer = nn.Linear(hidden_dim, output_dim, bias=True)

        self.meanPooling=meanPooling

    def forward(self, embeddings):
        h = self.first_layer(embeddings)
        h = self.norm_layer(h)
        h = self.actv(h)
        proj_embeddings = self.final_layer(h)
        # if the length of augmented instances is different
        if self.meanPooling == True:
            return proj_embeddings.mean(1)
        return proj_embeddings

