import torch
import torch.nn as nn
from model.func_activ import gelu
from model.layer_norm import LayerNorm

class ProjectHead(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()
        hidden_dim = cfg.model_cfg.ff_hidden_dim

        self.first_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm_layer = nn.LayerNorm(hidden_dim)
        self.actv = gelu
        self.final_layer = nn.Linear(hidden_dim, output_dim, bias=True)


    def forward(self, embedding):
        h = self.first_layer(embedding)
        h = self.norm_layer(h)
        h = self.actv(h)
        proj_embedding = self.final_layer(h)


        return proj_embedding

