import torch
import torch.nn as nn
from model.func_activ import gelu
from model.layer_norm import LayerNorm

class MLP(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activ = gelu
        self.norm = LayerNorm(hidden_dim)
        self.fin_linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, h_masked):
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.fin_linear(h_masked)
        return logits_lm

