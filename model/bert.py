import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer_norm import LayerNorm
from model.func_activ import gelu

class BERT(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()
        # Entry
        self.embed = Embeddings(cfg,input_dim)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm1 = LayerNorm(cfg.hidden_dim)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(output_dim)
        self.drop = nn.Dropout(cfg.prob_drop_hidden)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h

class Embeddings(nn.Module):

    def __init__(self, cfg, input_dim, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(input_dim, cfg.hidden_dim)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden_dim) 
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg.hidden_dim)
        self.emb_norm = True

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        ########################
        self.drop = nn.Dropout(cfg.p_drop_attn)
        ########################
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        ########################
        scores = self.drop(F.softmax(scores, dim=-1))
        ########################
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = self.merge_last(h, 2)
        self.scores = scores
        return h

    def split_last(self, x, shape):
        "split the last dimension to given shape"
        shape = list(shape)
        # 统计-1出现的次数
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
        return x.view(*x.size()[:-1], *shape)

    def merge_last(self, x, n_dims):
        "merge the last n_dims to a dimension"
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ff_hidden_dim)
        self.fc2 = nn.Linear(cfg.ff_hidden_dim, cfg.hidden_dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))
