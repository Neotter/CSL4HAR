'''
Date: 2022-08-05 20:13:48
LastEditors: MonakiChen
LastEditTime: 2022-08-09 16:52:21
FilePath: \CSL4HAR\model\lstm.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import NamedTuple
import json

class LSTMConfig(NamedTuple):
    "Configuration for BERT4CL model"
    seq_len: int = 20  # Maximum Length for Positional Embeddings
    input_dim: int = 6 # GRU Input Dimension
    embed_dim = input_dim
    num_rnn: int = 2  # number of rnn
    num_layers: str = "[2, 1]"  # Factorized embedding parameterization
    rnn_io: int = "[[6,20], [20, 10]]"  # Numher of BERT4CL Hidden Layers
    num_linear: int = 1  # Numher of Heads in Multi-Headed Attention Layers
    linear_io: str = "[[10, 3]]" # Switch of embedding normalization
    activ: bool = False # Switch of fixing BERT parameters
    dropout: bool = False

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class LSTM(nn.Module):
    def __init__(self, cfg, input_dim=None, output_dim=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seq_len = cfg.seq_len
        self.num_rnn = cfg.num_rnn
        self.num_layers = eval(cfg.num_layers)
        self.rnn_io = eval(cfg.rnn_io)
        self.linear_io = eval(cfg.linear_io)
        self.num_linear = cfg.num_linear
        
        self.activ = cfg.activ
        self.dropout = cfg.dropout

        for i in range(self.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), nn.LSTM(self.input_dim, self.rnn_io[i][1], num_layers=self.num_layers[i], batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                 nn.LSTM(self.rnn_io[i][0], self.rnn_io[i][1], num_layers=self.num_layers[i],
                                         batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(self.seq_len))
        for i in range(self.num_linear):
            if self.output_dim is not None and i == self.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(self.linear_io[i][0], self.output_dim))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(self.linear_io[i][0], self.linear_io[i][1]))
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *input, mode):
        if mode == 'train':
            return self.calc_ce_loss(*input)
        if mode == 'predict':
            return self.predict(*input)

    def predict(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h

    def calc_ce_loss(self, batch):
        inputs, label = batch
        logits = self.predict(inputs, training=True)
        ce_loss = self.criterion(logits, label)
        return ce_loss