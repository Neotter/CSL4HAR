'''
Date: 2022-04-13 17:05:36
LastEditors: MonakiChen
LastEditTime: 2022-06-08 09:54:58
FilePath: \S3IMU\CODE\model\bert.py
'''
import torch.nn as nn
import torch.nn.functional as F

from typing import NamedTuple
import json

class GRUConfig(NamedTuple):
    "Configuration for LIBERT model"
    seq_len: int = 20  # Maximum Length for Positional Embeddings
    input_dim: int = 6 # GRU Input Dimension
    embed_dim = input_dim
    num_rnn: int = 2  # number of rnn
    num_layers: str = "[2, 1]"  # Factorized embedding parameterization
    rnn_io: int = "[[6,20], [20, 10]]"  # Numher of libert Hidden Layers
    num_linear: int = 1  # Numher of Heads in Multi-Headed Attention Layers
    linear_io: str = "[[10, 3]]" # Switch of embedding normalization
    activ: bool = False # Switch of fixing BERT parameters
    dropout: bool = False

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class GRU(nn.Module):
    def __init__(self, cfg, input_dim=None, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_rnn = cfg.num_rnn
        self.num_layers = eval(cfg.num_layers)
        self.rnn_io = eval(cfg.rnn_io)
        self.linear_io = eval(cfg.linear_io)
        self.num_linear = cfg.num_linear
        
        self.activ = cfg.activ
        self.dropout = cfg.dropout

        for i in range(self.num_rnn):
            if input_dim is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input_dim, self.rnn_io[i][1], num_layers=self.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(self.rnn_io[i][0], self.rnn_io[i][1], num_layers=self.num_layers[i],
                                         batch_first=True))

        for i in range(self.num_linear):
            if output_dim is not None and i == self.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(self.linear_io[i][0], output_dim))
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
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu
        return h

    def calc_ce_loss(self, batch):
        inputs, label = batch
        logits = self.predict(inputs, training=True)
        ce_loss = self.criterion(logits, label)
        return ce_loss

    