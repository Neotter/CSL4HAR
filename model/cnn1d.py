import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.seq_len, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool1d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = self.pool(h)
            # h = bn(h)
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h