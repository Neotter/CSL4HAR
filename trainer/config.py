'''
Date: 2022-08-05 20:13:49
LastEditors: MonakiChen
LastEditTime: 2022-08-09 14:05:48
FilePath: \CSL4HAR\trainer\config.py
'''
from typing import NamedTuple
import json

class ConfigBase(NamedTuple):
    "Configuration template"
    batch_size: int = 512 # pretrain batch size.
    n_epochs: int = 3200 # number of epochs
    lr: float = 1e-3 # learning rate.
    saving_epoch: int = 100 # interval epoch for saving model
    early_stopping_epoch: int = 0 # number of epoch for early stopping
    testing_epoch: int = 1 # interval epoch of testing pretrain.
    training_rate: float = 0.8 # training rate.
    vali_rate: float = 0.1 # validation rate.
    label_index: int = 0 # label index, 0 is activate, 2 is user id, 3 is model', choices=[0, 1, 2]
    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class PretrainConfig(NamedTuple):
    "Configuration for Pretrain phase"
    batch_size: int = 512 # pretrain batch size.
    n_epochs: int = 3200 # number of epochs
    lr: float = 1e-3 # learning rate.
    saving_epoch: int = 100 # interval epoch for saving model
    early_stopping_epoch: int = 0 # number of epoch for early stopping
    testing_epoch: int = 1 # interval epoch of testing pretrain.
    training_rate: float = 0.8 # training rate.
    vali_rate: float = 0.1 # validation rate.
    label_index: int = 0 # label index, 0 is activate, 2 is user id, 3 is model', choices=[0, 1, 2]
    ##############################################################

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class TrainConfig(NamedTuple):
    "Configuration for train phase"
    batch_size: int = 512 # pretrain batch size.
    n_epochs: int = 700 # number of epochs
    lr: float = 1e-3 # learning rate.
    saving_epoch: int = 100 # interval epoch for saving model
    early_stopping_epoch: int = 0 # number of epoch for early stopping
    testing_epoch: int = 1 # interval epoch of testing pretrain.
    training_rate: float = 0.8 # training rate.
    vali_rate: float = 0.1 # validation rate.
    label_index: int = 0 # label index, 0 is activate, 2 is user id, 3 is model', choices=[0, 1, 2]
    ##############################################################
    label_rate: float = 0.01
    balance: bool = True    
    lambda1: float = 0.0
    lambda2: float = 0.005

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))





