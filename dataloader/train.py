'''
Date: 2022-08-05 20:13:48
LastEditors: MonakiChen
LastEditTime: 2022-08-09 16:27:09
FilePath: \CSL4HAR\dataloader\train.py
'''
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from utils.log_helper import *
from dataloader.preprocessing import *
from dataloader.helper import *

class DatasetSelector4Train(object):
    def __init__(self, args):
        super().__init__()

        """ Load dataset config """
        self.dataset = args.dataset
        self.dataset_version = args.dataset_version
        self.dataset_cfg = args.dataset_cfg
        self.pretrained_embedding_path = args.load_path        

        """ Load raw dataset """
        self.data, self.labels = load_raw_data(self.dataset, self.dataset_version)
        """ Load embedding """
        self.embed, self.labels = load_embedding_label(self.dataset, self.dataset_version, self.pretrained_embedding_path)

        """ Other config"""
        self.change_shape = True
        self.merge_mode = 'all'
        self.merge = 20

        """ set datset config for train """
        self.label_index = args.ds_train_cfg.label_index
        self.label_names, self.label_num = load_dataset_label_names(self.dataset_cfg, self.label_index)

        self.batch_size = args.ds_train_cfg.batch_size
        self.label_rate = args.ds_train_cfg.label_rate
        self.balance = args.ds_train_cfg.balance
        self.training_rate = args.ds_train_cfg.training_rate
        self.vali_rate = args.ds_train_cfg.vali_rate

        self.pipeline = [Normalization(args.ds_model_cfg.input_dim)]
        self.Dataset4Embedding = Dataset4Train

    def get_dataloader(self):
        data_train, label_train, data_valid, label_valid, data_test, label_test = \
            partition_and_reshape(
                self.embed, self.labels, 
                label_index=self.label_index, 
                training_rate=self.training_rate, 
                vali_rate=self.vali_rate, 
                change_shape=self.change_shape,
                merge = self.merge,
                merge_mode=self.merge_mode)

        if self.balance:
            data_train, label_train, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=self.label_rate)
        else:
            data_train, label_train, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=self.label_rate) 
        
        data_set_train = Dataset4Train(data_train, label_train, pipeline=self.pipeline)
        data_set_valid = Dataset4Train(data_valid, label_valid, pipeline=self.pipeline)
        data_set_test = Dataset4Train(data_test, label_test, pipeline=self.pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=self.batch_size)
        data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=self.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=self.batch_size)

        return data_loader_train, data_loader_valid, data_loader_test

class Dataset4Train(Dataset):
        def __init__(self, data, labels, pipeline=[]):
            super().__init__()
            self.pipeline = pipeline
            self.data = data
            self.labels = labels

        def __getitem__(self, index):
            instance = self.data[index]
            for proc in self.pipeline:
                instance = proc(instance)
            return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

        def __len__(self):
            return len(self.data)


