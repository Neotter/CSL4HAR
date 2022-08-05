'''
Date: 2022-04-15 14:37:35
LastEditors: MonakiChen
LastEditTime: 2022-06-08 09:14:34
FilePath: \S3IMUd:\project\s3imu\code\dataloader\dataloader_lbert_pretrain.py
'''
import os, json
import random

from torch.utils.data import DataLoader, Dataset

import torch
import numpy as np
from utils.log_helper import *
from dataloader.preprocessing.normalization import Normalization
from dataloader.dataloader_base import DataLoaderBase

class DataLoaderLIBERTEmbedding(DataLoaderBase):
    def __init__(self, args, logging):
        super().__init__(args, logging)
        
        self.change_shape = True
        self.merge_mode = 'all'
        self.merge = 20
        self.batch_size = args.train_cfg.batch_size
        self.balance = args.train_cfg.balance
        self.label_rate = args.train_cfg.label_rate

        self.pipeline = [Normalization(args.model_cfg.input_dim)]

        self.embedding_path = args.embedding_path

    def load_embedding_label(self):
        label_name = 'label_' + self.dataset_version
        embed = np.load(self.embedding_path).astype(np.float32)
        labels = np.load(os.path.join('dataset', self.dataset, label_name + '.npy')).astype(np.float32)
        return embed, labels
        
    def get_dataloader(self):
        embed, labels = self.load_embedding_label()
        data_train, label_train, data_valid, label_valid, data_test, label_test = \
            self.partition_and_reshape(
                    embed, labels, 
                    label_index=self.label_index, 
                    training_rate=self.training_rate, 
                    vali_rate=self.vali_rate, 
                    change_shape=self.change_shape,
                    merge = self.merge,
                    merge_mode=self.merge_mode
                    )

        if self.balance:
            data_train, label_train, _, _ \
            = self.prepare_simple_dataset_balance(data_train, label_train, training_rate=self.label_rate)
        else:
            data_train, label_train, _, _ \
            = self.prepare_simple_dataset(data_train, label_train, training_rate=self.label_rate) 


        data_set_train = self.IMUDataset(data_train, label_train, pipeline=self.pipeline)
        data_set_valid = self.IMUDataset(data_valid, label_valid, pipeline=self.pipeline)
        data_set_test = self.IMUDataset(data_test, label_test, pipeline=self.pipeline)

        self.data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=self.batch_size)
        self.data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=self.batch_size)
        self.data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=self.batch_size)

        return self.data_loader_train, self.data_loader_valid, self.data_loader_test

    def prepare_simple_dataset_balance(self, data, labels, training_rate=0.8):
        labels_unique = np.unique(labels)
        label_num = []
        for i in range(labels_unique.size):
            label_num.append(np.sum(labels == labels_unique[i]))
        train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
        if train_num == min(label_num):
            print("Warning! You are using all of label %d." % label_num.index(train_num))
        index = np.zeros(data.shape[0], dtype=bool)
        for i in range(labels_unique.size):
            class_index = np.argwhere(labels == labels_unique[i])
            class_index = class_index.reshape(class_index.size)
            np.random.shuffle(class_index)
            temp = class_index[:train_num]
            index[temp] = True
        t = np.min(labels)
        data_train = data[index, ...]
        data_test = data[~index, ...]
        label_train = labels[index, ...] - t
        label_test = labels[~index, ...] - t
        logging.info('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0], label_train.shape[0] * 1.0 / labels.size))
        return data_train, label_train, data_test, label_test

    def prepare_simple_dataset(self, data, labels, training_rate=0.2):
        arr = np.arange(data.shape[0])
        np.random.shuffle(arr)
        data = data[arr]
        labels = labels[arr]
        train_num = int(data.shape[0] * training_rate)
        data_train = data[:train_num, ...]
        data_test = data[train_num:, ...]
        t = np.min(labels)
        label_train = labels[:train_num] - t
        label_test = labels[train_num:] - t
        labels_unique = np.unique(labels)
        label_num = []
        for i in range(labels_unique.size):
            label_num.append(np.sum(labels == labels_unique[i]))
        logging.info('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
            % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
        return data_train, label_train, data_test, label_test

    class IMUDataset(Dataset):
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

