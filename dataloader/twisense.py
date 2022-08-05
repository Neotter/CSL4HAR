'''
Date: 2022-04-15 14:37:35
LastEditors: MonakiChen
LastEditTime: 2022-06-08 09:14:34
FilePath: \S3IMUd:\project\s3imu\code\dataloader\dataloader_lbert_pretrain.py
'''

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from utils.log_helper import *
from dataloader.preprocessing.normalization import Normalization
from dataloader.preprocessing.masking import Masking
from dataloader.dataloader_base import DataLoaderBase

class DataLoaderTwisense(DataLoaderBase):
    def __init__(self, args, logging):
        super().__init__(args, logging)

        self.change_shape = False
        self.merge_mode = 'all'
        
        # self.mask_ratio = args.mask_cfg.mask_ratio
        # self.mask_alpha = args.mask_cfg.mask_alpha
        # self.max_gram = args.mask_cfg.max_gram
        # self.mask_prob = args.mask_cfg.mask_prob
        # self.replace_prob = args.mask_cfg.replace_prob

        # self.pipeline = [Normalization(self.dataset_cfg.dimension), Masking(self.mask_ratio, self.mask_alpha, self.max_gram, self.mask_prob, self.replace_prob)]
        self.pipeline = [Normalization(self.dataset_cfg.dimension)]

        self.logging = logging

    def get_dataloader(self):
        data_train, label_train, data_vali, label_vali, _, _ = self.partition_and_reshape(
            self.data, self.labels, 
            label_index=self.label_index, 
            training_rate=self.training_rate, 
            vali_rate=self.vali_rate, 
            change_shape=self.change_shape,
            merge = 0,
            merge_mode=self.merge_mode
            )

        self.logging.info('Train Size: {}, Vali Size: {}'.format(data_train.shape[0], data_vali.shape[0]))
        
        data_set_train = self.TwisenseDataset4Pretrain(data_train,label_train, pipeline=self.pipeline)
        data_set_test = self.TwisenseDataset4Pretrain(data_vali,label_vali, pipeline=self.pipeline)

        self.data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=self.batch_size)
        self.data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=self.batch_size)

        return self.data_loader_train, self.data_loader_test

    class TwisenseDataset4Pretrain(Dataset):
        """ Load sentence pair (sequential or random order) from corpus """
        def __init__(self, data, labels, pipeline=[]):
            super().__init__()
            self.pipeline = pipeline
            self.data = data
            self.labels = labels

        def __getitem__(self, index):
            instance = self.data[index]
            # 每次取出样本后就进行pipeline中的处理,得到mask_seq, masked_pos, seq,三个值
            # mask_seq是被masked掉的序列,
            # masked_pos是被masked掉位置,
            # seq是需要predict的值
            for proc in self.pipeline:
                instance = proc(instance)
            seq = instance
            
            
            return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

        def __len__(self):
            return len(self.data)


