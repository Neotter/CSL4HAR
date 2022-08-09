import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from utils.log_helper import *
from dataloader.preprocessing import *
from dataloader.helper import *

class DatasetSelector4Pretrain(object):
    def __init__(self, args):
        super().__init__()

        """ Load dataset config """
        self.dataset = args.dataset
        self.dataset_version = args.dataset_version
        self.dataset_cfg = args.dataset_cfg

        """ Load raw dataset """
        self.data, self.label = load_raw_data(self.dataset, self.dataset_version)

        self.change_shape = False
        self.merge_mode = 'all'

        """ set datset config for pretrain """
        self.label_index = args.train_cfg.label_index
        self.label_names, self.label_num = load_dataset_label_names(self.dataset_cfg, self.label_index)

        self.batch_size = args.train_cfg.batch_size
        self.training_rate = args.train_cfg.training_rate
        self.vali_rate = args.train_cfg.vali_rate

        self.augmentation = args.augmentation

        if self.augmentation == 'scsense':
            self.pipeline = [Normalization(self.dataset_cfg.dimension)]
            self.Dataset4Pretrain = Dataset4Scsense
        if self.augmentation == 'spanmasking':
            self.masking_cfg = args.mask_cfg
            self.pipeline = [Normalization(self.dataset_cfg.dimension), Masking(self.masking_cfg)]
            self.Dataset4Pretrain = Dataset4Spanmasking
        if self.augmentation == 'clipping':
            self.clip_rate = args.augment_rate
            self.pipeline = [Normalization(self.dataset_cfg.dimension),Clipping(self.clip_rate)]
            self.Dataset4Pretrain = Dataset4Clipping
        if self.augmentation == 'delwords':
            self.del_rate = args.augment_rate
            self.pipeline = [Normalization(self.dataset_cfg.dimension),WordDeletion(self.del_rate)]
            self.Dataset4Pretrain = Dataset4Delwords
        # if self.augmentation == 'rotated':
        #     self.rotate_angles = 20
        #     self.pipeline = [Normalization(self.dataset_cfg.dimension),WordDeletion(self.del_rate)]
        #     self.Dataset4Pretrain = Dataset4Delwords

    def get_dataloader(self):
        # partition_and_reshape
        data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(
            self.data, self.label, 
            label_index=self.label_index, 
            training_rate=self.training_rate, 
            vali_rate=self.vali_rate, 
            change_shape=self.change_shape,
            merge = 0,
            merge_mode=self.merge_mode
            )
        # vali used for selecting the best pretrained model
        data_set_train = self.Dataset4Pretrain(data_train, label_train, pipeline=self.pipeline)
        data_set_vali = self.Dataset4Pretrain(data_vali, label_vali, pipeline=self.pipeline)

        self.data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=self.batch_size)
        self.data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=self.batch_size)

        return self.data_loader_train, self.data_loader_vali

class Dataset4Spanmasking(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        # 每次取出样本后就进行pipeline中的处理,得到mask_seq, masked_pos, seq,三个值
        # raw_seq是原始的样本
        # masked_seq是被masked掉的序列,
        # masked_pos是被masked掉的位置,
        # pred_seq是需要predict的值
        raw_seq = instance
        for proc in self.pipeline:
            instance_p1 = proc(instance)
            instance_p2 = proc(instance)
            if proc == Normalization:
                raw_seq = instance_p1
        mask_seq_1, masked_pos, pred_seq = instance_p1
        mask_seq_2, masked_pos, pred_seq = instance_p2
        # return torch.from_numpy(raw_seq), torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)
        return torch.from_numpy(raw_seq), torch.from_numpy(mask_seq_1), torch.from_numpy(mask_seq_2)

    def __len__(self):
        return len(self.data)

class Dataset4Scsense(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        # 每次取出样本后就进行pipeline中的处理,得到mask_seq, masked_pos, seq,三个值
        # mask_seq是被masked掉的序列,
        # masked_pos是被masked掉位置,
        # seq是需要predict的值
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance), torch.from_numpy(instance), torch.from_numpy(instance)

    def __len__(self):
        return len(self.data)

class Dataset4Clipping(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]

        for proc in self.pipeline:
            instance = proc(instance)
            if type(proc) == Normalization:
                raw_seq = instance
        clipped_seq = instance
        return torch.from_numpy(raw_seq), torch.from_numpy(raw_seq), torch.from_numpy(clipped_seq)

    def __len__(self):
        return len(self.data)

class Dataset4Delwords(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance_p1 = proc(instance)
            instance_p2 = proc(instance)
            if type(proc) == Normalization:
                raw_seq = instance_p1
        delwords_seq_1 = instance_p1
        delwords_seq_2 = instance_p2
        return torch.from_numpy(raw_seq), torch.from_numpy(delwords_seq_1), torch.from_numpy(delwords_seq_2)

    def __len__(self):
        return len(self.data)


