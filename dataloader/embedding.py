import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from utils.log_helper import *
from dataloader.preprocessing import *
from dataloader.helper import *

class DatasetSelector4Embedding(object):
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

        self.pipeline = [Normalization(self.dataset_cfg.dimension)]
        self.Dataset4Embedding = Dataset4Embedding

    def get_dataloader(self):
        
        data_set_all = self.Dataset4Embedding(self.data, self.label, pipeline=self.pipeline)

        data_loader_all = DataLoader(data_set_all, shuffle=False, batch_size=self.batch_size)

        return data_loader_all

class Dataset4Embedding(Dataset):
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


