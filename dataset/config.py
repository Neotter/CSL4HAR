import json
from typing import NamedTuple

class DatasetConfig(NamedTuple):
    """ Hyperparameters for training """
    sr: int = 0  # sampling rate
    # dataset = Narray with shape (size, seq_len, dimension)
    size: int = 0  # data sample number
    seq_len: int = 0  # seq length
    dimension: int = 0  # feature dimension

    activity_label_index: int = -1  # index of activity label
    activity_label_size: int = 0  # number of activity label
    activity_label: list = []  # names of activity label.

    user_label_index: int = -1  # index of user label
    user_label_size: int = 0  # number of user label

    position_label_index: int = -1  # index of phone position label
    position_label_size: int = 0  # number of position label
    position_label: list = []  # names of position label.

    model_label_index: int = -1  # index of phone model label
    model_label_size: int = 0  # number of model label

    @classmethod
    def from_json(cls, js):
        return cls(**js)

    def load_dataset_cfg(dataset_cfg_path, dataset, version):
        path = dataset_cfg_path
        dataset_config_all = json.load(open(path, "r"))
        name = dataset + "_" + version
        if name in dataset_config_all:
            return DatasetConfig.from_json(dataset_config_all[name])
        else:
            return None