from typing import NamedTuple
import os, json, random, torch
import numpy as np

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

def load_raw_data(dataset, dataset_version):
    data_path = os.path.join('dataset', dataset, 'data_' + dataset_version + '.npy')
    label_path = os.path.join('dataset', dataset, 'label_' + dataset_version + '.npy')
    data = np.load(data_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)
    return data, labels

def load_embedding_label(dataset, dataset_version, embedding_path):
        label_name = 'label_' + dataset_version
        embed = np.load(embedding_path).astype(np.float32)
        labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
        return embed, labels

def load_dataset_label_names(dataset_config, label_index):
        for p in dir(dataset_config):
            if getattr(dataset_config, p) == label_index and "label_index" in p:
                temp = p.split("_")
                label_num = getattr(dataset_config, temp[0] + "_" + temp[1] + "_size")
                if hasattr(dataset_config, temp[0] + "_" + temp[1]):
                    return getattr(dataset_config, temp[0] + "_" + temp[1]), label_num
                else:
                    return None, label_num
        return None, -1

def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True, merge=0, merge_mode='all', shuffle=True):
        # 产生1到所有数据集长度的index
    arr = np.arange(data.shape[0])
    # 打乱arr里面的顺序
    if shuffle:
        np.random.shuffle(arr)
    # 把arr里面的顺序应用到data和labels上
    data = data[arr]
    labels = labels[arr]
    # 切分训练集,交叉测试集和测试集
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    # label_index 可以指明需要的是user的label还是activity的label
    # 注意每一个采样点都有其对应的label, 比如一个label为1的动作,采样120个点,这120个点的label都为一
    # t为最小的标签, 一般是0或1
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    # 这个change_shape实际上控制input_dim,比如trm可以一次性输入120时个点,
    # 而gru的input为20, 一个样本需要分割成6份,每份20个点
    if change_shape:
        # (1670, 120, 72) => (10020, 20, 72)
        # 改变一个样本的点数, 对于trm来说一个样本有120*72个点, 由于原始数据集中每个样本就120个点
        # 所以默认change_shape=False
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        # merge_mode是指多个dataset融合在一起共同训练,暂时可以不用理
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    return data_train, label_train, data_vali, label_vali, data_test, label_test

def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])

def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)

def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            # example: label的一个实例为[1,2,3]
            # 那么代表activity_label是1, user_label是2, model_label是3.
            # 确保每个样本的所有label都一致
            # 比如本来是label是(10020, 20),这里都把他们变为(10020,1)
            temp_label = np.unique(label[i])
            # 如果一个实例的全部标签都相同的情况, 比如activity_label,user_label和model_label都是3
            if temp_label.size == 1:
                index[i] = True 
                # 如果一个样本中的20个点label都一致,则认为该样本确实为该label
                # 否则因为该样本的标签不一致, 需要丢弃
                label_new.append(label[i, 0]) # 往label中添加相应的label
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                # 如果一个20个点的样本中存在不一致的标签, 则把unique得到的第二个标签作为新标签,否则为0
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    # index的长度和data一致,因此这里只会取出index中值为True的data,而label_new也只会保存对应位置的标签
    return data[index], np.array(label_new)

def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        # summarize the num of instances of each label
        label_num.append(np.sum(labels == labels_unique[i]))
    # if the num of instances of one label is small than the average label sparsity
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
    return data_train, label_train, data_test, label_test

def prepare_simple_dataset(data, labels, training_rate=0.2):
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
    return data_train, label_train, data_test, label_test