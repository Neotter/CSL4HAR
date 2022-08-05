from typing import NamedTuple
import os, json, random, torch
import numpy as np

from config.dataset import DatasetConfig

class DataLoaderBase(object):
    def __init__(self, args, logging):
        """ Load config """

        self.dataset = args.dataset
        self.dataset_version = args.dataset_version
        self.dataset_cfg = args.dataset_cfg
        # 设置使用什么标签, 标签是动作还是身份
        self.label_index = args.train_cfg.label_index
        self.label_names, self.label_num = self.load_dataset_label_names(self.dataset_cfg, self.label_index)
        
        """ Load raw data """
        self.data_path = os.path.join('dataset', args.dataset, 'data_' + args.dataset_version + '.npy')
        self.label_path = os.path.join('dataset', args.dataset, 'label_' + args.dataset_version + '.npy')
        self.data, self.labels = self.load_raw_data(data_path=self.data_path, label_path=self.label_path)

        self.batch_size = args.train_cfg.batch_size
        self.training_rate = args.train_cfg.training_rate
        self.vali_rate = args.train_cfg.vali_rate


    def load_raw_data(self,data_path,label_path):
        # # model—_cfg中的feature_dimension不能大于dataset_cfg的dimension
        # if model_cfg.feature_num > dataset_cfg.dimension:
        #     print("Bad Crossnum in model cfg")
        #     sys.exit()
        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path).astype(np.float32)
        return data, labels

    def load_dataset_label_names(self, dataset_config, label_index):
            for p in dir(dataset_config):
                if getattr(dataset_config, p) == label_index and "label_index" in p:
                    temp = p.split("_")
                    label_num = getattr(dataset_config, temp[0] + "_" + temp[1] + "_size")
                    if hasattr(dataset_config, temp[0] + "_" + temp[1]):
                        return getattr(dataset_config, temp[0] + "_" + temp[1]), label_num
                    else:
                        return None, label_num
            return None, -1

    def partition_and_reshape(self, data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True, merge=0, merge_mode='all', shuffle=True):
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
            data_train = self.reshape_data(data_train, merge)
            data_vali = self.reshape_data(data_vali, merge)
            data_test = self.reshape_data(data_test, merge)
            label_train = self.reshape_label(label_train, merge)
            label_vali = self.reshape_label(label_vali, merge)
            label_test = self.reshape_label(label_test, merge)
        if change_shape and merge != 0:
            # merge_mode是指多个dataset融合在一起共同训练,暂时可以不用理
            data_train, label_train = self.merge_dataset(data_train, label_train, mode=merge_mode)
            data_test, label_test = self.merge_dataset(data_test, label_test, mode=merge_mode)
            data_vali, label_vali = self.merge_dataset(data_vali, label_vali, mode=merge_mode)
        return data_train, label_train, data_vali, label_vali, data_test, label_test

    def reshape_data(self, data, merge):
        if merge == 0:
            return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        else:
            return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])

    def reshape_label(self, label, merge):
        if merge == 0:
            return label.reshape(label.shape[0] * label.shape[1])
        else:
            return label.reshape(label.shape[0] * label.shape[1] // merge, merge)

    def merge_dataset(self, data, label, mode='all'):
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

