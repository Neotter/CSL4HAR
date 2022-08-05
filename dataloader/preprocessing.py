import numpy as np

from typing import NamedTuple
import json

class MaskConfig(NamedTuple):
    """ Hyperparameters for training """
    mask_ratio: float = 0  # masking probability
    mask_alpha: int = 0  # How many tokens to form a group.
    max_gram: int = 0  # number of max n-gram to masking
    mask_prob: float = 1.0
    replace_prob: float = 0.0

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class Masking:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, masking_cfg):

        self.mask_ratio = masking_cfg.mask_ratio # masking probability
        self.mask_alpha = masking_cfg.mask_alpha
        self.max_gram = masking_cfg.max_gram
        self.mask_prob = masking_cfg.mask_prob
        self.replace_prob = masking_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def span_mask(self, seq_len, max_gram=3, p=0.2, goal_num_predict=15):
        # 最长的span的长度
        ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
        # The geometric distribution Geo(p)
        # 从几何分布中采样span的长度, 最短为1,最长为10,平均采样区间为3.8
        # 几何分布的特点是值越小采样到的概率越高
        pvals = p * np.power(1 - p, np.arange(max_gram))
        # alpha = 6
        # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
        pvals /= pvals.sum(keepdims=True)
        mask_pos = set()
        while len(mask_pos) < goal_num_predict:
            # 按照概率分布p抽取ngrams中的元素,得到的值为span的长度.
            n = np.random.choice(ngrams, p=pvals)
            # span的长度不得超过剩余的需要产生的预训练样本数.
            # 例如,剩余要产生的样本数为10,那么随机采样的的n要小于10
            n = min(n, goal_num_predict - len(mask_pos))
            anchor = np.random.randint(seq_len)
            if anchor in mask_pos:
                continue
            for i in range(anchor, min(anchor + n, seq_len - 1)):
                mask_pos.add(i)
        return list(mask_pos)
        # M_max = L * p_m   

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        # 设置一个样本产生的预训练样本masked的最大长度, 由mask_ratio控制
        # 比如总的长度为120,mask_ratio是0.15, 那么就一个样本最多可以masked掉18个点
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = self.span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)
        
        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            # span masking掩盖掉了mask_prob %的token,其中80%的概率被替换为mask,replace_prob %的概率被随机值替换
            # 这部分代码是有问题的,原版的span masking是肯定会mask掉总token的15%, 并且这15%的token中有80%被0替换,10%被随机值替换.
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            # 根据成功率replace_prob使用随机值替换,代码中原本的值是不需要随机替换
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        # instance_mask: 替换后的序列
        # np.array(mask_pos_index) 替换的位置index
        # 需要predict的ground truth
        return instance_mask, np.array(mask_pos_index), np.array(seq)

import numpy as np

class Normalization:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # copy之后从第二个数组中取出self.feature_len，说明列才是代表每个轴
        # npy中的shape是这样的(9166, 120, 6)，代表hhar有9166个样本，每个样本持续120个点，6个轴
        # 每次调用__call__会提取出一个(120,6)的样本,self.feature_len是指IMU的轴的数量
        # 比如feature_len=3,则只有acc的xyz参与训练
        instance_new = instance.copy()[:, :self.feature_len]

        # instance_new中[:3]是acc，[3:6]是gyro，[6:9]是mag
        # 至于为什么要判断>= 6 或者== 9,这是因为如果只有acc一个传感器,那么就没有
        # Normalization的必要了, 因为paper中claim的点就是不同的传感器的scale不同, 才需要做
        # Normalization.
        if instance_new.shape[1] >= 6 and self.norm_acc:
            # 只拿前三个轴，也就是acc
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm

        if instance_new.shape[1] == 9 and self.norm_mag:
            # 沿着列求2的范数
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            # (120, 6)展开成（720）后沿着列连续重复复制3次
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new

class Normalization:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # copy之后从第二个数组中取出self.feature_len，说明列才是代表每个轴
        # npy中的shape是这样的(9166, 120, 6)，代表hhar有9166个样本，每个样本持续120个点，6个轴
        # 每次调用__call__会提取出一个(120,6)的样本,self.feature_len是指IMU的轴的数量
        # 比如feature_len=3,则只有acc的xyz参与训练
        instance_new = instance.copy()[:, :self.feature_len]

        # instance_new中[:3]是acc，[3:6]是gyro，[6:9]是mag
        # 至于为什么要判断>= 6 或者== 9,这是因为如果只有acc一个传感器,那么就没有
        # Normalization的必要了, 因为paper中claim的点就是不同的传感器的scale不同, 才需要做
        # Normalization.
        if instance_new.shape[1] >= 6 and self.norm_acc:
            # 只拿前三个轴，也就是acc
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm

        if instance_new.shape[1] == 9 and self.norm_mag:
            # 沿着列求2的范数
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            # (120, 6)展开成（720）后沿着列连续重复复制3次
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new

class Clipping:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, clip_rate):
        super().__init__()
        self.clip_rate = clip_rate

    def __call__(self, instance):
        
        instance_new = instance.copy()
        instance_len = int(instance_new.shape[0] - instance_new.shape[0]*self.clip_rate)
        instance_new = instance_new[:instance_len,:]

        return instance_new

class WordDeletion:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, del_rate):
        super().__init__()
        self.del_rate = del_rate

    def __call__(self, instance):

        instance_new = instance.copy()

        num_del = int(instance_new.shape[0]*self.del_rate)

        arr = np.arange(instance.shape[0])
        arr = np.random.choice(arr, num_del, replace=False)
        
        instance_new = np.delete(instance_new, arr,axis=0)

        return instance_new



def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

    return np.matmul(X, matrices)

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed