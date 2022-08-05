import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSense(nn.Module):

    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(cfg.flat_num, 12)
        self.lin2 = nn.Linear(12, output)


    def forward(self, input_seqs, training=False):
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        hs = []
        for i in range(self.sensor_num):
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i) + "_" + str(j + 1))
                bn = self.__getattr__('bn' + str(i) + "_" + str(j + 1))
                t = bn(F.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(F.relu(self.conv1(h)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.flatten(h)
        h = self.lin2(F.relu(self.lin1(h)))
        return h