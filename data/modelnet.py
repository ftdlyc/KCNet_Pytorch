import os
import torch
import torch.utils.data as data
import numpy as np
from .augmentation import *


class ModelNetDataset(data.Dataset):

    def __init__(self, root, point_nums=2048, train=True, argumentation=True):
        self.root = root
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        file = open(os.path.join(root, 'modelnet10_shape_names.txt'), 'r')
        self.shape_list = [str.rstrip() for str in file.readlines()]
        file.close()
        self.class_nums = len(self.shape_list)

        if train:
            file = open(os.path.join(root, 'modelnet10_train.txt'), 'r')
        else:
            file = open(os.path.join(root, 'modelnet10_test.txt'), 'r')
        for line in file.readlines():
            line = line.rstrip()
            name = line[0:-5]
            label = self.shape_list.index(name)
            self.dataset.append((os.path.join(os.path.join(root, name), line + '.txt'), label))
        file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_path, label = self.dataset[index]
        data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]

        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(np.float32))

        return pc, label
