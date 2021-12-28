import torch
from torch import nn
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(train=False, path='/home/hcl/workspace/vit-pytorch/dataset/cifar-10-batches-py'):
    data = None
    labels = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle(path + '/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])

            if i == 1:
                labels = batch[b'targets']
            else:
                labels = np.concatenate([labels, batch[b'targets']])
    else:
        batch = unpickle(path + '/test_batch')
        data = batch[b'data']
        labels = batch[b'targets']
    return data, labels


def target_transform(label):
    label = np.array(label)  # 变为ndarray
    target = torch.from_numpy(label).long()  # 变为torch.LongTensor
    return target
