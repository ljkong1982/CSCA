import os
import torch
import numpy as np
import cv2
import math
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import pickle

def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


def GenerateRun(way = 5, shot = 5, query = 15, n_iteration = 0, img_dir = ''):
    np.random.seed(n_iteration)
    
    classes = np.random.permutation(np.arange(20))[:way]
    shuffle_indices = np.arange(600)
    dataset = None
    lf_list = []#label & fname

    data_from_plk = _load_pickle(img_dir)

    dataset = torch.ones(way, (shot + query), data_from_plk['data'].shape[1])
    data_from_plk['data'] = data_from_plk['data'].reshape(20,600,-1)

    for i in range(way):
        shuffle_indices = np.random.permutation(shuffle_indices)
        dataset[i] = data_from_plk['data'][classes[i]][shuffle_indices[:shot + query]]

    m_train_data = dataset[:,:shot,:]
    
    m_test_data = dataset[:,shot:,:]
    
    return m_train_data, m_test_data
