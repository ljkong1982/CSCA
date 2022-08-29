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


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

    
def GetFeatures(lf_list, all_file_name):
    
    fname_list = []
    
    for n in lf_list:
        cls_index = n[0]
        f_list = n[1]
        
        for findex in f_list:
            fname_list.append(all_file_name[cls_index][findex])# 111...111 222...222 333...333 444...444 555...555
    #extract_feature
    data_set = readImg(fname_list)
    data_loader = DataLoader(data_set, batch_size=256, shuffle=False, num_workers=8)
    
    model = models.resnet101(pretrained=True)

    flatten_layer = nn.Flatten()
    
    if(torch.cuda.is_available()):
        model = model.cuda()
    
    model.eval()
    
    with torch.no_grad():
        for i, (image) in enumerate(data_loader):
            if(torch.cuda.is_available()):
                image=image.cuda()
            
            model.avgpool.register_forward_hook(get_activation('avgpool'))
            val_pred = model(image)
                
    output = activation['avgpool']
    
    output = flatten_layer(output)
    #seperate classes
    
    n=0
    each_c_len = len(lf_list[0][1])
    
    dataset = torch.zeros(len(lf_list), each_c_len, output.shape[1])#cfg['ways'], cfg['shot']+cfg['queries']
    for i in range(len(lf_list)):#5
        for j in range(each_c_len):#20
            dataset[i][j] = output[n]
            n+=1
        
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


if __name__ == '__main__':
    
    img_dir = "../../miniimagenet/data/test/" #64 600
    all_file_name = []
    for cls_name in os.listdir(img_dir):
        cls_fname = []
        for img_name in os.listdir(os.path.join(img_dir, cls_name)):

            if '.jpg' not in img_name:
                continue
            cls_fname.append(os.path.join(img_dir,cls_name,img_name))
        all_file_name.append(cls_fname)
    
    
    m_train_data,m_test_data = GenerateRun(5, 5, 15, 1, all_file_name)
    print(m_train_data.shape,m_test_data.shape)#torch.Size([5, 20, 2048])