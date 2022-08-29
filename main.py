import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.models as models
import argparse
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import random
from torch.optim import lr_scheduler
from dataloader import GenerateRun
from model import channel_attention
from statistics import mean, stdev

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniImagenet', help='CUB/miniImagenet/tiered_imagenet/cifar')
    parser.add_argument('--Epoch', type=int, default=10000, help='number of test iterations')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--meta_train_epoch', type=int, default=15, help='10/15/20/25')
    parser.add_argument('--lr', type=int, default=0.1)
    opt = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    img_dir = "./pretrained_models_features/" + opt.dataset + "/output.plk"

    acc_list = []

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for epoch in range(opt.Epoch):
        
        model = [channel_attention(channel_sz = 640) for _ in range(opt.way)]

        if(use_gpu):
            for m in model:
                m.cuda()

        mtrain_data, mtest_data = GenerateRun(way = opt.way, shot = opt.shot, query = opt.query, n_iteration = epoch, img_dir = img_dir)
        
        
        #train channel attention
        
        for i in range(opt.way):
            
            proto = [[] for _ in range(opt.way)]
            
            cls_start_index = i*opt.shot
            model[i].train()
            optimizer = torch.optim.Adam(model[i].parameters(), opt.lr)
            mtrain_data = mtrain_data.reshape(1, opt.way * opt.shot,-1).unsqueeze(0)
            mtrain_data = mtrain_data.permute(2,3,1,0)
            mtrain_data = mtrain_data.cuda()
            
            
            for m_epoch in range(opt.meta_train_epoch):
                optimizer.zero_grad()
                
                output = model[i](mtrain_data)
                output = output.squeeze(-1).squeeze(-1)
                
                for idx in range(opt.way):
                    proto[idx] = torch.mean(output[idx * opt.shot : idx * opt.shot+5], 0, True)
                    
                #1.intra class
                class_inner_sim = torch.mean(cos(output[cls_start_index : cls_start_index+5], proto[i]))
                class_inner_dis = 1-class_inner_sim
                
                #2.inter class
                
                class_outer_sim = class_inner_sim - class_inner_sim
                
                for c_i in range(opt.way):
                    if c_i != i:
                        class_outer_sim += torch.mean(cos(output[c_i * opt.shot : c_i * opt.shot+5], proto[i]))
                class_outer_sim = class_outer_sim/(opt.way-1)

                batch_loss = -1*torch.log((class_inner_sim - 0.5) / (1 - class_outer_sim))
                batch_loss.backward()
                optimizer.step()
        #prototypes for each classes after channel attention
        
        prototypes = []
        
        for m in model:
            m.eval()
        
        with torch.no_grad():
            for i in range(opt.way):
                output = model[i](mtrain_data).squeeze(-1).squeeze(-1)
                prototypes.append(torch.mean(output[i * opt.shot:i * opt.shot+5],0,True))
                
        assert len(prototypes) == opt.way
        
        mtest_data = mtest_data.reshape(1, opt.way * opt.query,-1).unsqueeze(0)
        mtest_data = mtest_data.permute(2,3,1,0)
        mtest_data = mtest_data.cuda()
        
        correct = 0
        
        with torch.no_grad():
            for i in range(opt.way):
                sim_matrix = [[] for _ in range(opt.way)]
                
                for j in range(opt.way):
                    output = model[i](mtest_data).squeeze(-1).squeeze(-1)
                    sim_matrix[j] = cos(output[i * opt.query:(i+1) * opt.query],prototypes[j]).cpu().data.numpy()
                
                pred = np.argmax(sim_matrix,axis=0)
                correct += np.count_nonzero(pred==i)

        acc = correct/(opt.way * opt.query)
        acc_list.append(acc)
        if epoch>0:
            print('{}/{} : acc {:0.2f} current avg: {:0.2f} +- {:0.2f}'.format(epoch+1, opt.Epoch, acc, 100*mean(acc_list), stdev(acc_list)))
        else:
            print('{}/{} : acc {:0.2f} current avg: {:0.2f}'.format(epoch+1, opt.Epoch, acc, 100*mean(acc_list)))
    
    print("final accuracy found {} +- {}".format(100*mean(acc_list),stdev(acc_list)))
                
                
                
                
            
        
        