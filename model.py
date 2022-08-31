import torch
import torch.nn as nn


class channel_attention(nn.Module):
    def __init__(self, channel_sz=2048):
        super(channel_attention, self).__init__()
        self.c_a =  nn.Conv2d(channel_sz,channel_sz, kernel_size = 1, stride=1, groups=channel_sz, bias=False)
        self.c_a.weight = torch.nn.Parameter(torch.ones(channel_sz,1,1,1))
    
    def forward(self, x):
        x = self.c_a(x)
        return x
    
