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

    
if __name__ == "__main__":  
    model = channel_attention(channel_sz = 10)
    
    input = torch.ones([5, 5, 10])
    print(input.shape)
    input = input.reshape(1, 25,-1).unsqueeze(0)#torch.Size([1, 1, 25, 2048])
    print(input.shape)
    input = input.permute(2,3,1,0)
    print(input.shape)
    
    
    
    output = model(input)
    print(output.shape)
    
    
    
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    