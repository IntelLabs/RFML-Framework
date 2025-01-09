
import torch
import torch.nn as nn
import torch.nn.functional as F

class Pad(nn.Module):
    def __init__(self, shape, mode='zero'):
        super(Pad, self).__init__()
        self.shape = shape ## L, R, T, B
        self.mode = mode ## zero, I

    def forward(self, x):
        if self.mode == "zero":
            return F.pad(x, self.shape)
        elif self.mode == "I":
            p = x[:,:,0,:]
            return F.pad(torch.cat((x, p[:,:,None,:]), 2), (self.shape[0],self.shape[1],0,0))