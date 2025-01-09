
import torch
import torch.nn as nn

from .layers.normalization import EnergyNormalization
from .layers.pad import Pad
from .layers.flatten import Flatten

class FULLY_CONV(nn.Module):
    def __init__(self, input_shape, num_classes, n_kernels=32, k_size=3, dilation=1, fc_size=256, norm="bn"):
        super(FULLY_CONV, self).__init__()

        self.energy_norm = EnergyNormalization()

        self.pad_c1 = Pad((int((k_size)/2) + (dilation-1), int((k_size)/2) + (dilation-1), 0, 0), 'zero')
        self.c1 = nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=(1,k_size),
                              dilation=(1, dilation), bias=False)
        if norm == "bn":
            self.n1 = nn.BatchNorm2d(n_kernels)
        else:
            self.n1 = nn.Dropout(0.5)

        self.pad_c2 = Pad((int((k_size)/2) + (dilation-1), int((k_size)/2) + (dilation - 1), 0, 1), 'I')
        self.c2 = nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=(2,k_size),
                            dilation=(1, dilation))
        if norm == "bn":
            self.n2 = nn.BatchNorm2d(n_kernels)
        else:
            self.n2 = nn.Dropout(0.5)                    

        ## replace fully connected layers with 2d convolutions
        self.fc1 = nn.Conv2d(in_channels=n_kernels, out_channels=fc_size, kernel_size=input_shape[1:])
        if norm == "bn":
            self.n3 = nn.BatchNorm2d(fc_size)
        else:
            self.n3 = nn.Dropout(0.5)

        self.fc_size = fc_size
        self.num_classes = num_classes
        self.out = nn.Conv2d(in_channels=fc_size, out_channels=num_classes, kernel_size=(1,1))

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.energy_norm(x)
        x = self.pad_c1(x)
        x = self.c1(x)
        x = self.relu(x)
        x = self.n1(x)

        x = self.pad_c2(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.n2(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.n3(x)
        x = self.out(x)
        x = Flatten()(x)    ## flatten to take out last two dimensions 
        return x
    
    def reset_out(self, num_classes):
        self.out = nn.Conv2d(in_channels=self.fc_size, out_channels=num_classes, kernel_size=(1,1)) 
        self.num_classes = num_classes

