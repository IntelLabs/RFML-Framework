
import torch
import torch.nn as nn

from .layers.flatten import Flatten
from .layers.normalization import EnergyNormalization
from .layers.awgn import AWGN

'''
Source: https://github.com/brysef/rfml
'''
class CLDNN(nn.Module):
    def __init__(self, input_shape, num_classes, snr_min=-10., snr_max=20.):
        super(CLDNN, self).__init__()
        self.num_classes = num_classes

        self.awgn = AWGN(snr_min = snr_min, snr_max = snr_max)
        # self.norm = EnergyNormalization()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1,7), padding=(0,3), bias=False)
        self.a1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(50)

        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1,7), padding=(0,3), bias=True)
        self.a2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(50)

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1,7), padding=(0,3), bias=True)
        self.a3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(50)

        self.flatten1 = Flatten(preserve_time=True)

        self.hidden_size = num_classes
        self.gru = nn.GRU(input_size=200, hidden_size=self.hidden_size, batch_first=True, num_layers=1, bidirectional=False)

        self.flatten2 = Flatten()

        self.lin1 = nn.Linear(input_shape[-1]*self.hidden_size*1, 256)
        self.a4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # x = self.norm(x)
        x = self.awgn(x)

        x = self.conv1(x)
        x = self.a1(x)
        a = self.bn1(x)

        x = self.conv2(a)
        x = self.a2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.a3(x)
        x = self.bn3(x)

        x = torch.cat((a, x), dim=1)

        x = self.flatten1(x)
        hidden = x.new(1, batch_size, self.num_classes)
        hidden.zero_()
        x, _ = self.gru(x, hidden)

        x = self.flatten2(x)

        x = self.lin1(x)
        x = self.a4(x)
        x = self.bn4(x)

        x = self.out(x)

        return x

    def freeze(self):
        for name, module in self.named_children():
            if "lin" not in name and "bn4" not in name and "out" not in name:
                for p in module.parameters():
                    p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_out(self, num_classes):
        self.out = nn.Linear(256, num_classes)	
        self.num_classes = num_classes