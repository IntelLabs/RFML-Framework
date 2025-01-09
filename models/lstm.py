
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTM, self).__init__()
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=1, batch_first=True)
        self.do = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x):
        h0, c0 = self.hidden_init(x)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.do(x)
        x = self.linear(x)
        x = self.out(x)
        x = x[:,-1,:]
        return x

    def hidden_init(self,x):
        h0 = torch.zeros(2, x.size(0), 128)
        c0 = torch.zeros(2, x.size(0), 128)
        return [t.cuda() for t in (h0, c0)]

    def freeze(self):
        for name, module in self.named_children():
            if "linear" not in name and "out" not in name:
                for p in module.parameters():
                    p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_out(self, num_classes):
        self.out = nn.Linear(64, num_classes)
        self.num_classes = num_classes

