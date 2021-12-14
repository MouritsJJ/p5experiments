from torch._C import device
import torch.nn as nn
import torch
from torch.nn.modules.activation import ReLU
from constants import *

class Classifier(nn.Module):
    def __init__(self, ngpu):
        super(Classifier, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # --> 128x128x3
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            # --> 64x64x64
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True), 

            # --> 32*32*128
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            
            # --> 16x16x256
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            # --> 8x8x512

            nn.Flatten(),
            nn.Linear(8*8*512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, n_labels)
        )

    def forward(self, input):
        return self.main(input)