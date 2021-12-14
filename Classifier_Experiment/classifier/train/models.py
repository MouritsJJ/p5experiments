import torch.nn as nn
 
from constants import *

class Classifier(nn.Module):
    def __init__(self, ngpu):
        super(Classifier, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # --> 128x128x3
            nn.Conv2d(3, ngf, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            # --> 64x64x64
            nn.Conv2d(ngf, ngf * 2, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True), 

            # --> 32x32x128
            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            
            # --> 16x16x256
            nn.Conv2d(ngf * 4, ngf * 8, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            # --> 8x8x512

            nn.Flatten(),
            nn.Linear(ngf * 8 * 8 * 8, ngf * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Linear(ngf * 8 * 8, ngf * 8 * 8),
            nn.ReLU(inplace=True),
            # --> 1x4096
            
            nn.Linear(ngf * 8 * 8, n_labels)
            # --> 8
        )

    def forward(self, input):
        return self.main(input)