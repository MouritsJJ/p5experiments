from torch._C import device
import torch.nn as nn
import torch
from constants import *

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.label_generator = nn.Sequential(
            nn.Embedding(n_labels, embedded_dimension),
            nn.Linear(embedded_dimension, 16)
        )

        self.latent = nn.Sequential(
            nn.Linear(nz, 4*4*1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # input_channels, output_channels, kernel_size, stride, padding
            nn.ConvTranspose2d(ngf * 16 + 1, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 32, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        latent, label = input
        latent = torch.flatten(latent, start_dim=1)
        
        label_output = self.label_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)

        latent_output = self.latent(latent)
        latent_output = latent_output.view(-1, 1024, 4, 4)

        concat = torch.cat((latent_output, label_output), dim=1)
        
        return self.main(concat)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.labels_disc = nn.Sequential(
            nn.Embedding(n_labels, embedded_dimension),
            nn.Linear(embedded_dimension, 1*image_size*image_size)
        )

        self.main = nn.Sequential(
            nn.Conv2d(nc+1, ndf, 6, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 4, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        img, label = input
        label_output = self.labels_disc(label)
        label_output = label_output.view(-1, 1, image_size, image_size)
        concat = torch.cat((img, label_output), dim=1)
        return self.main(concat)