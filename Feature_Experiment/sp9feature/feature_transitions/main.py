import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import shutil
import torch
from torch import load
from multiprocessing import Process
from pathlib import Path

from bce_gen import *
from base_gen import *

matplotlib.use('agg')

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def load_gen(gen, file_name):
    """
    Loads the model in file_name
    Input:
        disc - discriminator (nn.module)
        gen - generator (nn.module)
        disc_op - disciminator optimizer (torch)
        gen_op - generator optimizer (torch)
        file_name - file name to read from
    """

    models = load(f'../{file_name}', map_location=torch.device('cpu'))
    gen.load_state_dict(models['generator'])
    gen.eval()

def save_img(path, img):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.savefig(f'./{path}.png')
    plt.close()


def create_folders():
    path_bce = Path(f'./bce_images')
    path_base = Path(f'./base_images')

    if path_bce.is_dir():
        shutil.rmtree(f'{str(path_bce)}/')
    path_bce.mkdir()
        
    if path_base.is_dir():
        shutil.rmtree(f'{str(path_base)}/')
    path_base.mkdir()



def gen_images(fixed_noise, fixed_label, netG, folder):
    
    for k in range(120):
        n = 0
        noise = torch.clone(fixed_noise)
        Path(f'./{folder}/{k}').mkdir()
        for i in np.arange(-10, 10.01, 0.1):
            for j in range(64):
                noise[j][k][0][0] = i
            
            with torch.no_grad():
                output = netG((noise, fixed_label)).detach().cpu()
                last= vutils.make_grid(output, padding=2, normalize=True)  
                save_img(f"./{folder}/{k}/{n}", last)
            n += 1



def main():
    create_folders()
    processes = []
    netG_bce = Generator(0).to(device)
    netG_base = Generator(0).to(device)
    nz = 120
    load_gen(netG_bce, "BCE_model")
    load_gen(netG_base, "base_model")
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    fixed_label = torch.tensor([[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3],[3],[3],[3],[3],[3],[4],[4],[4],[4],[4],[4],[4],[4],[5],[5],[5],[5],[5],[5],[5],[5],[6],[6],[6],[6],[6],[6],[6],[6],[7],[7],[7],[7],[7],[7],[7],[7]], device=device)

    gen_images(fixed_noise, fixed_label, netG_bce, "bce_images")
    gen_images(fixed_noise, fixed_label, netG_base, "base_images")

if __name__ == "__main__":
    main()