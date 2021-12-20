"""
Code heavily inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (Accessed 19/12-2021)
with only small adjustments
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import os

from DataIO import DataIO
from models import *
from constants import *
from utils import *

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataIO = DataIO('data', training_iteration)

def create_generator():
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    return netG

def create_discriminator():
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    return netD

def generate_images(gen, num_of_imgs_per_class, batch_size):
    for label in classes:
        os.makedirs(f'{dataIO.path}/generated_images/{label}', exist_ok=True)

    with torch.no_grad():
        batches = int(num_of_imgs_per_class / batch_size)
        print(f'num of batches: {batches}')
        for label in range(n_labels):
            print(f"starting label {label}")
            current_image = 1
            for _ in range(batches):
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                labels = torch.full((batch_size, 1), label, device=device)
                output = gen((noise, labels))

                for image_number in range(batch_size):
                    denormed_image = denormalize(output[image_number].cpu())
                    dataIO.save_single_image(denormed_image,
                    f'generated_images/{classes[label]}/{current_image}')
                    current_image += 1

def main():
    print(f'device: {device}')

    # ensure batch size and num of imgs per class add up
    assert num_of_imgs_pr_class % batch_size == 0

    gen = create_generator()
    disc = create_discriminator()

    optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=beta_params)
    optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=beta_params)

    dataIO.load_models(disc, gen, optimizerD, optimizerG, model_path)

    generate_images(gen, num_of_imgs_pr_class, batch_size)
    
main()