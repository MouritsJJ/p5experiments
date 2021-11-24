import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
import argparse

from DataIO import DataIO
from models import *
from constants import *

parser = argparse.ArgumentParser(description='P5 - GAN')
parser.add_argument('-w','--wandb', help='Disables wandb', required=False)
args = vars(parser.parse_args())

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01) #(tensor, mean, standard deviation)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Set random seed for reproducibility
def set_seed(manualSeed=999, makeRandom=False):
    if makeRandom:
        manualSeed = random.randint(1, 10000) 
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


def create_generator():
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    
    netG.apply(weights_init)

    return netG

def create_discriminator():
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # Apply the weights_init function to randomly initialize all weights
    netD.apply(weights_init)

    return netD

def labels_train_disc(netD, netG, images, img_labels, criterion, real_label, fake_label, noise, optimizerD, b_size):
    netD.zero_grad()
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    
    # Forward pass real batch through D
    output = netD((images, img_labels)).view(-1)
    
    # Calculate gradients for D in backward pass
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    fake = netG((noise, img_labels))
    label.fill_(fake_label)

    # Classify all fake batch with D
    output = netD((fake, img_labels)).view(-1)

    # Calculate the gradients for all-fake batch, accumulated (summed) with previous gradients
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake

    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1


def labels_train_gen(netD, netG, latent, img_labels, real_label, criterion, optimizerG, b_size):
    netG.zero_grad()

    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    # Generate fake images
    fake = netG((latent, img_labels))

    # Pass fake images to the discriminator
    output = netD((fake, img_labels)).view(-1)

    # Calculate G's loss based on this output
    errG = criterion(output, label)

    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()

    # Update G
    optimizerG.step()

    return errG, D_G_z2
    

def train(netD, netG, criterion, fake_label, real_label, optimizerD, optimizerG, fixed_noise, fixed_label):
    print("Starting Training Loop...")
    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(dataloader):
            print(f'Iteration [{i}/{len(dataloader)-1}]', end='\r', flush=True)
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            labels = labels.to(device)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            errD, D_x, D_G_z1 = labels_train_disc(netD, netG, real_cpu, labels, criterion, real_label, fake_label, noise, optimizerD, b_size)
            errG, D_G_z2 = labels_train_gen(netD, netG, noise, labels, real_label, criterion, optimizerG, b_size)
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch*len(dataloader)+i) % 15 == 0:
                with torch.no_grad():
                    fake = netG((fixed_noise, fixed_label)).detach().cpu()
                last_img = vutils.make_grid(fake, padding=2, normalize=True)
                # dataIO.save_last_image(f"Step {(epoch)*len(dataloader)+i}", last_img)
                if args['wandb'] is None:
                    wandb.log({"img": [wandb.Image(last_img, caption=f"Step {(epoch)*len(dataloader)+i}")], "D(x) in Discriminator": D_x, "D(G(z)) in Discriminator": D_G_z1, "D(G(z)) in Generator": D_G_z2, "Generator loss": errG.item(), "Discriminator loss": errD.item()})
            elif args['wandb'] is None:
                wandb.log({"D(x) in Discriminator": D_x, "D(G(z)) in Discriminator": D_G_z1, "D(G(z)) in Generator": D_G_z2, "Generator loss": errG.item(), "Discriminator loss": errD.item()})

        print(' ' * 20, end='\r')
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z))_D: %.4f\tD(G(z))_G: %.4f'
                % (epoch+1, num_epochs,
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
    return G_losses, D_losses

def main():
    print(f'running on: {device}')
    set_seed(999)
    netG = create_generator()
    netD = create_discriminator()

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Generate fixed labels for cGAN
    fixed_label = torch.tensor([[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3],[3],[3],[3],[3],[3],[4],[4],[4],[4],[4],[4],[4],[4],[5],[5],[5],[5],[5],[5],[5],[5],[6],[6],[6],[6],[6],[6],[6],[6],[7],[7],[7],[7],[7],[7],[7],[7]], device=device)
 
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=beta_params)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=beta_params)

    G_losses, D_losses = train(netD, netG, criterion, fake_label, real_label, optimizerD, optimizerG, fixed_noise, fixed_label)
    dataIO.create_loss_image(D_losses, G_losses)
    dataIO.save_cost(G_losses, D_losses, 'losses')
    dataIO.save_models(netD, netG, optimizerD, optimizerG, 'model')

if __name__ == "__main__":
    dataIO = DataIO('data', training_iteration)

    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats),
                            ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers, drop_last=True)

    if args['wandb'] is None:
        wandb.init(project='sp8dset', entity='p5_synthetic_gan', name='Model4_sameN_sameD')
        config = wandb.config
        config.workers = workers
        config.stats = stats
        config.batch_size = batch_size
        config.image_size = image_size
        config.nz = nz
        config.num_epochs = num_epochs
        config.lr = lr
        config.beta_params = beta_params
        config.disc_training_times = disc_training_times
        config.gen_training_times = gen_training_times
        config.embedded_dimension = embedded_dimension
        
    main()
