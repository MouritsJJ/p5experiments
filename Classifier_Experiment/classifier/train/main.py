import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import wandb
import argparse

from DataIO import DataIO
from models import *
from constants import *
from utils import *

parser = argparse.ArgumentParser(description='P5 - GAN')
parser.add_argument('-w','--wandb', help='Disables wandb', required=False)
args = vars(parser.parse_args())

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

idx_to_class = []

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

def create_classifier():
    # Create the Discriminator
    classifier = Classifier(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        classifier = nn.DataParallel(classifier, list(range(ngpu)))
        
    # Apply the weights_init function to randomly initialize all weights
    classifier.apply(weights_init)

    return classifier

def train_classifer(classifier, images, img_labels, criterion, optimizerD, b_size):
    classifier.zero_grad()
    
    # Forward pass real batch through D
    output = classifier(images).view(b_size, n_labels)
    
    # Calculate gradients for D in backward pass
    loss = criterion(output, img_labels)

    softmax_layer = nn.Softmax(dim=1)
    softmax_output = softmax_layer(output)
    acc = calc_acc(softmax_output, img_labels)
    loss.backward()

    optimizerD.step()
    return loss, acc, softmax_output
    


def train(classifier, criterion, optimizerC):
    print("Starting Training Loop...")
    # Lists to keep track of progress
    losses = []
    best_loss = 99999
    counter = 0

    for epoch in range(num_epochs):
        Acc_total = []
        for i, (data, labels) in enumerate(dataloader):
            print(f'Iteration [{i}/{len(dataloader)-1}]', end='\r', flush=True)
            images = data.to(device)
            b_size = images.size(0)
            labels = labels.to(device)

            loss, acc, softmax_output = train_classifer(classifier, images, labels, criterion, optimizerC, b_size)
            Acc_total.append(acc)
            losses.append(loss)
        
        for i in range(10):
            prediction_class = get_class_name(idx_to_class, torch.argmax(softmax_output[i]))
            target_class = get_class_name(idx_to_class, labels[i])
            denormalized_img = denormalize(images[i].cpu())
            dataIO.save_single_image(denormalized_img, f'result_{epoch+1}_img-{i}_p-{prediction_class}_t-{target_class}')

        print(' ' * 20, end='\r')
        print('[%d/%d]: Loss: %.4f Acc_avg: %.4f' % (epoch+1, num_epochs, loss.item(), mean(Acc_total)))

        print("Starting validation")
        validation_loss, accuracy = validate_classifier(classifier, criterion)

        if (validation_loss < best_loss):
            print(f"New best validation loss {validation_loss} with an accuracy of: {accuracy}")
            best_loss = validation_loss
            dataIO.save_classifier(classifier, optimizerC, 'best_model')
            counter = 0
        else:
            print(f'validation loss: {validation_loss}   best_loss: {best_loss}')
            counter = counter + 1
        
        if args['wandb'] is not None:
            wandb.log({"Validation loss": validation_loss, "validation_accuracy": accuracy, "training loss": loss, "Training accuracy": mean(Acc_total)})

        if counter > patience:
            print(f"Early stopped at Epoch {epoch+1}")
            return losses
    
    return losses

def validate_classifier(classifier, criterion):
    losses = []
    accuracies = []
    classifier.eval()

    with torch.no_grad():
        for i, (data, labels) in enumerate(validation_dataloader):
            print(f'Validation Iteration [{i}/{len(validation_dataloader)-1}]', end='\r', flush=True)
            images = data.to(device)
            b_size = images.size(0)
            labels = labels.to(device)

            output = classifier(images).view(b_size, n_labels)

            losses.append(criterion(output, labels))

            softmax_layer = nn.Softmax(dim=1)
            softmax_output = softmax_layer(output)
            acc = calc_acc(softmax_output, labels)
            accuracies.append(acc)

    print(' ' * 30)
    classifier.train()
    return mean(losses), mean(accuracies)

def main():
    print(f'running on: {device}')
    set_seed(999)
    classifier = create_classifier()

    criterion = nn.CrossEntropyLoss()
 
    optimizerC = optim.Adam(classifier.parameters(), lr=lr, betas=beta_params)

    _ = train(classifier, criterion, optimizerC)
    dataIO.save_classifier(classifier, optimizerC, 'last_model')

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

    validation_dataset = dset.ImageFolder(root=validation_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats),
                            ]))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers, drop_last=True)
                    

    idx_to_class = index_to_class(dataset.class_to_idx)
    
    print(idx_to_class)

    if args['wandb'] is not None:
        wandb.init(project=project_name, entity=entity_name, name=model_name)
        config = wandb.config
        config.workers = workers
        config.stats = stats
        config.batch_size = batch_size
        config.image_size = image_size
        config.num_epochs = num_epochs
        config.lr = lr
        config.beta_params = beta_params
        config.patience = patience
        
    main()
