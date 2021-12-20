import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from DataIO import DataIO
from models import *
from constants import *
from utils import *

dataset = dset.ImageFolder(root=imgs_path,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats),
                            ]))

testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

idx_to_class = index_to_class(dataset.class_to_idx)
dataIO = DataIO('data', training_iteration)

"""
Function heavily inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (Accessed 19/12-2021)
with only small adjustments
"""
def create_classifier():
    # Create the Discriminator
    classifier = Classifier(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        classifier = nn.DataParallel(classifier, list(range(ngpu)))

    return classifier

def classify(classifier):
    correct_for_class = [0] * n_labels
    total_for_class = [0] * n_labels
    images_processed = 0

    with torch.no_grad():
        softmax_layer = nn.Softmax(dim=1)
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            images_processed += images.size(0)
            outputs = classifier(images)
            softmax_outputs = softmax_layer(outputs)
            predictions = torch.argmax(softmax_outputs, dim=1).view(images.size(0))
            
            correct, total = number_of_correct(predictions, labels, n_labels)
            correct_for_class = [sum(x) for x in zip(correct_for_class, correct)]
            total_for_class = [sum(x) for x in zip(total_for_class, total)]

            #correct_preds, wrong_preds = sort_predictions(predictions, labels)
            #dataIO.save_prediction_images(correct_preds, wrong_preds, images)

            acc = [0] * n_labels
            for i in range(n_labels):
                if total_for_class[i] != 0:
                    acc[i] = round(correct_for_class[i] / total_for_class[i] * 100, 4)
            
            print(f'Classified [{images_processed}/{len(dataset)}]     correct_for: {acc}', end='\r', flush=True)
    
    print(f'Classified [{images_processed}/{len(dataset)}]     correct_for: {acc}', flush=True)

    print('Class | Accuraccy')
    print('-----------------')
    for i in range(n_labels):
        acc = round(correct_for_class[i] / total_for_class[i] * 100, 4)
        
        print('%-5s | %8.4f' % (get_class_name(idx_to_class, i), acc))

def main():
    classifier = create_classifier()

    optimizerC = optim.Adam(classifier.parameters(), lr=lr, betas=beta_params)
    
    dataIO.load_classifier(classifier, optimizerC, model_path)
    print('Loaded classifier')

    classify(classifier)

main()
