import torch

from torch.functional import Tensor
from constants import *

def index_to_class(class_to_idx_dic):
    values = []
    for k, _ in class_to_idx_dic.items():
        values.append(k)

    return values

def denormalize(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def calc_acc(softmax_output, targets):
    softmax_indices = torch.argmax(softmax_output, dim=1)
    correct = 0
    for i in range(len(softmax_indices)):
        if softmax_indices[i] == targets[i]:
            correct += 1
    
    return correct / len(softmax_indices)

def get_class_name(class_names, index):
    if isinstance(index, Tensor):
        return class_names[index.item()].split('car_')[-1]

    return class_names[index].split('car_')[-1]

def sort_predictions(predictions, targets):
    wrong_predictions = []
    correct_predictions = []

    assert len(predictions) - len(targets) == 0


    for i in range(len(predictions)):
        if predictions[i] != targets[i]:
            wrong_predictions.append((i, targets[i], predictions[i]))
        else:
            correct_predictions.append((i, targets[i], predictions[i]))

    return correct_predictions, wrong_predictions

def number_of_correct(predictions, targets, n_labels):
    correct_for_class = [0] * n_labels
    total_for_class = [0] * n_labels
    
    assert len(predictions) - len(targets) == 0
    
    for prediction, target in zip(predictions, targets):
        if prediction == target:
            correct_for_class[target] += 1
                
        total_for_class[target] += 1

    return correct_for_class, total_for_class