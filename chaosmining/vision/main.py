import sys
import os
import csv
import time
import argparse
from typing import Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

def parse_argument():
    parser = argparse.ArgumentParser(description='Parse Argument for Vision Experiment')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to the file saving the formulas')
    parser.add_argument('--experiment', '-e', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--name', '-n', type=str, required=True, 
                        help='Name of run')
    parser.add_argument('--model_name', '-m', type=str, required=True, choices = ['alexnet', 'googlenet', 'densenet121', 'resnet18', 'resnet50', 'vgg13', 'vit_b_16', 'vit_l_32'], help='Name of the model architecture')
    parser.add_argument('--seed', '-s', type=int, default=None, 
                        help='which seed for random number generator to use')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='which GPU to use, negative value denotes cpu will be used')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='the number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size of data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use the pretrained weights of the model')
    parser.add_argument('--deterministic', action='store_true',
                        help='Using deterministic mode and disable benchmark algorithms')
    parser.add_argument('--debug', action='store_true',
                        help='Using debug mode')

    args = parser.parse_args()
    
    return args

def topk_corrects(output, target, topk=(1,)):
    """Computes the number of corrects @k for the specified values of k.
       topk should be a tuple of integers in ascending order.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].sum()
        res.append(correct_k.double().item())
    return res
    
def train_epoch(model, dataloader, target_index, num_classes, criterion, optimizer, scheduler, device, verbose=True):
    epoch_loss = 0.0
    epoch_acc = 0
    count = 0

    piter = tqdm(dataloader, desc='Epoch', unit='batch', position=1, leave=False, disable=not verbose)
    for inputs, targets in piter:

        inputs = inputs.to(device)
        target = targets[target_index].to(device)

        batch_size = inputs.size(0)
        nxt_count = count+batch_size
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_loss = loss.item() * batch_size/nxt_count + epoch_loss * count/nxt_count
        epoch_acc = ((preds == target).sum()/np.prod(preds.size())).item() * batch_size/nxt_count + epoch_acc * count/nxt_count
        count = nxt_count
        piter.set_postfix(accuracy=100. * epoch_acc, loss=epoch_loss)

    scheduler.step()
    train_stats = {
        'train_loss': epoch_loss,
        'train_acc': epoch_acc * 100,
    }
    
    return model, train_stats

def test(model, dataloader, target_index, num_classes, device, topk: Tuple[int, int]=(1,), verbose=True):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = [0]*len(topk)
    count = 0

    # Iterate over data.
    with torch.no_grad():
        piter = tqdm(dataloader, desc='Test', unit='batch', disable=not verbose)
        for inputs, targets in piter:

            inputs = inputs.to(device)
            target = targets[target_index].to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            batch_corrects = topk_corrects(outputs, target, topk=topk)

            # statistics
            corrects = [correct + batch_correct for correct, batch_correct in zip(corrects, batch_corrects)]
            accs = [correct / count for correct in corrects]
            piter.set_postfix(accuracy=100. * accs[0])

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * accs[0]}')
    
    test_stats = {}
    for i in range(len(topk)):
        test_stats[f'test_top{topk[i]}_acc'] = 100.*accs[i]

    return test_stats