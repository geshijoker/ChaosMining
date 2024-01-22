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
from torch.utils.data import Dataset, DataLoader

def parse_argument():
    parser = argparse.ArgumentParser(description='Parse Argument for Audio Experiment')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to the folder of audio files')
    parser.add_argument('--experiment', '-e', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--name', '-n', type=str, required=True, 
                        help='Name of run')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'],
                        help='Train split or val split')
    parser.add_argument('--model_name', '-m', type=str, required=True, choices = ['RNN', 'LSTM', 'TCN', 'Tran'], 
                        help='Name of the model architecture')
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
    parser.add_argument('--deterministic', action='store_true',
                        help='Using deterministic mode and disable benchmark algorithms')
    parser.add_argument('--debug', action='store_true',
                        help='Using debug mode')

    args = parser.parse_args()
    
    return args