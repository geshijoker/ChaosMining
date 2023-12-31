import os
import sys
import csv
import copy
import time
import datetime
import random
import math
import numpy as np
import argparse
from PIL import Image

import torch
from torchvision import datasets, models, transforms

sys.path.append("/home/geshi/ChaosMining")
from chaosmining.utils import check_make_dir

# python3 create_vision_data.py --input_path ../../data --output_path ../data/vision

parser = argparse.ArgumentParser(description='Parse arguments to create vision data with irrelevant features')
parser.add_argument('--input_path', type=str, required=True,
                   help='Path to load raw data')
parser.add_argument('--output_path', type=str, required=True,
                   help='Path to save generated data')
args = parser.parse_args()

batch_size = 128
bg_size = 224
fg_size = 32
n_channels = 3

bg_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
    ]),
}

bg_trainset = datasets.Flowers102(args.input_path, split='train', transform=bg_data_transforms['train'], download=True)
bg_valset = datasets.Flowers102(args.input_path, split='val', transform=bg_data_transforms['val'], download=True)

fg_data_transforms = {
    'train': transforms.Compose([
    ]),
    'val': transforms.Compose([
    ]),
}

fg_trainset = datasets.CIFAR10(args.input_path, train=True, transform=fg_data_transforms['train'], download=True)
fg_valset = datasets.CIFAR10(args.input_path, train=False, transform=fg_data_transforms['val'], download=True)

# Random Background Fixed Position
train_path = os.path.join(args.output_path, 'RBFP', 'train')
check_make_dir(train_path)
val_path = os.path.join(args.output_path, 'RBFP', 'val')
check_make_dir(val_path)
fields = ['image', 'foreground_label', 'position_x', 'position_y']
filename = "meta_data.csv"

with open(os.path.join(train_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_trainset):
        foreground = np.array(img)
        fg_label = label
        background = np.random.randint(0, 256, (bg_size, bg_size, n_channels), dtype=np.uint8)
        pos = [fg_size*3, fg_size*3]
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(train_path, img_name)
        image_pil.save(img_path)

        csvwriter.writerow([img_name, str(fg_label), pos[0], pos[1]]) 
        
with open(os.path.join(val_path, filename), 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    for idx, (img, label) in enumerate(fg_valset):
        foreground = np.array(img)
        fg_label = label
        background = np.random.randint(0, 256, (bg_size, bg_size, n_channels), dtype=np.uint8)
        pos = [fg_size*3, fg_size*3]
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(val_path, img_name)
        image_pil.save(img_path)

        csvwriter.writerow([img_name, str(fg_label), pos[0], pos[1]])     

# Random Background Random Position
train_path = os.path.join(args.output_path, 'RBRP', 'train')
check_make_dir(train_path)
val_path = os.path.join(args.output_path, 'RBRP', 'val')
check_make_dir(val_path)
fields = ['image', 'foreground_label', 'position_x', 'position_y']
filename = "meta_data.csv"

with open(os.path.join(train_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_trainset):
        foreground = np.array(img)
        fg_label = label
        background = np.random.randint(0, 256, (bg_size, bg_size, n_channels), dtype=np.uint8)
        pos = np.random.randint(0, fg_size*6, 2, dtype=np.uint8)
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(train_path, img_name)
        image_pil.save(img_path)
        
        csvwriter.writerow([img_name, str(fg_label), pos[0], pos[1]]) 
        
with open(os.path.join(val_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_valset):
        foreground = np.array(img)
        fg_label = label
        background = np.random.randint(0, 256, (bg_size, bg_size, n_channels), dtype=np.uint8)
        pos = np.random.randint(0, fg_size*6, 2, dtype=np.uint8)
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(val_path, img_name)
        image_pil.save(img_path)
        
        csvwriter.writerow([img_name, str(fg_label), pos[0], pos[1]]) 

    
# Structural Background Fixed Position
train_path = os.path.join(args.output_path, 'SBFP', 'train')
check_make_dir(train_path)
val_path = os.path.join(args.output_path, 'SBFP', 'val')
check_make_dir(val_path)
fields = ['image', 'foreground_label', 'background_label', 'position_x', 'position_y']
filename = "meta_data.csv"

with open(os.path.join(train_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_trainset):
        foreground = np.array(img)
        fg_label = label
        bg_id = np.random.randint(0, len(bg_trainset))
        background = np.array(copy.deepcopy(bg_trainset[bg_id][0]))
        bg_label = bg_trainset[bg_id][1]
        pos = [fg_size*3, fg_size*3]
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(train_path, img_name)
        image_pil.save(img_path)

        csvwriter.writerow([img_name, str(fg_label), str(bg_label), pos[0], pos[1]]) 
        
with open(os.path.join(val_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_valset):
        foreground = np.array(img)
        fg_label = label
        bg_id = np.random.randint(0, len(bg_valset))
        background = np.array(copy.deepcopy(bg_valset[bg_id][0]))
        bg_label = bg_valset[bg_id][1]
        pos = [fg_size*3, fg_size*3]
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(val_path, img_name)
        image_pil.save(img_path)

        csvwriter.writerow([img_name, str(fg_label), str(bg_label), pos[0], pos[1]]) 
    
# Structural Background Random Position
train_path = os.path.join(args.output_path, 'SBRP', 'train')
check_make_dir(train_path)
val_path = os.path.join(args.output_path, 'SBRP', 'val')
check_make_dir(val_path)
fields = ['image', 'foreground_label', 'background_label', 'position_x', 'position_y']
filename = "meta_data.csv"

with open(os.path.join(train_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_trainset):
        foreground = np.array(img)
        fg_label = label
        bg_id = np.random.randint(0, len(bg_trainset))
        background = np.array(copy.deepcopy(bg_trainset[bg_id][0]))
        bg_label = bg_trainset[bg_id][1]
        pos = np.random.randint(0, fg_size*6, 2, dtype=np.uint8)
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(train_path, img_name)
        image_pil.save(img_path)
        
        csvwriter.writerow([img_name, str(fg_label), str(bg_label), pos[0], pos[1]]) 
        
with open(os.path.join(val_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for idx, (img, label) in enumerate(fg_valset):
        foreground = np.array(img)
        fg_label = label
        bg_id = np.random.randint(0, len(bg_valset))
        background = np.array(copy.deepcopy(bg_valset[bg_id][0]))
        bg_label = bg_valset[bg_id][1]
        pos = np.random.randint(0, fg_size*6, 2, dtype=np.uint8)
        background[pos[0]:pos[0]+fg_size, pos[1]:pos[1]+fg_size] = foreground

        image_pil = Image.fromarray(background)
        img_name = f'{idx:04d}.png'
        img_path = os.path.join(val_path, img_name)
        image_pil.save(img_path)
        
        csvwriter.writerow([img_name, str(fg_label), str(bg_label), pos[0], pos[1]]) 
