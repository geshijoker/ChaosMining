import time
import datetime
import random
import sys
import os
import argparse

import numpy as np
from tqdm import tqdm, trange
from ptflops import get_model_complexity_info
from torchinfo import summary
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

from chaosmining.data_utils import ChaosVisionDataset
from chaosmining.utils import check_make_dir
from chaosmining.vision import parse_argument, train_epoch, test
from chaosmining.vision.contribs import *
from chaosmining.vision.models import resnet18, resnet50

from captum.attr import IntegratedGradients, Saliency, DeepLift, FeatureAblation, visualization
from scipy.ndimage import gaussian_filter 
"""
example command to run:
python examples/eval_vision_localization.py -d ./data/vision/RBRP/ -e ./runs/vision/RBRP/ -n arc_vit_b_16 -s SEED --model_name vit_b_16 --gpu 1 --num_classes 10 --batch_size 2 --deterministic --debug
"""

# load and parse argument
args = parse_argument()

if args.gpu<0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    if args.gpu<torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda") 
print('Using device: {}'.format(device))

# set up the seed
if args.seed:
    seed = args.seed
else:
    seed = torch.seed()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

experiment = args.experiment
run_name = args.name + f'_seed_{seed}'
log_path = os.path.join(experiment, run_name)

if not os.path.isdir(log_path):
    sys.exit('The name of the run does not exist')

# set up benchmark running
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)
    sys.stdout = open(os.path.join(log_path, 'log.txt'), 'a+')

if 'resnet' in args.model_name:
    model = eval(args.model_name)()
else:
    model =  models.get_model(args.model_name, weights='DEFAULT')
    
batch_size = args.batch_size
num_classes = args.num_classes
sigma = 6
n_steps = 20
fg_size = (32, 32)

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

target_transform = transforms.Compose([
        ToTensor(),
    ])

# load data
root_dir = args.data
val_data = os.path.join(root_dir, 'val')
val_csv_file = os.path.join(val_data, 'metadata.csv')
valset = ChaosVisionDataset(val_data, val_csv_file, transform=data_transform, target_transform=target_transform)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

target_names = valset.get_target_names()
target_index = target_names.index('foreground_label')

# prepare for training
model = nn.Sequential(
    model,
    nn.Linear(1000, num_classes)
)
print(model)

replace_relu_with_inplace_relu(model)

sample_shape = (1, 3, 224, 224)
sample = torch.rand(*sample_shape)
model.eval()
out = model(sample)
print('sample output', out.shape)
summary(model, input_size=sample_shape)
model.to(device)

# load pretrained model
param_files = [f for f in os.listdir(log_path) if f.endswith('.pt')]
print('model', os.path.join(log_path, param_files[0]))
model.load_state_dict(torch.load(os.path.join(log_path, param_files[0]), map_location=device)['model_state_dict'])
model.eval()

ig = IntegratedGradients(model)
sa = Saliency(model)
dl = DeepLift(model)
fa = FeatureAblation(model)

writer = SummaryWriter(log_path)
sa_ious, ig_ious, dl_ious, fa_ious = [], [], [], []

with torch.no_grad():
    val_stats = test(model, val_loader, target_index, num_classes, device, (1, 5), args.debug) 
    print(val_stats)
    count = 0
    
    piter = tqdm(val_loader, desc='Test', unit='batch', disable=not args.debug)
    for inputs, targets in piter:

        inputs = inputs.to(device)
        target = targets[target_index].to(device)
        positions = (targets[-2].numpy(), targets[-1].numpy())
        
        count += inputs.size(0)

        outputs = model(inputs)
        _, preds = outputs.topk(1)
        
        sa_attr = sa.attribute(inputs, target=preds.view(-1))
        sa_attr_sample = np.transpose(sa_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_sa_attr_sample = gaussian_filter(sa_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_sa_attr_sample, np.prod(fg_size))
        sa_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        sa_ious.extend(sa_iou)
        
        ig_attr = ig.attribute(inputs, 0*torch.ones_like(inputs).to(device), target=preds.view(-1), n_steps=n_steps)
        ig_attr_sample = np.transpose(ig_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_ig_attr_sample = gaussian_filter(ig_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_ig_attr_sample, np.prod(fg_size))
        ig_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        ig_ious.extend(ig_iou)
        
        dl_attr = dl.attribute(inputs, 0*torch.ones_like(inputs).to(device), target=preds.view(-1))
        dl_attr_sample = np.transpose(dl_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_dl_attr_sample = gaussian_filter(dl_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_dl_attr_sample, np.prod(fg_size))
        dl_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        dl_ious.extend(dl_iou)
        
        feature_mask = np.arange(49)
        feature_mask = feature_mask.reshape((7,7,1)).repeat((32), axis=-1).reshape(7, 224)
        feature_mask = np.tile(np.expand_dims(feature_mask, 1), (32, 1)).reshape(224, 224)
        feature_mask=torch.from_numpy(feature_mask)
        fa_attr = fa.attribute(inputs, 0*torch.ones_like(inputs).to(device), target=preds.view(-1), feature_mask=feature_mask.to(device))
        fa_attr_sample = np.transpose(fa_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_fa_attr_sample = gaussian_filter(fa_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_fa_attr_sample, np.prod(fg_size))
        fa_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        fa_ious.extend(fa_iou)
        
    avg_sa_iou = np.mean(sa_ious)
    avg_ig_iou = np.mean(ig_ious)
    avg_dl_iou = np.mean(dl_ious)
    avg_fa_iou = np.mean(fa_ious)

    hparam_dict = {'model_architecture':args.model_name}
    metric_dict = val_stats
    metric_dict['sa_iou'] = avg_sa_iou
    metric_dict['ig_iou'] = avg_ig_iou
    metric_dict['dl_iou'] = avg_dl_iou
    metric_dict['fa_iou'] = avg_fa_iou
    
    writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
