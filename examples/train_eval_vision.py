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
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/geshi/ChaosMining")

from chaosmining.data_utils import ChaosVisionDataset
from chaosmining.utils import check_make_dir
from chaosmining.vision import parse_argument, train_epoch, test
"""
example command to run:
python examples/train_eval_vision.py -d data/vision/RBFP/ -e runs/vision/RBFP/ -n resnet18 -s 9999 --model_name resnet18 --gpu 0 --num_classes 10 --num_epochs 20 --batch_size 128 --learning_rate 0.001 --pretrained --deterministic --debug
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

if os.path.isdir(log_path):
    sys.exit('The name of the run has alrealy exist')
else:
    check_make_dir(log_path)

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
    sys.stdout = open(os.path.join(log_path, 'log.txt'), 'w')
    
if args.pretrained:
    model =  models.get_model(args.model_name, weights="DEFAULT")
else:
    model =  models.get_model(args.model_name, weights=None)
    
batch_size = args.batch_size
num_epochs = args.num_epochs
num_classes = args.num_classes
lr = args.learning_rate

# define transforms

class ToTensor(object):
    """Convert values in landmarks to Tensors."""
    def __call__(self, landmarks):
        return [torch.tensor(landmark) for landmark in landmarks]

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

target_transform = transforms.Compose([
        ToTensor(),
    ])

# load data
root_dir = args.data
train_data = os.path.join(root_dir, 'train')
train_csv_file = os.path.join(train_data, 'meta_data.csv')
trainset = ChaosVisionDataset(train_data, train_csv_file, transform=data_transform, target_transform=target_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

val_data = os.path.join(root_dir, 'val')
val_csv_file = os.path.join(val_data, 'meta_data.csv')
valset = ChaosVisionDataset(val_data, val_csv_file, transform=data_transform, target_transform=target_transform)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

target_names = trainset.get_target_names()
target_index = target_names.index('foreground_label')

# prepare for training
model = nn.Sequential(
    model,
    nn.Linear(1000, num_classes)
)

sample_shape = (1, 3, 224, 224)
sample = torch.rand(*sample_shape)
model.eval()
out = model(sample)
print('sample output', out.shape)
summary(model, input_size=sample_shape)

model.to(device)
model.train()

# prepare for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)
# scheduler = lr_scheduler.StepLR(optimizer, 20)

print('Starting training loop; initial compile can take a while...')
since = time.time()
start_epoch = 0
writer = SummaryWriter(log_path)

def save_checkpoint():
    utctime = datetime.datetime.now(datetime.timezone.utc).strftime("%m-%d-%Y-%H:%M:%S")
    model_path = os.path.join(log_path, utctime+'.pt')
    torch.save({'model_state_dict': model.state_dict()}, model_path)

pbar = trange(num_epochs, desc='Train', unit='epoch', initial=start_epoch, position=0, disable=not args.debug)
# Iterate over data.
for epoch in pbar:
    model, train_stats = train_epoch(model, train_loader, target_index, num_classes, criterion, optimizer, scheduler, device, args.debug)

    writer.add_scalar('time eplased', time.time() - since, epoch)
    for stat in train_stats:
        writer.add_scalar(stat, train_stats[stat], epoch)

    pbar.set_postfix(loss=train_stats['train_loss'], acc=train_stats['train_acc'])

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {train_stats["train_loss"]}, acc: {train_stats["train_acc"]}')

model.eval()
with torch.no_grad():
    val_stats = test(model, val_loader, target_index, num_classes, device, (1, 5), args.debug) 

    hparam_dict = {'model_architecture':args.model_name, 'learning_rate':lr, 'batch_size':batch_size}
    metric_dict = val_stats
    writer.add_hparams(hparam_dict, metric_dict)
    save_checkpoint()

writer.flush()
writer.close()