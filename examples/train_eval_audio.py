import time
import os
import sys
import copy
import datetime
import random
import math
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm, trange
from thop import profile, clever_format
from torchinfo import summary

sys.path.append("/home/geshi/ChaosMining")
from chaosmining.data_utils import ChaosAudioDataset
from chaosmining.utils import check_make_dir
from chaosmining.audio.models import *
from chaosmining.audio import parse_argument, train_epoch, test
from chaosmining.audio.functions import *

"""
example command to run:
python examples/train_eval_audio.py -d /data/home/geshi/ChaosMining/data/audio/RBFP/ -e /data/home/geshi/ChaosMining/runs/audio/RBFP/ -n arc_TRAN -s 9999 --model_name TRAN --n_channels 10 --length 16000 --gpu 0 --num_epochs 30 --batch_size 128 --learning_rate 0.0001 --deterministic --debug
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
    
batch_size = args.batch_size
num_epochs = args.num_epochs
n_channels = args.n_channels
lr = args.learning_rate
length = args.length

# define datasets

train_set = ChaosAudioDataset(args.data, "train", "meta_data.csv")
val_set = ChaosAudioDataset(args.data, "val", "meta_data.csv")

num_classes = len(train_set.classes)

# define dataloaders
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
)

# create model
if args.model_name == "RNN":
    model = AudioRNN(n_channels, num_classes, hidden_dim=60, n_layers=3)
elif args.model_name == "LSTM":
    model = AudioLSTM(n_channels, num_classes, hidden_dim=60, n_layers=3)
elif args.model_name == "TCN":
    model = AudioTCN(n_channels, num_classes, n_channel=60)
elif args.model_name == "TRAN":
    model = AudioTrans(n_channels, num_classes, hidden_dim=60, n_layers=3)
else:
    sys.exit("The model {} is not supported".format(args.model_name))

# sanity check
sample_shape = (batch_size, n_channels, length)
sample = torch.rand(*sample_shape)
model.eval()
out = model(sample)
print('sample output', out.shape)
summary(model, input_size=sample_shape)

flops, params = profile(model.cpu(), inputs=(sample.cpu(),))
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Parameters: {params}")

model.to(device)
model.train()

# prepare for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 20)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30)

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
    model, train_stats = train_epoch(model, train_loader, num_classes, criterion, optimizer, scheduler, device, args.debug)

    writer.add_scalar('time eplased', time.time() - since, epoch)
    for stat in train_stats:
        writer.add_scalar(stat, train_stats[stat], epoch)

    pbar.set_postfix(loss=train_stats['train_loss'], acc=train_stats['train_acc'])

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {train_stats["train_loss"]}, acc: {train_stats["train_acc"]}')

model.eval()
with torch.no_grad():
    val_stats = test(model, val_loader, num_classes, device, (1, 5), args.debug) 

    hparam_dict = {'model_architecture':args.model_name, 'learning_rate':lr, 'batch_size':batch_size}
    metric_dict = val_stats
    print(metric_dict)
    writer.add_hparams(hparam_dict, metric_dict)
    save_checkpoint()

writer.flush()
writer.close()
