import time
import os
import sys
import copy
import datetime
import random
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm, trange
from torchinfo import summary

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

from chaosmining.data_utils import ChaosAudioDataset
from chaosmining.utils import check_make_dir
from chaosmining.audio.models import *
from chaosmining.audio import parse_argument, test
from chaosmining.audio.functions import *

from captum.attr import IntegratedGradients, Saliency, DeepLift, FeatureAblation
"""
example command to run:
python examples/eval_audio_localization.py -d ./data/audio/RBFP/ -e ./runs/audio/RBFP/ -n arc_RNN -s SEED --model_name RNN --n_channels 10 --length 16000 --gpu 0 --batch_size 32 --deterministic --debug
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

batch_size = args.batch_size
n_channels = args.n_channels
length = args.length

# define datasets
train_set = ChaosAudioDataset(args.data, "train", "metadata.csv")
val_set = ChaosAudioDataset(args.data, "val", "metadata.csv")

num_classes = len(train_set.classes)

# define dataloaders
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
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
model.to(device)

# load pretrained model
param_files = [f for f in os.listdir(log_path) if f.endswith('.pt')]
print('model', os.path.join(log_path, param_files[0]))
model.load_state_dict(torch.load(os.path.join(log_path, param_files[0]), map_location=device)['model_state_dict'], strict=False)
model.eval()

ig = IntegratedGradients(model)
sa = Saliency(model)
dl = DeepLift(model)
fa = FeatureAblation(model)

writer = SummaryWriter(log_path)
sa_scores, ig_scores, dl_scores, fa_scores = [], [], [], []

with torch.no_grad():
    val_stats = test(model, val_loader, num_classes, device, (1, 5), args.debug) 
    print(val_stats)
    count = 0

    model.train()
    piter = tqdm(val_loader, desc='Test', unit='batch', disable=not args.debug)
    for inputs, targets, pos, _ in piter:
        inputs = inputs.to(device)
        targets = targets.to(device)
        pos = pos.tolist()

        count += inputs.size(0)

        outputs = model(inputs)
        _, preds = outputs.topk(1)

        sa_attr = sa.attribute(inputs, preds.squeeze())
        sa_ma = sa_attr.abs().mean(-1).detach().cpu().numpy()
        sa_selected = sa_ma[np.arange(inputs.size(0)), pos]
        sa_ra = sa_selected/np.linalg.norm(sa_ma, 1, axis=-1)
        sa_score = np.mean(sa_ra)
        sa_scores.append(sa_score)

        ig_attr = ig.attribute(inputs, torch.zeros_like(inputs).to(device), preds.squeeze(), n_steps=10)
        ig_ma = ig_attr.abs().mean(-1).detach().cpu().numpy()
        ig_selected = ig_ma[np.arange(inputs.size(0)), pos]
        ig_ra = ig_selected/np.linalg.norm(ig_ma, 1, axis=-1)
        ig_score = np.mean(ig_ra)
        ig_scores.append(ig_score)

        dl_attr = dl.attribute(inputs, torch.zeros_like(inputs).to(device), preds.squeeze())
        dl_ma = dl_attr.abs().mean(-1).detach().cpu().numpy()
        dl_selected = dl_ma[np.arange(inputs.size(0)), pos]
        dl_ra = ig_selected/np.linalg.norm(dl_ma, 1, axis=-1)
        dl_score = np.mean(dl_ra)
        dl_scores.append(dl_score)

        feature_mask = np.arange(n_channels)
        feature_mask = feature_mask[np.newaxis,:,np.newaxis]
        feature_mask = feature_mask.repeat(length, axis=-1).repeat(inputs.size(0), axis=0)
        feature_mask = torch.from_numpy(feature_mask)
        fa_attr = fa.attribute(inputs, torch.zeros_like(inputs).to(device), target=preds.squeeze(), feature_mask=feature_mask.to(device))
        fa_ma = fa_attr.abs().mean(-1).detach().cpu().numpy()
        fa_selected = fa_ma[np.arange(inputs.size(0)), pos]
        fa_ra = fa_selected/np.linalg.norm(fa_ma, 1, axis=-1)
        fa_score = np.mean(fa_ra)
        fa_scores.append(fa_score)
        
    avg_sa_score = np.mean(sa_scores)
    avg_ig_score = np.mean(ig_scores)
    avg_dl_score = np.mean(dl_scores)
    avg_fa_score = np.mean(fa_scores)

    hparam_dict = {'model_architecture':args.model_name}
    metric_dict = val_stats
    metric_dict['sa_score'] = avg_sa_score
    metric_dict['ig_score'] = avg_ig_score
    metric_dict['dl_score'] = avg_dl_score
    metric_dict['fa_score'] = avg_fa_score
    
    writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
