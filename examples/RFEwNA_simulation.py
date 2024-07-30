import time
import datetime
import random
import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import numpy as np
from tqdm import tqdm, trange
from ptflops import get_model_complexity_info
from torchinfo import summary

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from chaosmining.data_utils import create_simulation_data, read_formulas
from chaosmining.simulation import parse_argument, functions
from chaosmining.simulation.models import MLPRegressor
from chaosmining.utils import check_make_dir

from captum.attr import IntegratedGradients, Saliency, DeepLift, FeatureAblation

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

"""
example command to run:
python examples/RFEwNA_simulation.py -d ./data/symbolic_simulation/formula.csv -e ./runs/RFEwNA -n rfe_ig -s SEED -g 0 --num_noises 100 --ny_var 0.01 --optimizer Adam --learning_rate 0.001 --dropout 0.0 --xai ig --deterministic --debug
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

if args.xai=='sa':
    xai_model = Saliency
elif args.xai=='ig':
    xai_model = IntegratedGradients
elif args.xai=='dl':
    xai_model = DeepLift
elif args.xai=='fa':
    xai_model = FeatureAblation
else:
    xai_model = None
    
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
    
test_ratio = args.test_ratio
loss_name = args.loss
optimizer_name = args.optimizer
lr = args.learning_rate
num_epochs = args.num_epochs
num_data = args.num_data
num_noises = args.num_noises
X_var = args.X_var
y_var = args.ny_var
arc_depth = args.arc_depth
arc_width = args.arc_width
dropout = args.dropout
n_steps = args.num_steps

formulas = read_formulas(args.data)
hidden_layer_sizes = [arc_width]*arc_depth

def train(model, dataloader, num_epochs, optimizer):
    pbar = trange(num_epochs, desc='Train', unit='epoch', initial=0, disable=not args.debug)
    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            outputs = model(inputs)
            # defining loss
            loss = criterion(outputs, targets)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        pbar.set_postfix(loss = '%.3f' % running_loss)
    return running_loss

writer = SummaryWriter(log_path)

for index, formula in enumerate(formulas):

    function = formula[1]
    num_features = formula[0]
    
    X, y_true, y_noise, intercepts, derivatives, integrations = create_simulation_data(function, num_features, num_noises, num_data, X_var, y_var, n_steps = n_steps)
    print('X', X.shape, 'y true', y_true.shape, 'y noise', y_noise.shape, 
          'intercepts', len(intercepts), intercepts[0].shape,
          'derivatives', len(derivatives), derivatives[0].shape, 
          'integrations', len(integrations), integrations[0].shape)

    intercepts = np.stack(intercepts, axis=1)
    derivatives = np.stack(derivatives, axis=1)
    integrations = np.stack(integrations, axis=1)
    y = y_true + y_noise

    X_train, X_test, \
    y_train, y_test, \
    y_true_train, y_true_test, \
    intercepts_train, intercepts_test, \
    derivatives_train, derivatives_test, \
    integrations_train, integrations_test \
    = train_test_split(X, y, y_true, intercepts, derivatives, integrations, test_size=test_ratio, random_state=seed)

    reduce_rate = 0.8
    best_score = 10
    num_cur_features = num_features+num_noises
    select_arr = np.ones(num_cur_features)

    while num_cur_features>0:
        bool_arr = np.array(select_arr, dtype='bool') 
        
        train_set = TensorDataset(Tensor(X_train[...,bool_arr]), Tensor(y_train))
        train_loader = DataLoader(train_set, batch_size=y_train.shape[0], shuffle=True)
    
        model = MLPRegressor(int(np.sum(select_arr)), hidden_layer_sizes, p=0.0)
        model.to(device)
        model.train()
    
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
    
        train(model, train_loader, num_epochs, optimizer)
    
        model.eval()
        y_pred = model(Tensor(X_test[...,bool_arr]).to(device)).detach().cpu().numpy()
        score = mean_absolute_error(y_pred, y_test)
    
        if score > best_score:
            break
        else:
            best_score = score
    
        if not xai_model:
            break
        else:
            xai = xai_model(model)
    
        num_remove = int(num_cur_features*(1-reduce_rate))
        if num_remove<1:
            break
        
        xai_attr_test = xai.attribute(Tensor(X_test[...,bool_arr]).to(device))
        abs_xai_attr_test = np.abs(xai_attr_test.detach().cpu().numpy()).mean(0)
        
        inds = np.argpartition(abs_xai_attr_test, num_remove)[:num_remove]
        inds_to_remove = remaining_inds[inds]
        select_arr[inds_to_remove] = 0
        
        remaining_inds = np.nonzero(select_arr)[0]
        num_cur_features -= num_remove

    print('formula', index, 'score', best_score, 'selected', np.where(select_arr==1)[0])
    
    hparam_dict = {'formula_id':index, 'num_features':num_features, 'num_data':num_data, 'num_noises':num_noises, 'y_var':y_var, 'xai':args.xai}
    metric_dict = {'score':best_score, 'fprec': np.sum(select_arr[:num_features])/np.sum(select_arr), 'preset': num_features/(num_features+num_noises)}
    writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
