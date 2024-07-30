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
python examples/train_eval_simulation_topk.py -d ./data/symbolic_simulation/formula.csv -e ./runs/topk_simulation/ -n 14 -s SEED --num_noises 100 --ny_var 0.01 --optimizer Adam --learning_rate 0.001 --deterministic --debug
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

Pred_scores=[]
DeepLift_scores=[]
FA_scores=[]
Saliency_scores=[] 
IG_scores=[]

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
    
    train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
    train_loader = DataLoader(train_set, batch_size=y_train.shape[0], shuffle=True)
    test_set = TensorDataset(Tensor(X_test), Tensor(y_test))
    test_loader = DataLoader(test_set, batch_size=y_test.shape[0], shuffle=False)

    model = MLPRegressor(num_features+num_noises, hidden_layer_sizes, p=dropout)    
    model.to(device)
    model.train()

    criterion = getattr(nn, loss_name)(reduction='mean')
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    print('Starting training loop; initial compile can take a while...')
    since = time.time()
    model.train()

    loss = train(model, train_loader, num_epochs, optimizer)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {loss:3f}')
    model.eval()

    y_pred = model(Tensor(X_test).to(device)).detach().cpu().numpy()
    mean_abs_y_diff, std_abs_y_diff = functions.mean_std_absolute_error(y_pred, y_true_test)
    Pred_score = functions.uniformity_score(y_pred, y_true_test)

    sa = Saliency(model)
    ig = IntegratedGradients(model)
    dl = DeepLift(model)
    fa = FeatureAblation(model)

    sa_attr_test = sa.attribute(Tensor(X_test).to(device))
    ig_attr_test = ig.attribute(Tensor(X_test).to(device), n_steps=10)
    dl_attr_test = dl.attribute(Tensor(X_test).to(device))
    fa_attr_test = fa.attribute(Tensor(X_test).to(device))

    sa_topk_inds = functions.abs_argmax_topk(sa_attr_test.detach().cpu().numpy(), num_features)
    ig_topk_inds = functions.abs_argmax_topk(ig_attr_test.detach().cpu().numpy(), num_features)
    dl_topk_inds = functions.abs_argmax_topk(dl_attr_test.detach().cpu().numpy(), num_features)
    fa_topk_inds = functions.abs_argmax_topk(fa_attr_test.detach().cpu().numpy(), num_features)

    Saliency_score = functions.top_features_score(sa_topk_inds, num_features)
    IG_score = functions.top_features_score(ig_topk_inds, num_features)
    DeepLift_score = functions.top_features_score(dl_topk_inds, num_features)
    FA_score = functions.top_features_score(fa_topk_inds, num_features)

    Pred_scores.append(Pred_score) 
    Saliency_scores.append(Saliency_score)
    IG_scores.append(IG_score)
    DeepLift_scores.append(DeepLift_score)
    FA_scores.append(FA_score)

    hparam_dict = {'formula_id':index, 'num_features':num_features, 'num_data':num_data, 'num_noises':num_noises, 'y_var':y_var}
    metric_dict = {'Pred':Pred_score, 'Saliency':Saliency_score, 'IG':IG_score, 'DeepLift':DeepLift_score, 'FA':FA_score}
    writer.add_hparams(hparam_dict, metric_dict)

hparam_dict = {'formula_id':'mean', 'num_features':'N/A', 'num_data':num_data, 'num_noises':num_noises, 'y_var':y_var}
metric_dict = {'Pred':np.mean(Pred_scores), 'Saliency':np.mean(Saliency_scores), 'IG':np.mean(IG_scores), 'DeepLift':np.mean(DeepLift_scores), 'FA':np.mean(FA_scores)}
writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
