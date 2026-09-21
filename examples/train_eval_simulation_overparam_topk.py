import time
import random
import sys
import os
import argparse

import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chaosmining.data_utils import create_simulation_data, read_formulas
from chaosmining.simulation import parse_argument, functions
from chaosmining.simulation.models import MLPResRegressor
from chaosmining.utils import check_make_dir

from captum.attr import Saliency

from sklearn.model_selection import train_test_split

"""
python run_experiment.py -d /menglinliu/ChaosMining/data/symbolic_simulation/formula.csv -e ./data/ -n 14 -s 9999 --num_noises 100 --ny_var 0.01 --optimizer Adam --learning_rate 0.001 --deterministic --debug
"""

args = parse_argument()

if args.gpu < 0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.gpu}' if args.gpu < torch.cuda.device_count() else "cuda")
print('Using device:', device)

seed = args.seed if args.seed else torch.seed()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

experiment = args.experiment
run_name = args.name + f'_seed_{seed}'
log_path = os.path.join(experiment, run_name)
plot_path = os.path.join(log_path, "sa_curves")

if os.path.isdir(log_path):
    sys.exit('The name of the run already exist')

check_make_dir(log_path)
check_make_dir(plot_path)

if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

if not args.debug:
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
hidden_layer_sizes = [arc_width] * arc_depth

# 只训练 + 每 epoch 记录每个 feature 的 SA
def train_and_record_sa(model, loader, X_test, n_sig, epochs, opt):
    sa = Saliency(model)
    sa_history = [[] for _ in range(n_sig)]
    pbar = trange(epochs, desc='Train', unit='epoch')

    for ep in pbar:
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # 只算 SA
        model.eval()
        with torch.no_grad():
            xt = Tensor(X_test).to(device)
        attr = sa.attribute(xt)
        mean_abs = np.abs(attr.cpu().numpy()).mean(0)

        for f in range(n_sig):
            sa_history[f].append(mean_abs[f])

        pbar.set_postfix(loss=f'{total_loss:.2f}')

    return sa_history

def plot_sa(sa_hist, formula_idx, save_path):
    plt.figure(figsize=(12,6))
    epochs = np.arange(1, len(sa_hist[0])+1)
    for i, vals in enumerate(sa_hist):
        plt.plot(epochs, vals, linewidth=2, label=f'Feature {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('|Saliency| (mean)')
    plt.title(f'Formula {formula_idx}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'formula_{formula_idx}_sa.png'), dpi=300)
    plt.close()

for idx, formula in enumerate(formulas):
    print(f'\n=== Formula {idx} ===')
    func = formula[1]
    n_sig = formula[0]

    X, yt, yn, *_ = create_simulation_data(
        func, n_sig, num_noises, num_data, X_var, y_var, n_steps=n_steps
    )
    y = yt + yn

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, yt, test_size=test_ratio, random_state=seed
    )

    train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
    train_loader = DataLoader(train_set, batch_size=y_train.shape[0], shuffle=True)

    model = MLPResRegressor(n_sig + num_noises, hidden_layer_sizes, p=dropout).to(device)
    criterion = getattr(nn, loss_name)()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    sa_hist = train_and_record_sa(model, train_loader, X_test, n_sig, num_epochs, optimizer)
    plot_sa(sa_hist, idx, plot_path)

print('\nDone. Only SA figures saved.')
if not args.debug:
    sys.stdout.close()