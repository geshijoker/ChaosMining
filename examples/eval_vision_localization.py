import time
import datetime
import random
import sys
import os
import copy
import argparse
from thop import profile, clever_format
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from chaosmining.data_utils import ChaosVisionDataset
from chaosmining.data_utils import ChaosVisionHFDataset
from chaosmining.utils import check_make_dir
from chaosmining.vision import parse_argument, train_epoch, test
from chaosmining.vision.contribs import *
from chaosmining.vision.models import resnet18, resnet50

from captum.attr import IntegratedGradients, Saliency, DeepLift, FeatureAblation, visualization, Lime, KernelShap, LayerGradCam, Occlusion, LRP, Deconvolution, ShapleyValueSampling, GradientShap, GuidedBackprop, NoiseTunnel,FeaturePermutation,DeepLiftShap
from scipy.ndimage import gaussian_filter 
os.environ["HF_TOKEN"] =""
os.environ["HF_ENDPOINT"] = ""
"""
example command to run:
python eval_vision_localization.py -c RBRP -e ./results/vision/RBRP/ -n arc_vit_b_16 -s 42 --model_name vit_b_16 --gpu 0 --num_classes 10 --batch_size 128 --deterministic --debug
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
load data
root_dir = args.data
val_data = os.path.join(root_dir, 'val')
val_csv_file = os.path.join(val_data, 'metadata.csv')
valset = ChaosVisionDataset(val_data, val_csv_file, transform=data_transform, target_transform=target_transform)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
valset = ChaosVisionHFDataset(
    hf_dataset_name="geshijoker/chaosmining",
    config_name="vision_" + args.cfg_suffix,
    split="validation",
    transform=data_transform,
    target_transform=target_transform
)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
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
input = torch.randn(1, 3, 224, 224).to(device)
model_copy = copy.deepcopy(model)
flops, params = profile(copy.deepcopy(model), inputs=(input,), verbose=False)
gflops = flops / 1e9
params_m = params / 1e6

print(f"Model Params: {params_m:.2f} M")
print(f"Model GFLOPs: {gflops:.2f} GFLOPs")
# load pretrained model
param_files = [f for f in os.listdir(log_path) if f.endswith('.pt')]
print('model', os.path.join(log_path, param_files[0]))
model.load_state_dict(torch.load(os.path.join(log_path, param_files[0]), map_location=device)['model_state_dict'])
model.eval()

ig = IntegratedGradients(model)
sa = Saliency(model)
dl = DeepLift(model)
fa = FeatureAblation(model)
guided_backprop = GuidedBackprop(model)
deep_lift_shap = DeepLiftShap(model)
lime = Lime(model)
ks = KernelShap(model)

writer = SummaryWriter(log_path)
writer.add_scalar('model/params_M', params_m)
writer.add_scalar('model/GFLOPs', gflops)
sa_ious, ig_ious, dl_ious, fa_ious = [], [], [], []
guided_backprop_ious = []
deep_lift_shap_ious = []
lime_scores, ks_scores = [], []

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
        
        lime_attr = lime.attribute(inputs, target=preds.view(-1), n_samples=100)
        lime_attr_sample = np.transpose(lime_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_lime_attr_sample = gaussian_filter(lime_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_lime_attr_sample, np.prod(fg_size))
        lime_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        lime_scores.extend(lime_iou)

        ks_attr = ks.attribute(inputs, target=preds.view(-1), n_samples=100)
        ks_attr_sample = np.transpose(ks_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_ks_attr_sample = gaussian_filter(ks_attr_sample, sigma=sigma, axes=(1,2))
        min_x, max_x, min_y, max_y = get_multi_box(smooth_ks_attr_sample, np.prod(fg_size))
        ks_iou = calculate_multi_iou((min_x, min_y, max_x-min_x, max_y-min_y), (positions[0], positions[1], fg_size[0]*np.ones(batch_size), fg_size[1]*np.ones(batch_size)))
        ks_scores.extend(ks_iou)

        guided_backprop_attr = guided_backprop.attribute(
            inputs,
            target=preds.view(-1)
        )
        guided_backprop_attr_sample = np.transpose(guided_backprop_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_guided_backprop_attr_sample = gaussian_filter(guided_backprop_attr_sample, sigma=sigma, axes=(1,2))
        bs, h, w, c = smooth_guided_backprop_attr_sample.shape
        min_x, max_x, min_y, max_y = get_multi_box(smooth_guided_backprop_attr_sample, np.prod(fg_size))
        guided_backprop_iou = calculate_multi_iou(
            (min_x, min_y, max_x - min_x, max_y - min_y),
            (positions[0], positions[1], fg_size[0]*np.ones(bs), fg_size[1]*np.ones(bs))
        )
        guided_backprop_ious.extend(guided_backprop_iou)
        deep_lift_shap_attr = deep_lift_shap.attribute(
            inputs,
            target=preds.view(-1),
            baselines=inputs * 0
        )
        deep_lift_shap_attr_sample = np.transpose(deep_lift_shap_attr.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
        smooth_deep_lift_shap_attr_sample = gaussian_filter(deep_lift_shap_attr_sample, sigma=sigma, axes=(1,2))
        bs, h, w, c = smooth_deep_lift_shap_attr_sample.shape
        min_x, max_x, min_y, max_y = get_multi_box(smooth_deep_lift_shap_attr_sample, np.prod(fg_size))
        deep_lift_shap_iou = calculate_multi_iou(
            (min_x, min_y, max_x - min_x, max_y - min_y),
            (positions[0], positions[1], fg_size[0]*np.ones(bs), fg_size[1]*np.ones(bs))
        )
        deep_lift_shap_ious.extend(deep_lift_shap_iou)
    avg_lrp_iou = np.mean(lrp_ious)
    avg_deconvolution_iou = np.mean(deconvolution_ious)
    avg_feature_permutation_iou = np.mean(feature_permutation_ious)
    avg_guided_backprop_iou = np.mean(guided_backprop_ious)
    avg_deep_lift_shap_iou = np.mean(deep_lift_shap_ious)
    avg_guided_backprop_iou = np.mean(guided_backprop_ious)
    avg_sa_iou = np.mean(sa_ious)
    avg_ig_iou = np.mean(ig_ious)
    avg_dl_iou = np.mean(dl_ious)
    avg_fa_iou = np.mean(fa_ious)
    var_deep_lift_shap_iou = np.var(deep_lift_shap_ious)
    var_guided_backprop_iou = np.var(guided_backprop_ious)
    var_sa_iou = np.var(sa_ious)
    var_ig_iou = np.var(ig_ious)
    var_dl_iou = np.var(dl_ious)
    var_fa_iou = np.var(fa_ious)
    avg_lime_iou = np.mean(lime_scores) if len(lime_scores) > 0 else 0
    var_lime_iou = np.var(lime_scores) if len(lime_scores) > 0 else 0

    avg_ks_iou = np.mean(ks_scores) if len(ks_scores) > 0 else 0
    var_ks_iou = np.var(ks_scores) if len(ks_scores) > 0 else 0

    hparam_dict = {'model_architecture':args.model_name}
    metric_dict = val_stats
    metric_dict['guided_backprop_iou'] = avg_guided_backprop_iou
    metric_dict['deep_lift_shap_iou'] = avg_deep_lift_shap_iou
    metric_dict['sa_iou'] = avg_sa_iou
    metric_dict['ig_iou'] = avg_ig_iou
    metric_dict['dl_iou'] = avg_dl_iou
    metric_dict['fa_iou'] = avg_fa_iou
    metric_dict['var_guided_backprop_iou'] = var_guided_backprop_iou
    metric_dict['var_deep_lift_shap_iou'] = var_deep_lift_shap_iou
    metric_dict['var_sa_iou'] = var_sa_iou
    metric_dict['var_ig_iou'] = var_ig_iou
    metric_dict['var_dl_iou'] = var_dl_iou
    metric_dict['var_fa_iou'] = var_fa_iou
    metric_dict['lime_iou'] = avg_lime_iou
    metric_dict['ks_iou'] = avg_ks_iou
    metric_dict['var_lime_iou'] = var_lime_iou
    metric_dict['var_ks_iou'] = var_ks_iou
    writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
