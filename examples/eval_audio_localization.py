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
from thop import profile, clever_format
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chaosmining.audio.functions import HFChaosMiningAudioDataset
from chaosmining.utils import check_make_dir
from chaosmining.audio.models import *
from chaosmining.audio import parse_argument, test
from chaosmining.audio.functions import *

from captum.attr import IntegratedGradients, Saliency, DeepLift, FeatureAblation, visualization, Lime, KernelShap, LayerGradCam, Occlusion, LRP, Deconvolution, ShapleyValueSampling, GradientShap, GuidedBackprop, NoiseTunnel,FeaturePermutation,DeepLiftShap
"""
python eval_audio_localization.py -d ./data/audio/RBFP/ -e ./results/audio/RBFP/ -n arc_RNN_seed_42 --model_name RNN --n_channels 10 --length 16000 --gpu 0 --batch_size 32 --deterministic --debug
"""
os.environ["HF_TOKEN"] =""
os.environ["HF_ENDPOINT"] = ""
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
    torch.cuda.manual_seed_all(seed)

experiment = args.experiment
run_name = args.name + f'_seed_{seed}'
log_path = os.path.join(experiment, run_name)
os.makedirs(log_path, exist_ok=True)

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

subset = args.hf_dataset
if not subset:
    raise ValueError("--hf_dataset must be specified for HFChaosMiningAudioDataset")
train_set = HFChaosMiningAudioDataset(subset_name=subset, split="train", target_length=length)
val_set = HFChaosMiningAudioDataset(subset_name=subset, split="val", target_length=length)
num_classes = len(train_set.classes)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

if args.model_name == "RNN":
    model = AudioRNN(n_channels, num_classes, hidden_dim=60, n_layers=3)
elif args.model_name == "LSTM":
    model = AudioLSTM(n_channels, num_classes, hidden_dim=60, n_layers=3)
elif args.model_name == "TCN":
    model = AudioTCN(n_channels, num_classes, n_channel=60)
elif args.model_name == "TRAN":
    model = AudioTrans(n_channels, num_classes, hidden_dim=60, n_layers=3)
elif args.model_name == "Wav2Vec2Model":
    model = AudioWav2Vec2(n_channels, num_classes, freeze_feature_extractor=args.freeze_feature_extractor)
elif args.model_name == "RNNT":
    model = AudioRNNT(n_channels, num_classes)
elif args.model_name == "Conformer":
    model = AudioConformer(n_channels, num_classes)
else:
    sys.exit("The model {} is not supported".format(args.model_name))


if args.model_name == "TCN":
    target_layer = model.conv4
elif args.model_name == "Conformer":
    target_layer = None
    for m in reversed(list(model.conformer.modules())):
        if isinstance(m, nn.Conv1d) and m.groups > 1: 
            target_layer = m
            break

    if target_layer is None:
        print("Warning: No suitable target layer found for Conformer model. Guided Grad-CAM will not be available.")
else:
    target_layer = None

# sanity check
sample_shape = (batch_size, n_channels, length)
sample = torch.rand(*sample_shape)
model.eval()
out = model(sample)
print('sample output', out.shape)
summary(model, input_size=sample_shape)
model.to(device)

sample_input = torch.randn(1, n_channels, length).to(device)
model_copy = copy.deepcopy(model)
macs, params = profile(model_copy, inputs=(sample_input,), verbose=False)

gflops = (macs * 2) / 1e9
print(f"Model Params: {params/1e6:.2f} M")
print(f"Model GFLOPs: {gflops:.2f} GFLOPs (per sample)")

# load pretrained model
param_files = [f for f in os.listdir(log_path) if f.endswith('.pt')]
print('model', os.path.join(log_path, param_files[0]))
model.load_state_dict(torch.load(os.path.join(log_path, param_files[0]), map_location=device)['model_state_dict'], strict=False)
model.eval()

deep_lift_shap = DeepLiftShap(model)
guided_backprop = GuidedBackprop(model)
ks = KernelShap(model)
ks_scores = []
sa = Saliency(model)
sa_scores = []
ig = IntegratedGradients(model)
ig_scores = []
dl = DeepLift(model)
dl_scores = []
fa = FeatureAblation(model)
fa_scores = []
lime = Lime(model)
lime_scores = []


writer = SummaryWriter(log_path)
writer.add_scalar('model/params_M', params/1e6)
writer.add_scalar('model/GFLOPs', gflops)
sa_scores, ig_scores, dl_scores, fa_scores = [], [], [], []
ks_scores = []
lime_scores = []
deep_lift_shap_scores = []
guided_backprop_scores = []

with torch.no_grad():
    val_stats = test(model, val_loader, num_classes, device, (1, 5), args.debug) 
    print(val_stats)
    count = 0

    model.train()
    piter = tqdm(val_loader, desc='Test', unit='batch', disable=not args.debug)
    for inputs, targets, pos, _ in piter:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # pos = pos.tolist()

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
        dl_ra = dl_selected/np.linalg.norm(dl_ma, 1, axis=-1)
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

        deep_lift_shap_attr = deep_lift_shap.attribute(inputs, torch.zeros_like(inputs).to(device), target=preds.squeeze())
        deep_lift_shap_ma = deep_lift_shap_attr.abs().mean(-1).detach().cpu().numpy()
        deep_lift_shap_selected = deep_lift_shap_ma[np.arange(inputs.size(0)), pos]
        deep_lift_shap_ra = deep_lift_shap_selected/np.linalg.norm(deep_lift_shap_ma, 1, axis=-1)
        deep_lift_shap_score = np.mean(deep_lift_shap_ra)
        deep_lift_shap_scores.append(deep_lift_shap_score)
         

        guided_backprop_attr = guided_backprop.attribute(inputs, target=preds.squeeze())
        guided_backprop_ma = guided_backprop_attr.abs().mean(-1).detach().cpu().numpy()
        guided_backprop_selected = guided_backprop_ma[np.arange(inputs.size(0)), pos]
        guided_backprop_ra = guided_backprop_selected/np.linalg.norm(guided_backprop_ma, 1, axis=-1)
        guided_backprop_score = np.mean(guided_backprop_ra)
        guided_backprop_scores.append(guided_backprop_score)

        lime_attr = torch.zeros_like(inputs)
        lime_ma = lime_attr.abs().mean(-1).detach().cpu().numpy()
        lime_selected = lime_ma[np.arange(inputs.size(0)), pos]
        lime_ra = lime_selected / np.linalg.norm(lime_ma, 1, axis=-1)
        lime_score = np.mean(lime_ra)
        lime_scores.append(lime_score)
        lime_attr = torch.cat([
            lime.attribute(
                inputs[i:i+1],
                target=preds[i].item(),
                n_samples=30
            )
            for i in range(inputs.size(0))
        ], dim=0)

        lime_ma = lime_attr.abs().mean(-1).detach().cpu().numpy()
        lime_selected = lime_ma[np.arange(inputs.size(0)), pos]
        lime_ra = lime_selected / np.linalg.norm(lime_ma, 1, axis=-1)
        lime_score = np.mean(lime_ra)
        lime_scores.append(lime_score)

        ks_attr = torch.cat([
            ks.attribute(
                inputs[i:i+1],
                target=preds[i].item(),
                n_samples=30
            )
            for i in range(inputs.size(0))
        ], dim=0)

        ks_ma = ks_attr.abs().mean(-1).detach().cpu().numpy()
        ks_selected = ks_ma[np.arange(inputs.size(0)), pos]
        ks_ra = ks_selected / np.linalg.norm(ks_ma, 1, axis=-1)
        ks_score = np.mean(ks_ra)
        ks_scores.append(ks_score)
        
    avg_sa_score = np.mean(sa_scores)
    var_sa_score = np.var(sa_scores)
    avg_ig_score = np.mean(ig_scores)
    var_ig_score = np.var(ig_scores)
    avg_dl_score = np.mean(dl_scores)
    var_dl_score = np.var(dl_scores)
    avg_fa_score = np.mean(fa_scores)
    var_fa_score = np.var(fa_scores)
    avg_deep_lift_shap_score = np.mean(deep_lift_shap_scores)
    var_deep_lift_shap_score = np.var(deep_lift_shap_scores)
    avg_guided_backprop_score = np.mean(guided_backprop_scores)
    var_guided_backprop_score = np.var(guided_backprop_scores)
    avg_lime_score = np.mean(lime_scores)
    var_lime_score = np.var(lime_scores)
    avg_ks_score = np.mean(ks_scores)
    var_ks_score = np.var(ks_scores)
    avg_ggc_score = np.mean(ggc_scores)
    var_ggc_score = np.var(ggc_scores)

    hparam_dict = {'model_architecture':args.model_name}
    metric_dict = val_stats
    metric_dict['sa_score_var'] = var_sa_score
    metric_dict['sa_score'] = avg_sa_score
    metric_dict['ig_score_var'] = var_ig_score
    metric_dict['ig_score'] = avg_ig_score
    metric_dict['dl_score_var'] = var_dl_score

    metric_dict['dl_score'] = avg_dl_score
    metric_dict['fa_score_var'] = var_fa_score
    metric_dict['fa_score'] = avg_fa_score
    metric_dict['deep_lift_shap_score_var'] = var_deep_lift_shap_score
    metric_dict['deep_lift_shap_score'] = avg_deep_lift_shap_score
    metric_dict['guided_backprop_score_var'] = var_guided_backprop_score
    metric_dict['guided_backprop_score'] = avg_guided_backprop_score
    metric_dict['lime_score_var'] = var_lime_score
    metric_dict['lime_score'] = avg_lime_score
    metric_dict['ks_score_var'] = var_ks_score
    metric_dict['ks_score'] = avg_ks_score
    metric_dict['ggc_score_var'] = var_ggc_score
    metric_dict['ggc_score'] = avg_ggc_score
    
    writer.add_hparams(hparam_dict, metric_dict)

writer.flush()
writer.close()
