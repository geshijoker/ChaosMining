import os
import sys
import csv
import time
import copy
import random
import warnings
warnings.filterwarnings('ignore')
import argparse

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

import numpy as np
import librosa
from scipy.io import wavfile

sys.path.append("/home/geshi/ChaosMining")
from chaosmining.utils import check_make_dir

# python3 create_audio_data.py --input_path_fg /data/home/geshi/data/ --input_path_bg /data/home/geshi/data/ --output_path /data/home/geshi/ChaosMining/data/audio 

parser = argparse.ArgumentParser(description='Parse arguments to create audio data with irrelevant features')
parser.add_argument('--input_path_fg', type=str, required=True,
                   help='Path to load foreground raw data')
parser.add_argument('--input_path_bg', type=str, required=True,
                   help='Path to load background raw data')
parser.add_argument('--output_path', type=str, required=True,
                   help='Path to save generated data')
parser.add_argument('--n_channels', type=int, default=10,
                   help='Number of channels of audio file')
parser.add_argument('--sample_rate', type=int, default=16000,
                   help='The sample rate of output audio file')
args = parser.parse_args()

sample_rate = args.sample_rate
n_channels = args.n_channels

# Speech Commands Dataset
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, data_root, subset: str = None):
        super().__init__(data_root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# Create training and testing split of the data. 
fg_train_set = SubsetSC(args.input_path_fg, "training")
fg_val_set = SubsetSC(args.input_path_fg, "validation")
fg_test_set = SubsetSC(args.input_path_fg, "testing")

print("foreground dataset size train {}, val {}, test {}".format(len(fg_train_set), len(fg_val_set), len(fg_test_set)))
labels = sorted(list(set(datapoint[2] for datapoint in fg_train_set)))
print("number of classes", len(labels))
for label in labels:
    check_make_dir(os.path.join(args.output_path, "RBFP", "train", label))
    check_make_dir(os.path.join(args.output_path, "RBFP", "val", label))
    check_make_dir(os.path.join(args.output_path, "RBRP", "train", label))
    check_make_dir(os.path.join(args.output_path, "RBRP", "val", label))
    check_make_dir(os.path.join(args.output_path, "SBFP", "train", label))
    check_make_dir(os.path.join(args.output_path, "SBFP", "val", label))
    check_make_dir(os.path.join(args.output_path, "SBRP", "train", label))
    check_make_dir(os.path.join(args.output_path, "SBRP", "val", label))

# Load background data
bg_train_folder = os.path.join(args.input_path_bg, "rfcx-species-audio-detection", "train")
bg_val_folder = os.path.join(args.input_path_bg, "rfcx-species-audio-detection", "test")

bg_train_files = [name for name in os.listdir(bg_train_folder) if name.endswith('.flac')]
bg_val_files = [name for name in os.listdir(bg_val_folder) if name.endswith('.flac')]

fields = ['Audio', 'Label', 'Position']
filename = "meta_data.csv"

# Random Background Fixed Position
base_path = os.path.join(args.output_path, "RBFP", "train")
with open(os.path.join(base_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for fg_id in range(len(fg_train_set)):
        # foreground
        fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_train_set[fg_id]
        fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
        pad_size = sample_rate - fg_waveform.shape[1]
        fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
        # background
        sample = np.clip(np.random.normal(0, 1, (sample_rate, n_channels))*0.25, a_min=-1, a_max=1)
        pos = 0
        sample[:,pos] = fg_waveform.squeeze()
        # save audio
        name = speaker_id+"_"+str(utterance_number)+".wav"
        wavfile.write(os.path.join(base_path, label, name), sample_rate, sample)
        csvwriter.writerow([name, str(label), str(pos)]) 

base_path = os.path.join(args.output_path, "RBFP", "val")
with open(os.path.join(base_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for fg_id in range(len(fg_val_set)):
        # foreground
        fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_val_set[fg_id]
        fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
        pad_size = sample_rate - fg_waveform.shape[1]
        fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
        # background
        sample = np.clip(np.random.normal(0, 1, (sample_rate, n_channels))*0.25, a_min=-1, a_max=1)
        pos = 0
        sample[:,pos] = fg_waveform.squeeze()
        # save audio
        name = speaker_id+"_"+str(utterance_number)+".wav"
        wavfile.write(os.path.join(base_path, label, name), sample_rate, sample)
        csvwriter.writerow([name, str(label), str(pos)]) 

# Random Background Random Position
base_path = os.path.join(args.output_path, "RBRP", "train")
with open(os.path.join(base_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for fg_id in range(len(fg_train_set)):
        # foreground
        fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_train_set[fg_id]
        fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
        pad_size = sample_rate - fg_waveform.shape[1]
        fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
        # background
        sample = np.clip(np.random.normal(0, 1, (sample_rate, n_channels))*0.25, a_min=-1, a_max=1)
        random_pos = random.randint(0, n_channels-1)
        sample[:,random_pos] = fg_waveform.squeeze()
        # save audio
        name = speaker_id+"_"+str(utterance_number)+".wav"
        wavfile.write(os.path.join(base_path, label, name), sample_rate, sample)
        csvwriter.writerow([name, str(label), str(pos)]) 

base_path = os.path.join(args.output_path, "RBRP", "val")
with open(os.path.join(base_path, filename), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for fg_id in range(len(fg_val_set)):
        # foreground
        fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_val_set[fg_id]
        fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
        pad_size = sample_rate - fg_waveform.shape[1]
        fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
        # background
        sample = np.clip(np.random.normal(0, 1, (sample_rate, n_channels))*0.25, a_min=-1, a_max=1)
        random_pos = random.randint(0, n_channels-1)
        sample[:,random_pos] = fg_waveform.squeeze()
        # save audio
        name = speaker_id+"_"+str(utterance_number)+".wav"
        wavfile.write(os.path.join(base_path, label, name), sample_rate, sample)
        csvwriter.writerow([name, str(label), str(pos)]) 

# Structural Background Fixed Position
for fg_id in range(len(fg_train_set)):
    # foreground
    fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_train_set[fg_id]
    fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
    pad_size = sample_rate - fg_waveform.shape[1]
    fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
    sample = np.random.rand(sample_rate, n_channels)
    pos = 0
    sample[:,pos] = fg_waveform.squeeze()
    # background
    for i in range(1, n_channels):
        bg_random_id = random.randint(0, len(bg_train_files)-1)
        bg_waveform, bg_sample_rate = librosa.load(os.path.join(bg_train_folder, bg_train_files[bg_random_id]), sr=sample_rate)
        bg_start = random.randint(0, len(bg_waveform)//sample_rate-1)
        max_volume = np.quantile(np.abs(bg_waveform), 0.99)
        bg_waveform = np.clip(bg_waveform[bg_start*sample_rate:(bg_start+1)*sample_rate]/max_volume, a_min=-1, a_max=1)
        sample[:,i] = bg_waveform.squeeze()
    # save audio
    base_path = os.path.join(args.output_path, "SBFP", "train")
    wavfile.write(os.path.join(base_path, label, speaker_id+"_"+str(utterance_number)+".wav"), sample_rate, sample)

for fg_id in range(len(fg_val_set)):
    # foreground
    fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_val_set[fg_id]
    fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
    pad_size = sample_rate - fg_waveform.shape[1]
    fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
    sample = np.random.rand(sample_rate, n_channels)
    pos = 0
    sample[:,pos] = fg_waveform.squeeze()
    # background
    for i in range(1, n_channels):
        bg_random_id = random.randint(0, len(bg_val_files)-1)
        bg_waveform, bg_sample_rate = librosa.load(os.path.join(bg_val_folder, bg_val_files[bg_random_id]), sr=sample_rate)
        bg_start = random.randint(0, len(bg_waveform)//sample_rate-1)
        max_volume = np.quantile(np.abs(bg_waveform), 0.99)
        bg_waveform = np.clip(bg_waveform[bg_start*sample_rate:(bg_start+1)*sample_rate]/max_volume, a_min=-1, a_max=1)
        sample[:,i] = bg_waveform.squeeze()
    # save audio
    base_path = os.path.join(args.output_path, "SBFP", "val")
    wavfile.write(os.path.join(base_path, label, speaker_id+"_"+str(utterance_number)+".wav"), sample_rate, sample)

# Structural Background Random Position
for fg_id in range(len(fg_train_set)):
    # foreground
    fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_train_set[fg_id]
    fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
    pad_size = sample_rate - fg_waveform.shape[1]
    fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
    sample = np.random.rand(sample_rate, n_channels)
    random_pos = random.randint(0, n_channels-1)
    sample[:,random_pos] = fg_waveform.squeeze()
    # background
    for i in range(n_channels):
        if i==random_pos:
            continue
        bg_random_id = random.randint(0, len(bg_train_files)-1)
        bg_waveform, bg_sample_rate = librosa.load(os.path.join(bg_train_folder, bg_train_files[bg_random_id]), sr=sample_rate)
        bg_start = random.randint(0, len(bg_waveform)//sample_rate-1)
        max_volume = np.quantile(np.abs(bg_waveform), 0.99)
        bg_waveform = np.clip(bg_waveform[bg_start*sample_rate:(bg_start+1)*sample_rate]/max_volume, a_min=-1, a_max=1)
        sample[:,i] = bg_waveform.squeeze()
    # save audio
    base_path = os.path.join(args.output_path, "SBFP", "train")
    wavfile.write(os.path.join(base_path, label, speaker_id+"_"+str(utterance_number)+".wav"), sample_rate, sample)

for fg_id in range(len(fg_val_set)):
    # foreground
    fg_waveform, fg_sample_rate, label, speaker_id, utterance_number = fg_val_set[fg_id]
    fg_waveform = torchaudio.functional.resample(fg_waveform, orig_freq=fg_sample_rate, new_freq=sample_rate)
    pad_size = sample_rate - fg_waveform.shape[1]
    fg_waveform = np.pad(fg_waveform.numpy(), ((0,0),(0,pad_size)), mode='constant', constant_values=0)
    sample = np.random.rand(sample_rate, n_channels)
    random_pos = random.randint(0, n_channels-1)
    sample[:,random_pos] = fg_waveform.squeeze()
    # background
    for i in range(n_channels):
        if i==random_pos:
            continue
        bg_random_id = random.randint(0, len(bg_val_files)-1)
        bg_waveform, bg_sample_rate = librosa.load(os.path.join(bg_val_folder, bg_val_files[bg_random_id]), sr=sample_rate)
        bg_start = random.randint(0, len(bg_waveform)//sample_rate-1)
        max_volume = np.quantile(np.abs(bg_waveform), 0.99)
        bg_waveform = np.clip(bg_waveform[bg_start*sample_rate:(bg_start+1)*sample_rate]/max_volume, a_min=-1, a_max=1)
        sample[:,i] = bg_waveform.squeeze()
    # save audio
    base_path = os.path.join(args.output_path, "SBFP", "val")
    wavfile.write(os.path.join(base_path, label, speaker_id+"_"+str(utterance_number)+".wav"), sample_rate, sample)

print("completed")