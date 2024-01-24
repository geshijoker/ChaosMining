from typing import List, Tuple, Dict, Set, Union
import csv
import os
import sys
import copy
import numpy as np
import scipy
from sympy import *
from scipy.io import wavfile

import pandas as pd
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch import Tensor
from torch import nn
import torch.nn.functional as F

def create_simulation_data(
    function: str,
    num_features: int = 1,
    num_noises: int = 10,
    num_data: int = 10000,
    X_var: float = 0.33,
    y_var: float = 0.01,
    n_steps: int = 20,
    ):
    """ Creating simulation data with noisy (normal distribution) features and labels.
        Args:
            pair: a pair of instance -- (number of variables, function string)
            num_features: number of relevant features
            num_noises: number of noisy/irrelevant features
            num_data: number of vectorized data to create
            X_var: the scale of the variance of features 
            y_var: the scale of noises adding to targets
            n_steps: the number of steps to integrate the gradients
        Return:
            X: the features
            y_true: the ground truth targets
            y_noise: the noise adding to targets
            derivatives: the value of derivatives
            integrations: the value of integrations
    """
    # create symbolic variables
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')
    variables = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]
    
    X = np.clip(np.random.randn(num_data, num_features+num_noises)*X_var, -1, 1)
    # X = (2*np.random.rand(num_data, num_features+num_noises)-1)*X_var
    X_features = [X[:, ind] for ind in range(num_features)]
    y_noise = np.random.randn(num_data, 1)*y_var
    
    expression = sympify(function, evaluate=False)
    exp_func = lambdify(variables[:num_features], expression, 'numpy')
    y_true = exp_func(*X_features)
    
    intercepts = []
    for ind in range(num_features):
        baseline = copy.deepcopy(X_features)
        baseline[ind] = np.zeros(num_data)
        base_true = exp_func(*baseline)
        intercepts.append(y_true-base_true)
    
    derivatives = []
    for ind in range(num_features):
        derivative = diff(expression, variables[ind])
        der_func = lambdify(variables[:num_features], derivative, 'numpy')
        der_true = der_func(*X_features)+np.zeros(num_data)
        derivatives.append(der_true)
    
#     integrations = []
#     for ind in range(num_features):
#         integration = integrate(expression, variables[ind])
#         int_func = lambdify(variables[:num_features], integration, 'numpy')
#         baseline_features = copy.deepcopy(X_features)
#         baseline_features[ind] = np.zeros_like(baseline_features[ind])
#         int_true = int_func(*X_features)-int_func(*baseline_features)+np.zeros(num_data)
#         integrations.append(int_true)

    integrations = [[] for ind in range(num_features)]
    for itr in range(n_steps):
        baseline = [np.zeros(num_data)+X_features[ind]*itr/n_steps for ind in range(num_features)]
        base_true = exp_func(*baseline)
        for ind in range(num_features):
            int_features = copy.deepcopy(baseline)
            int_features[ind] = baseline[ind]+X_features[ind]/n_steps
            int_true = exp_func(*int_features)
            integrations[ind].append(int_true-base_true)
    integrations = [np.stack(integrations[ind], axis=1) for ind in range(num_features)]
    integrations = [integrations[ind].sum(axis=-1) for ind in range(num_features)]
            
    return X, np.expand_dims(y_true,-1), y_noise, intercepts, derivatives, integrations
    
def read_formulas(file_path):
    formulas = []
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            formulas.append(lines)
    formulas = [[int(formula[0]), formula[1]] for formula in formulas]
    return formulas
    
if __name__ == '__main__':
    function = '(a-1)/(b**2+1)-c**3/(d**2+1)+e**5/(f**2+1)-g**7/(h**2+1)+cot(i+pi/2)'
    num_features = 9
    num_noises = 0
    num_data = 10000
    X_var = 0.33
    y_var = 0.01
    X, y_true, y_noise, intercepts, derivatives, integrations = create_simulation_data(function, num_features, num_noises, num_data, X_var, y_var)
    print('X', X.shape, 'y true', y_true.shape, 'y noise', y_noise.shape, 
          'intercepts', len(intercepts), intercepts[0].shape,
          'derivatives', len(derivatives), derivatives[0].shape, 
          'integrations', len(integrations), integrations[0].shape)
    
class ChaosVisionDataset(Dataset):
    """Vision dataset for chaos mining."""

    def __init__(self, root_dir, csv_file, transform=None, target_transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on a target.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def get_target_names(self):
        names = list(self.df.columns.values)
        names.pop(0)
        return names

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)  
        landmarks = self.df.iloc[idx, 1:].values

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            landmarks = self.target_transform(landmarks)
        sample = (image, landmarks)

        return sample

class ChaosAudioDataset(Dataset):
    """Audio dataset for chaos mining."""
    def __init__(self, root, split = None) -> None:
        if split is not None and split not in ["train", "val"]:
            raise ValueError("When `split` is not None, it must be one of ['training', 'validation', 'testing'].")
        
        root = os.fspath(root)
        self.folder = os.path.join(root, split)
        self.meta = None

        self.classes = os.listdir(self.folder)
        self.audio_files = []
        for i in range(len(self.classes)):
            cla = self.classes[i]
            files = os.listdir(os.path.join(self.folder, cla))
            self.audio_files.extend(list(zip(files, [i]*len(files))))

    def load_meta(csv_file):
        self.meta = pd.read_csv(csv_file, index_col=0)
            
    def __getitem__(self, idx):
        cla = self.audio_files[idx][1]
        audio_file = os.path.join(self.folder, self.classes[cla], self.audio_files[idx][0])
        sample_rate, waveform = wavfile.read(audio_file)
        waveform = waveform.transpose()
        if self.meta:
            pos = int(df.loc[self.audio_files[idx][0], 'Position'])
            sample = (Tensor(waveform), cla, pos, sample_rate)
        else:
            sample = (Tensor(waveform), cla, sample_rate)
        return sample

    def __len__(self):
        return len(self.audio_files)