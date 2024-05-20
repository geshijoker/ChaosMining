from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn

class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor)-> Tensor:
        x = self.net(x)
        return x
    
class MLPRegressor(nn.Module):
    def __init__(self, in_channels: int, sizes: List[int], p: int=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            LinearBlock(in_channels, sizes[0]),
            *[LinearBlock(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        ])
        self.dropout = nn.Dropout(p)
        self.project = nn.Linear(sizes[-1], 1)

    def forward(self, x: Tensor)-> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.project(x)
        return x

class LinearResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.relu1 = nn.GELU()
        self.linear2 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.relu2 = nn.GELU()

    def forward(self, x: Tensor)-> Tensor:
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out += x
        out = self.norm2(out)
        out = self.relu2(out)
        return out

class MLPResRegressor(nn.Module):
    def __init__(self, in_channels: int, sizes: List[int], p: int=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(in_channels, sizes[0]),
            *[LinearResBlock(sizes[i+1]) for i in range(len(sizes)-1)]
        ])
        self.dropout = nn.Dropout(p)
        self.project = nn.Linear(sizes[-1], 1)

    def forward(self, x: Tensor)-> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.project(x)
        return x