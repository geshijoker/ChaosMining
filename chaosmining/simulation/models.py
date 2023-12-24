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