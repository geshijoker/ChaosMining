import time
import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

class ToTensor(object):
    """Convert values in landmarks to Tensors."""
    def __call__(self, landmarks):
        return [torch.tensor(landmark) for landmark in landmarks]

def replace_relu_with_inplace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and getattr(child, 'inplace', True):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_inplace_relu(child)

def get_multi_box(attr_sample, size, sign="absolute_value"):
    bs = attr_sample.shape[0]
    if sign=="absolute_value":
        attr_sample = np.abs(attr_sample)
        attr_sample = attr_sample.mean(axis=-1)
    elif sign=="positive":
        attr_sample[attr_sample<0] = 0
        attr_sample = attr_sample.mean(axis=-1)
    elif sign=="negative":
        attr_sample[attr_sample>0] = 0
        attr_sample = np.abs(attr_sample)
        attr_sample = attr_sample.mean(axis=-1)
    else:
        attr_sample = attr_sample.mean(axis=-1)
    threshold = -np.partition(-attr_sample.reshape(bs, -1), size, axis=-1)[:, size]
    for dim in range(1, len(attr_sample.shape)):
        threshold = threshold[:,np.newaxis]
    min_x, max_x, min_y, max_y = [], [], [], []
    for i in range(bs):
        pos = np.argwhere(attr_sample[i]-threshold[i]>=0)
        min_x.append(pos[:,0].min())
        max_x.append(pos[:,0].max())
        min_y.append(pos[:,1].min())
        max_y.append(pos[:,1].max())
    # print('threshold', threshold.shape)
    # pos = np.argwhere(attr_sample-threshold>0)
    # print('pos shape', pos.shape)
    # pos = np.reshape(pos, (bs, size, -1))
    # min_x, max_x, min_y, max_y = pos[...,-2].min(axis=-1), pos[...,-2].max(axis=-1), pos[...,-1].min(axis=-1), pos[...,-1].max(axis=-1)
    return np.array(min_x), np.array(max_x), np.array(min_y), np.array(max_y)

def show_rects(pred, gtrue, attr_sample):
    min_x, min_y, max_x, max_y = pred
    min_xt, min_yt, max_xt, max_yt = gtrue
    fig, ax = plt.subplots()

    norm_attr_sample = np.floor((attr_sample-attr_sample.min())/attr_sample.max()*255).astype(int).mean(axis=-1)
    # Display the image
    ax.imshow(norm_attr_sample, cmap='Blues')

    # Create a Rectangle patch
    pred_rec = patches.Rectangle((min_y, min_x), max_y-min_y, max_x-min_x, linewidth=1, edgecolor='red', facecolor='none')
    true_rec = patches.Rectangle((min_yt, min_xt), max_yt-min_yt, max_xt-min_xt, linewidth=1, edgecolor='yellow', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(pred_rec)
    ax.add_patch(true_rec)

    plt.show()
    
def calculate_multi_iou(rectangle1, rectangle2):
    """
    Calculate Intersection over Union (IoU) between two rectangles.

    Parameters:
    - rectangle1: Tuple (x1, y1, width1, height1) representing the first rectangle.
    - rectangle2: Tuple (x2, y2, width2, height2) representing the second rectangle.

    Returns:
    - IoU score between the two rectangles.
    """

    # Extracting coordinates and dimensions
    x1, y1, width1, height1 = rectangle1
    x2, y2, width2, height2 = rectangle2

    # Calculating coordinates of the intersection rectangle
    x_intersection = np.maximum(x1, x2)
    y_intersection = np.maximum(y1, y2)
    x_intersection_end = np.minimum(x1 + width1, x2 + width2)
    y_intersection_end = np.minimum(y1 + height1, y2 + height2)

    # Calculating area of the intersection rectangle
    intersection_width = np.maximum(0, x_intersection_end - x_intersection)
    intersection_height = np.maximum(0, y_intersection_end - y_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculating area of the union
    area1 = width1 * height1
    area2 = width2 * height2
    union_area = area1 + area2 - intersection_area

    # Calculating IoU score
    iou_score = intersection_area / (union_area+1e-13)

    return iou_score