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

def get_box(attr_sample, size):
    attr_sample = attr_sample.mean(axis=-1)
    threshold = -np.partition(-attr_sample.flatten(), size)[size]
    pos = np.argwhere(attr_sample > threshold)
    min_x, max_x, min_y, max_y = pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,0].max()
    return min_x, max_x, min_y, max_y

def show_rects(pred, gtrue, attr_sample):
    min_x, min_y, max_x, max_y = pred
    min_xt, min_yt, max_xt, max_yt = gtrue
    fig, ax = plt.subplots()

    norm_attr_sample = np.floor((attr_sample-attr_sample.min())/attr_sample.max()*255).astype(int).mean(axis=-1)
    # Display the image
    ax.imshow(norm_attr_sample, cmap='Blues')

    # Create a Rectangle patch
    pred_rec = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, linewidth=1, edgecolor='red', facecolor='none')
    true_rec = patches.Rectangle((min_xt, min_yt), max_xt-min_xt, max_yt-min_yt, linewidth=1, edgecolor='yellow', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(pred_rec)
    ax.add_patch(true_rec)

    plt.show()
    
def calculate_iou(rectangle1, rectangle2):
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
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_end = min(x1 + width1, x2 + width2)
    y_intersection_end = min(y1 + height1, y2 + height2)

    # Calculating area of the intersection rectangle
    intersection_width = max(0, x_intersection_end - x_intersection)
    intersection_height = max(0, y_intersection_end - y_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculating area of the union
    area1 = width1 * height1
    area2 = width2 * height2
    union_area = area1 + area2 - intersection_area

    # Calculating IoU score
    iou_score = intersection_area / union_area if union_area != 0 else 0  # Avoid division by zero

    return iou_score