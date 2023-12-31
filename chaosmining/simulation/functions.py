import numpy as np

def mean_std_absolute_error(preds, targets):
    abs_diff = np.absolute(preds-targets)
    return np.mean(abs_diff), np.std(abs_diff)

def mean_std_absolute_error_ratio(preds, targets):
    abs_diff = np.absolute(preds-targets)
    abs_targets = np.absolute(targets)
    diff_ratio = abs_diff/(abs_targets+1e-7)
    return np.mean(diff_ratio), np.std(diff_ratio)

def abs_argmax_topk(arrays, k):
    inds = np.argpartition(np.abs(arrays), -k, axis=-1)[..., -k:]
    return np.flip(inds, axis=-1)

def top_features_score(topk_inds, num_features):
    return np.sum(topk_inds<num_features)/np.prod(topk_inds.shape)

def uniformity_score(a, b):
    pair_sum = np.abs(a)+np.abs(b)
    diff_ratio = np.abs(a-b)/(pair_sum+1e-7)
    inv_diff = 1-diff_ratio
    return np.mean(inv_diff)

def normalize_attr(attr, ord=1):
    return attr / np.linalg.norm(attr, ord=ord, axis=-1)[...,np.newaxis]