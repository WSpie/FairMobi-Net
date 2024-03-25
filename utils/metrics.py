from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

def js_divergence(y_true, y_pred):
    # Normalize the input arrays to represent probability distributions
    y_true_normalized = y_true / y_true.sum()
    y_pred_normalized = y_pred / y_pred.sum()
    jsd_value = jensenshannon(y_true_normalized.flatten(), y_pred_normalized.flatten())
    return jsd_value

def pearson_correlation(y_true, y_pred):
    correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return correlation

def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def nrmse(y_true, y_pred):
    nrmse_value = rmse(y_true, y_pred) / (y_true.max() - y_true.min())
    return nrmse_value

def cpc(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    cpc_value = numerator / denominator
    return cpc_value

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_rate(y_true_tensor, y_pred_tensor, z_tensor, weight=False):
    z_tensor = torch.argmax(z_tensor, axis=1)
    z_tensor = z_tensor.to(torch.int)
    z_class, z_counts = z_tensor.unique(return_counts=True)
    mae_rate_val = 0
    for comb in combinations(z_class, 2):
        z0, z1 = comb[0].item(), comb[1].item()
        idx0 = (z_tensor==z0).nonzero(as_tuple=True)[0]
        idx1 = (z_tensor==z1).nonzero(as_tuple=True)[0]
        mae0 = torch.mean(torch.abs(y_true_tensor[idx0] - y_pred_tensor[idx0]))
        mae1 = torch.mean(torch.abs(y_true_tensor[idx1] - y_pred_tensor[idx1]))
        if weight:
            w = y_true_tensor.shape[0] / (idx0.shape[0] + idx1.shape[0])
        else:
            w = 1
        mae_rate_val += w * torch.abs(mae0 - mae1)
    if type(mae_rate_val) == int:
        return mae_rate_val
    else:
        return mae_rate_val.item()