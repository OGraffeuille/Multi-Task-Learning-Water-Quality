import warnings

import numpy as np

import torch
import torch.distributions as D

from src.utility.utility_misc import count_nans


# Computes MSE, ignoring nan values (in case of some missing labels)
def mse_loss(y, pred):
    mask = ~torch.isnan(y)
    y = y[mask]
    pred = pred[mask]
    return torch.square(y - pred)
    

### Metric functions
# All metric functions take y_true and y_pred as numpy arrays of shape [num_data, num_preds]
#   y_mean, if given, is a numpy array corresponding to the task means, used to compute 
#   performance based on residuals from the mean
# Note that both y_true and y_pred can be NaNs:
#   y_true values may be NaNs in case of missing labels
#   y_pred values may be NaNs in case of different number of predictions for different datapoints


# RMSE
def root_mean_squared_error(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# RMSLE
def root_mean_squared_log_error(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# MAPE
def median_absolute_percentage_error(y_true, y_pred, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return 100 * np.median(np.abs((y_true - y_pred) / (y_true + eps)))

# Log MAE
def mean_absolute_error(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    return 10**(np.mean(np.abs(y_pred - y_true)))

# Log Bias
def bias(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    return 10**(np.mean(y_pred - y_true))

# Log Slope of linear fit
def slope(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    try: 
        with warnings.catch_warnings(): # Ignore RankWarning
            warnings.filterwarnings("ignore", category=np.RankWarning)
            Slope = np.polyfit(y_true, y_pred, deg=1)[0] 
    except Exception as e: # If LinAlgError then return np.nan
        Slope = np.nan
    return Slope

# Log R2
def r2(y_true, y_pred, y_mean, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    y_mean = np.log10(np.mean(y_mean[mask]))
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - 0 - (residual_sum_of_squares / (total_sum_of_squares + eps))
    
    return R2

# Intra-group R2
def r2_intra_group(y_true, y_pred, y_mean, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(y_mean)
    if np.sum(mask) < 2:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    y_mean = np.log10(y_mean[mask])
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (residual_sum_of_squares / (total_sum_of_squares + eps))

    return R2

# Intra-group R2 [alternate formulation, not used]
def r2_group_mean(y_true, y_pred, y_mean, T, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(y_mean)
    if np.sum(mask) < 2:
        return np.nan
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    y_mean = np.log10(y_mean[mask])
    T = T[mask]
    r2_group_mean_tot = 0
    for t in np.unique(T):
        y_true_t = y_true[t == T]
        y_pred_t = y_pred[t == T]
        y_mean_t = y_mean[t == T]
        assert np.sum(np.abs(y_mean_t - np.mean(y_mean_t))) < 1e-6, "ERROR: y_mean_T should be constant when calculating r2_group_mean."
        total_sum_of_squares = np.sum((y_true_t - y_mean_t) ** 2)
        residual_sum_of_squares = np.sum((y_true_t - y_pred_t) ** 2)
        r2_group_t = 1 - (residual_sum_of_squares / (total_sum_of_squares + eps))
        r2_group_mean_tot += y_true_t.shape[0] * r2_group_t
    return r2_group_mean_tot / y_true.shape[0]

# Median Symmetric Accuracy (Morley et al., 2018)
def median_symmetric_accuracy(y_true, y_pred, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    if np.sum(mask) < 1:
        return np.nan
    return 100 * np.exp(np.median(np.abs(y_pred / (y_true + eps))) - 1)

# Symmetric Signed Percentage Bias (Morley et al., 2018)
def symmetric_signed_percentage_bias(y_true, y_pred, eps=1e-6):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = np.log10(y_true[mask])
    y_pred = np.log10(y_pred[mask])
    if np.sum(mask) < 1:
        return np.nan
    MdLQ = np.median(y_pred / (y_true + eps))
    return 100 * np.sign(MdLQ) * np.exp(np.abs(MdLQ) - 1)