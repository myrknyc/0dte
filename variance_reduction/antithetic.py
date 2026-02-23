import numpy as np


def verify_antithetic_correlation(Z):
    n_paths = Z.shape[0]
    half = n_paths // 2
    
    Z_first = Z[:half, :]
    Z_second = Z[half:2*half, :]
    
    n_steps = Z_first.shape[1]
    correlations = np.zeros(n_steps)
    
    for i in range(n_steps):
        correlations[i] = np.corrcoef(Z_first[:, i], Z_second[:, i])[0, 1]
    
    avg_correlation = np.mean(correlations)
    
    return correlations, avg_correlation


def compute_antithetic_variance_reduction(payoffs_all):
    n_paths = len(payoffs_all)
    half = n_paths // 2
    
    payoffs_first = payoffs_all[:half]
    payoffs_second = payoffs_all[half:2*half]
    
    var_individual = (np.var(payoffs_first) + np.var(payoffs_second)) / 2
    
    payoffs_averaged = (payoffs_first + payoffs_second) / 2
    var_averaged = np.var(payoffs_averaged)
    
    if var_averaged > 1e-12:
        vr_factor = var_individual / var_averaged
    else:
        vr_factor = 1.0
    
    return vr_factor


def split_antithetic_pairs(Z):
    n_paths = Z.shape[0]
    half = n_paths // 2
    
    Z_original = Z[:half, :]
    Z_antithetic = Z[half:2*half, :]
    
    return Z_original, Z_antithetic