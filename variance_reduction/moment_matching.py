import numpy as np


def verify_moment_matching(Z):
    stats = {
        'mean': np.mean(Z),
        'std': np.std(Z, ddof=0),
        'skewness': compute_skewness(Z),
        'kurtosis': compute_kurtosis(Z),
    }
    
    return stats


def compute_skewness(Z):
    mean = np.mean(Z)
    std = np.std(Z, ddof=0)
    
    if std < 1e-10:
        return 0.0
    
    skewness = np.mean(((Z - mean) / std)**3)
    
    return skewness


def compute_kurtosis(Z):
    mean = np.mean(Z)
    std = np.std(Z, ddof=0)
    
    if std < 1e-10:
        return 0.0
    
    kurtosis = np.mean(((Z - mean) / std)**4) - 3.0
    
    return kurtosis


def apply_moment_matching_to_array(data):
    mean = np.mean(data)
    std = np.std(data, ddof=0)
    
    if std < 1e-10:
        return data
    
    data_adjusted = (data - mean) / std
    
    return data_adjusted


def compare_variance_with_without_mm(generate_function, n_samples=10000):
    samples_without_mm = generate_function(n_samples)
    
    samples_with_mm = apply_moment_matching_to_array(samples_without_mm)
    
    comparison = {
        'var_without_mm': np.var(samples_without_mm),
        'var_with_mm': np.var(samples_with_mm),
        'mean_without_mm': np.mean(samples_without_mm),
        'mean_with_mm': np.mean(samples_with_mm),
        'std_without_mm': np.std(samples_without_mm, ddof=0),
        'std_with_mm': np.std(samples_with_mm, ddof=0),
    }
    
    return comparison