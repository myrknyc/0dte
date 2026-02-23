import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from config import N_PATHS_DEFAULT, N_STEPS_DEFAULT


def generate_sobol_normals(n_paths, n_steps, seed=None):
    """Generate standard normal samples using Sobol sequence."""
    MAX_SOBOL_DIM = 21201
    
    if n_steps > MAX_SOBOL_DIM:
        print(f"Warning: n_steps={n_steps} exceeds Sobol limit ({MAX_SOBOL_DIM})")
        print("Falling back to pseudo-random normal generation")
        return np.random.standard_normal((n_paths, n_steps))
    
    sampler = Sobol(d=n_steps, scramble=True, seed=seed)
    uniform_samples = sampler.random(n=n_paths)
    Z = norm.ppf(uniform_samples)
    Z = np.clip(Z, -10, 10)  # Clip extreme values
    
    return Z


def apply_moment_matching(Z):
    """Adjust samples to have exact mean=0 and variance=1."""
    mean = np.mean(Z)
    std = np.std(Z, ddof=0)
    
    if std < 1e-10:
        print("Warning: Sample std very small, skipping moment matching")
        return Z
    
    Z_adjusted = (Z - mean) / std
    return Z_adjusted


def generate_antithetic_pairs(Z):
    """Create antithetic variates for variance reduction."""
    n_paths, n_steps = Z.shape
    half = n_paths // 2
    
    Z_half = Z[:half, :]
    Z_anti_half = -Z_half
    Z_antithetic = np.vstack([Z_half, Z_anti_half])
    
    if n_paths % 2 == 1:
        Z_extra = np.random.standard_normal((1, n_steps))
        Z_antithetic = np.vstack([Z_antithetic, Z_extra])
    
    return Z_antithetic


def generate_correlated_normals(n_paths=N_PATHS_DEFAULT, 
                                n_steps=N_STEPS_DEFAULT,
                                rho=0.0,
                                use_sobol=True,
                                use_moment_matching=True,
                                use_antithetic=True,
                                seed=None):
    if not -1 < rho < 1:
        raise ValueError(f"Correlation rho must be in (-1, 1), got {rho}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate independent normals
    if use_sobol:
        Z_combined = generate_sobol_normals(n_paths, 2 * n_steps, seed=seed)
        Z1 = Z_combined[:, :n_steps]
        Z_indep = Z_combined[:, n_steps:]
    else:
        Z1 = np.random.standard_normal((n_paths, n_steps))
        Z_indep = np.random.standard_normal((n_paths, n_steps))
    
    # Apply moment matching
    if use_moment_matching:
        Z1 = apply_moment_matching(Z1)
        Z_indep = apply_moment_matching(Z_indep)
    
    # Create correlation: Z2 = ρ·Z1 + √(1-ρ²)·Z_indep
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep
    
    # Apply antithetic variates
    if use_antithetic:
        Z1 = generate_antithetic_pairs(Z1)
        Z2 = generate_antithetic_pairs(Z2)
    
    return Z1, Z2