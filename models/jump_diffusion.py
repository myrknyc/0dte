import numpy as np
from config import JUMP_PARAMS


def generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, 
                              mu_jump, sigma_jump):
    # Compute jump probability per time step
    prob_jump = 1.0 - np.exp(-lambda_jump * dt_array)  # Shape: (n_steps,)
    
    # Broadcast to (1, n_steps) for comparison with (n_paths, n_steps) matrix
    prob_jump = prob_jump[np.newaxis, :]  # Shape: (1, n_steps)
    
    # Generate uniform random matrix for ALL paths and steps at once
    U = np.random.uniform(0, 1, size=(n_paths, n_steps))
    
    # Create boolean jump mask (vectorized comparison)
    jump_mask = (U < prob_jump)  # Shape: (n_paths, n_steps)
    
    # Count total jumps across all paths and steps
    n_total_jumps = np.sum(jump_mask)
    
    # Generate ALL jump sizes at once (single RNG call)
    if n_total_jumps > 0:
        all_jump_sizes = np.random.normal(mu_jump, sigma_jump, size=n_total_jumps)
    else:
        all_jump_sizes = np.array([])
    
    # Initialize output array (default: no jumps)
    jump_log_sizes = np.zeros((n_paths, n_steps))
    
    # Assign jump sizes where jumps occur (vectorized operation - no loops!)
    jump_log_sizes[jump_mask] = all_jump_sizes
    
    return jump_log_sizes


def compute_jump_statistics(jump_log_sizes, dt_array, total_time):
    # Count jumps
    jump_mask = (jump_log_sizes != 0)
    n_jumps = np.sum(jump_mask)
    
    # Extract actual jump sizes (non-zero entries)
    actual_jumps = jump_log_sizes[jump_mask]
    
    # Compute intensity
    n_paths, n_steps = jump_log_sizes.shape
    jump_intensity = n_jumps / (n_paths * total_time) if total_time > 0 else 0
    
    stats = {
        'n_jumps': int(n_jumps),
        'jump_intensity': jump_intensity,
        'mean_jump_size': np.mean(actual_jumps) if n_jumps > 0 else 0.0,
        'std_jump_size': np.std(actual_jumps) if n_jumps > 0 else 0.0,
        'jumps_per_path': n_jumps / n_paths if n_paths > 0 else 0.0,
    }
    
    return stats


def validate_bernoulli_approximation(lambda_jump, dt_max, threshold=0.01):
    lambda_dt = lambda_jump * dt_max
    
    # Approximate probability of 2 or more jumps
    prob_multiple = (lambda_dt**2) / 2.0
    
    valid = prob_multiple < threshold
    
    return valid, prob_multiple


def compute_jump_compensator(mu_jump, sigma_jump):
    """Merton jump compensator: k = E[e^J - 1] where J ~ N(mu_jump, sigma_jump^2).
    
    Used to correct the risk-neutral drift: drift becomes r - 0.5v - λk.
    This ensures E[S_T] = S0·e^(rT) even with jumps.
    """
    return np.exp(mu_jump + 0.5 * sigma_jump**2) - 1


def get_default_jump_params():
    return JUMP_PARAMS.copy()