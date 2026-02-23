import numpy as np
from config import TRADING_DAYS_PER_YEAR, MINUTES_PER_TRADING_DAY


def adaptive_exponential_grid(T, n_steps, alpha):
    # Normalized indices: u ∈ [0, 1]
    u = np.linspace(0, 1, n_steps + 1)
    
    # Exponential transformation
    # times = T · [exp(α·u) - 1] / [exp(α) - 1]
    exp_alpha = np.exp(alpha)
    times = T * (np.exp(alpha * u) - 1) / (exp_alpha - 1)
    
    # Numerical stability: ensure exact boundaries
    times[0] = 0.0
    times[-1] = T
    
    return times


def uniform_grid(T, n_steps):
    return np.linspace(0, T, n_steps + 1)


def generate_time_grid(T, n_steps, density_factor=3.0, use_adaptive=True):
    # Determine if 0DTE
    one_trading_day = 1.0 / TRADING_DAYS_PER_YEAR
    is_0dte = T < one_trading_day
    
    # Generate time points
    if use_adaptive and is_0dte:
        times = adaptive_exponential_grid(T, n_steps, density_factor)
    else:
        times = uniform_grid(T, n_steps)
    
    # Compute time steps
    dt = np.diff(times)
    
    # Hard-enforce dt.sum() == T (prevent floating-point drift from breaking CV)
    dt_sum = dt.sum()
    if dt_sum > 0 and abs(dt_sum - T) > 1e-15:
        dt *= (T / dt_sum)
    
    return times, dt


def get_grid_statistics(times, dt):
    # Convert to real-world units
    seconds_per_year = TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY * 60
    
    min_dt = np.min(dt)
    max_dt = np.max(dt)
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    
    stats = {
        'min_dt': min_dt,
        'max_dt': max_dt,
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'ratio': max_dt / min_dt,
        'min_dt_seconds': min_dt * seconds_per_year,
        'max_dt_minutes': max_dt * seconds_per_year / 60,
    }
    
    return stats


def print_grid_info(T, times, dt):
    stats = get_grid_statistics(times, dt)
    
    # Determine grid type
    one_trading_day = 1.0 / TRADING_DAYS_PER_YEAR
    is_0dte = T < one_trading_day
    grid_type = "Adaptive (0DTE)" if is_0dte else "Uniform"
    
    # Convert T to hours
    T_hours = T * TRADING_DAYS_PER_YEAR * 6.5
    
    print("\n" + "="*60)
    print("TIME GRID INFORMATION")
    print("="*60)
    print(f"Time to expiry: {T_hours:.2f} hours ({T:.5f} years)")
    print(f"Number of steps: {len(dt)}")
    print(f"Grid type: {grid_type}")
    
    print(f"\nTime Step Statistics:")
    print(f"  Minimum: {stats['min_dt_seconds']:.1f} seconds")
    print(f"  Maximum: {stats['max_dt_minutes']:.2f} minutes")
    print(f"  Average: {stats['mean_dt']*TRADING_DAYS_PER_YEAR*6.5*60:.2f} minutes")
    print(f"  Clustering ratio: {stats['ratio']:.1f}×")
    
    # Distribution analysis
    dt_minutes = dt * TRADING_DAYS_PER_YEAR * 6.5 * 60
    first_10_avg = np.mean(dt_minutes[:10])
    last_10_avg = np.mean(dt_minutes[-10:])
    
    print(f"\nDistribution:")
    print(f"  First 10 steps: avg {first_10_avg:.2f} minutes")
    print(f"  Last 10 steps: avg {last_10_avg:.1f} seconds")
    print("="*60)

