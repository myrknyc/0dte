import numpy as np
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def compute_mu_t(mu_0, r, t):
    mu_t = mu_0 * np.exp(r * t)
    return mu_t


def compute_mean_reversion_drift(S_current, mu_t, theta_mr):
    drift_mr = -theta_mr * (S_current - mu_t) / S_current
    return drift_mr


def compute_half_life(theta_mr):
    if theta_mr <= 0:
        raise ValueError(f"theta_mr must be positive, got {theta_mr}")
    
    # Half-life in years
    half_life_years = np.log(2) / theta_mr
    
    # Convert to various units
    half_life_dict = {
        'years': half_life_years,
        'days': half_life_years * TRADING_DAYS_PER_YEAR,
        'hours': half_life_years * TRADING_DAYS_PER_YEAR * 6.5,
        'minutes': half_life_years * TRADING_DAYS_PER_YEAR * 6.5 * 60,
    }
    
    return half_life_dict


def estimate_theta_from_half_life(half_life_hours):
    if half_life_hours <= 0:
        raise ValueError(f"half_life_hours must be positive, got {half_life_hours}")
    
    # Convert half-life to years
    half_life_years = half_life_hours / (TRADING_DAYS_PER_YEAR * 6.5)
    
    # Compute theta
    theta_mr = np.log(2) / half_life_years
    
    return theta_mr


def compute_equilibrium_variance(theta_mr, sigma_mr):
    if theta_mr <= 0:
        raise ValueError(f"theta_mr must be positive, got {theta_mr}")
    
    var_equilibrium = (sigma_mr**2) / (2 * theta_mr)
    
    return var_equilibrium


def should_use_mean_reversion(T, measure='risk_neutral'):
    one_day = 1.0 / TRADING_DAYS_PER_YEAR
    is_0dte = T < one_day
    
    if measure == 'risk_neutral':
        return False, "Risk-neutral measure: mean reversion excluded for arbitrage-free pricing"
    
    elif measure == 'real_world':
        if is_0dte:
            return True, "Real-world 0DTE: mean reversion captures intraday dynamics"
        else:
            return False, "Real-world longer expiry: mean reversion effect minimal"
    
    else:
        raise ValueError(f"measure must be 'risk_neutral' or 'real_world', got {measure}")


def get_typical_theta_ranges():
    ranges = {
        'very_weak': {
            'theta': 1.0,
            'half_life_hours': compute_half_life(1.0)['hours'],
            'description': 'Minimal mean reversion, almost GBM'
        },
        'weak': {
            'theta': 3.0,
            'half_life_hours': compute_half_life(3.0)['hours'],
            'description': 'Weak intraday mean reversion'
        },
        'moderate': {
            'theta': 5.0,
            'half_life_hours': compute_half_life(5.0)['hours'],
            'description': 'Moderate mean reversion (typical 0DTE)'
        },
        'strong': {
            'theta': 10.0,
            'half_life_hours': compute_half_life(10.0)['hours'],
            'description': 'Strong mean reversion'
        },
        'very_strong': {
            'theta': 20.0,
            'half_life_hours': compute_half_life(20.0)['hours'],
            'description': 'Very strong intraday mean reversion'
        },
    }
    
    return ranges