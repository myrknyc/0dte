import numpy as np
from scipy.stats import norm

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def compute_pathwise_delta(S_paths, S0, K, T, r, option_type='call'):
    # Extract terminal prices
    S_T = S_paths[:, -1]
    n_paths = len(S_T)
    
    # ITM indicator
    if option_type == 'call':
        itm = (S_T > K)
        sign = 1.0
    elif option_type == 'put':
        itm = (S_T < K)
        sign = -1.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    # Pathwise delta for each path
    # Δ = e^(-rT) · 1_{ITM} · S(T)/S₀
    discount_factor = np.exp(-r * T)
    delta_paths = sign * discount_factor * itm * (S_T / S0)
    
    # Average across paths
    delta = np.mean(delta_paths)
    delta_se = np.std(delta_paths, ddof=1) / np.sqrt(n_paths)
    
    return delta, delta_se


def compute_pathwise_gamma(S_paths, S0, K, T, r, bandwidth=None):

    # Extract terminal prices
    S_T = S_paths[:, -1]
    n_paths = len(S_T)
    
    # Automatic bandwidth selection (Silverman's rule)
    if bandwidth is None:
        sigma = np.std(S_T)
        bandwidth = 1.06 * sigma * (n_paths ** (-1/5))
    
    # Kernel approximation of Dirac delta
    # δ(S - K) ≈ (1/h) · φ((S - K)/h)
    z = (S_T - K) / bandwidth
    kernel = (1.0 / bandwidth) * norm.pdf(z)
    
    # Pathwise gamma for each path
    # Γ = e^(-rT) · kernel · (S(T)/S₀)²
    discount_factor = np.exp(-r * T)
    gamma_paths = discount_factor * kernel * (S_T / S0)**2
    
    # Average across paths
    gamma = np.mean(gamma_paths)
    gamma_se = np.std(gamma_paths, ddof=1) / np.sqrt(n_paths)
    
    return gamma, gamma_se


def compute_finite_diff_vega(pricing_function, v0, params, epsilon=0.01, 
                             convert_to_vol=True):
    # Price with base variance
    params_base = params.copy()
    params_base['v0'] = v0
    V_base = pricing_function(params_base)
    
    # Price with bumped variance
    params_bump = params.copy()
    v0_bumped = v0 * (1 + epsilon)
    params_bump['v0'] = v0_bumped
    V_bumped = pricing_function(params_bump)
    
    # Finite difference: ∂V/∂v
    dV_dv = (V_bumped - V_base) / (v0_bumped - v0)
    
    if convert_to_vol:
        # Convert to ∂V/∂σ
        # σ = √v → ∂v/∂σ = 2σ
        sigma = np.sqrt(v0)
        dV_dsigma = dV_dv * 2 * sigma
        
        # Scale to per 1% volatility change
        vega = dV_dsigma * (sigma / 100)
    else:
        vega = dV_dv
    
    return vega


def compute_finite_diff_theta(pricing_function, T, params, dt=None):
    if dt is None:
        dt = 1.0 / TRADING_DAYS_PER_YEAR  # One trading day
    
    # Price at current T
    V_current = pricing_function(T, params)
    
    # Price at T - dt (one day earlier expiry)
    if T > dt:
        T_earlier = T - dt
        V_earlier = pricing_function(T_earlier, params)
        theta = V_earlier - V_current  # Note: earlier - current (negative for long)
    else:
        # At or very close to expiry
        # Theta ≈ -V (option expires worthless or to intrinsic)
        theta = -V_current
    
    return theta


def compute_finite_diff_rho(pricing_function, r, params, epsilon=0.0001):
    # Price with base rate
    params_base = params.copy()
    params_base['r'] = r
    V_base = pricing_function(params_base)
    
    # Price with bumped rate
    params_bump = params.copy()
    r_bumped = r + epsilon
    params_bump['r'] = r_bumped
    V_bumped = pricing_function(params_bump)
    
    # Finite difference
    rho = (V_bumped - V_base) / epsilon
    
    return rho


def compute_all_greeks(S_paths, S0, K, T, r, v0, option_type='call',
                      pricing_function=None, params=None,
                      gamma_bandwidth=None):
    greeks = {}
    
    # Pathwise Greeks (always computed)
    delta, delta_se = compute_pathwise_delta(S_paths, S0, K, T, r, option_type)
    greeks['delta'] = delta
    greeks['delta_se'] = delta_se
    
    gamma, gamma_se = compute_pathwise_gamma(S_paths, S0, K, T, r, gamma_bandwidth)
    greeks['gamma'] = gamma
    greeks['gamma_se'] = gamma_se
    
    # Finite difference Greeks (if pricing function provided)
    if pricing_function is not None and params is not None:
        # Vega
        vega = compute_finite_diff_vega(pricing_function, v0, params)
        greeks['vega'] = vega
        
        # Theta (needs different signature)
        def theta_price_func(T_val, params_val):
            return pricing_function(params_val)
        
        theta = compute_finite_diff_theta(theta_price_func, T, params)
        greeks['theta'] = theta
        
        # Rho
        rho = compute_finite_diff_rho(pricing_function, r, params)
        greeks['rho'] = rho
    
    return greeks


def format_greeks_output(greeks, include_se=True):
    lines = []
    lines.append("="*50)
    lines.append("OPTION GREEKS")
    lines.append("="*50)
    
    if 'delta' in greeks:
        if include_se and 'delta_se' in greeks:
            lines.append(f"Delta: {greeks['delta']:.4f} ± {greeks['delta_se']:.4f}")
        else:
            lines.append(f"Delta: {greeks['delta']:.4f}")
    
    if 'gamma' in greeks:
        if include_se and 'gamma_se' in greeks:
            lines.append(f"Gamma: {greeks['gamma']:.6f} ± {greeks['gamma_se']:.6f}")
        else:
            lines.append(f"Gamma: {greeks['gamma']:.6f}")
    
    if 'vega' in greeks:
        lines.append(f"Vega:  ${greeks['vega']:.4f} (per 1% vol)")
    
    if 'theta' in greeks:
        lines.append(f"Theta: ${greeks['theta']:.4f} (per day)")
    
    if 'rho' in greeks:
        lines.append(f"Rho:   ${greeks['rho']:.4f} (per 1bp)")
    
    lines.append("="*50)
    
    return "\n".join(lines)
