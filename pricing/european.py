import numpy as np
from scipy.stats import norm
import time

from config import RISK_FREE_RATE
from pricing.black_scholes import black_scholes


def compute_payoff(S_T, K, option_type='call'):
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S_T, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    return payoff


def simulate_black_scholes_paths(S0, sigma, T, dt_array, Z, r=RISK_FREE_RATE):
    n_paths, n_steps = Z.shape
    S_paths_BS = np.zeros((n_paths, n_steps + 1))
    S_paths_BS[:, 0] = S0
    
    # Log-space BS simulation (matches combined model pattern)
    for i in range(n_steps):
        dt = dt_array[i]
        
        # Drift (log-space)
        drift = (r - 0.5 * sigma**2) * dt
        
        # Diffusion (log-space)
        diffusion = sigma * np.sqrt(dt) * Z[:, i]
        
        # Update
        log_return = drift + diffusion
        S_paths_BS[:, i + 1] = S_paths_BS[:, i] * np.exp(log_return)
    
    return S_paths_BS


def apply_control_variate(payoffs_complex, payoffs_BS, BS_analytical_price):
    # Compute optimal beta (consistent ddof=1 for both cov and var)
    covariance = np.cov(payoffs_complex, payoffs_BS, ddof=1)[0, 1]
    variance_BS = np.var(payoffs_BS, ddof=1)
    
    # Guard #2: stronger threshold + clamp to prevent insane beta
    if variance_BS > 1e-8:
        beta = covariance / variance_BS
        beta = np.clip(beta, -10.0, 10.0)
    else:
        beta = 0.0
    
    # Apply control variate
    payoffs_adjusted = payoffs_complex - beta * (payoffs_BS - BS_analytical_price)
    
    # Guard #7: if CV produces negative mean price, it's over-correcting
    cv_mean = np.mean(payoffs_adjusted)
    if cv_mean < -1e-4:
        print(f"⚠ CV produced negative price ({cv_mean:.4f}). Falling back to raw MC.")
        payoffs_adjusted = payoffs_complex
        beta = 0.0
    
    # Compute variance reduction factor
    var_before = np.var(payoffs_complex, ddof=1)
    var_after = np.var(payoffs_adjusted, ddof=1)
    
    if var_after > 1e-12:
        variance_reduction_factor = var_before / var_after
    else:
        variance_reduction_factor = 1.0
    
    return payoffs_adjusted, beta, variance_reduction_factor


def price_european_option(S_paths, K, T, r, option_type='call', 
                         use_control_variate=False, sigma_BS=None,
                         dt_array=None, Z1=None):
    start_time = time.time()
    
    # Extract terminal prices
    S_T = S_paths[:, -1]
    n_paths = len(S_T)
    
    # Compute payoffs
    payoffs = compute_payoff(S_T, K, option_type)
    
    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    
    # Apply control variate if requested
    if use_control_variate:
        if sigma_BS is None or dt_array is None or Z1 is None:
            raise ValueError("sigma_BS, dt_array, and Z1 required for control variate")
        
        # Guard #1: dt_array must sum to T
        dt_sum = float(np.sum(dt_array))
        if abs(dt_sum - T) > 1e-10:
            raise ValueError(
                f"dt_array sums to {dt_sum:.10f}, but T={T:.10f}. "
                f"CV invalid — dt_array/T mismatch."
            )
        
        # Guard: shape consistency
        if Z1.shape[1] != len(dt_array):
            raise ValueError(
                f"Z1 has {Z1.shape[1]} steps but dt_array has {len(dt_array)}. Mismatch."
            )
        
        # Simulate Black-Scholes on same random numbers
        S0 = S_paths[0, 0]
        S_paths_BS = simulate_black_scholes_paths(S0, sigma_BS, T, dt_array, Z1, r)
        S_T_BS = S_paths_BS[:, -1]
        
        # BS payoffs
        payoffs_BS = compute_payoff(S_T_BS, K, option_type)
        discounted_payoffs_BS = discount_factor * payoffs_BS
        
        # Analytical BS price
        BS_analytical = black_scholes(S0, K, T, r, sigma_BS, option_type)
        
        # Guard #6: sanity check — MC BS mean should ≈ analytical BS
        mc_bs_mean = np.mean(discounted_payoffs_BS)
        bs_tol = max(1e-3, 0.05 * abs(BS_analytical)) if BS_analytical > 0 else 1e-3
        if abs(mc_bs_mean - BS_analytical) > bs_tol:
            print(f"⚠ CV mismatch: MC_BS={mc_bs_mean:.4f}, BS_analytical={BS_analytical:.4f}. "
                  f"Skipping CV — dt_array/T/sigma_BS/r mismatch likely.")
            payoffs_for_pricing = discounted_payoffs
            beta = None
            vr_factor = 1.0
        else:
            # Apply control variate
            payoffs_for_pricing, beta, vr_factor = apply_control_variate(
                discounted_payoffs,
                discounted_payoffs_BS,
                BS_analytical
            )
    else:
        payoffs_for_pricing = discounted_payoffs
        beta = None
        vr_factor = 1.0
    
    # Compute price and statistics
    price = np.mean(payoffs_for_pricing)
    
    # Guard #5: protect against n_paths < 2 or degenerate distributions
    if n_paths < 2:
        std_error = float('nan')
    else:
        std_error = np.std(payoffs_for_pricing, ddof=1) / np.sqrt(n_paths)
    
    if not np.isfinite(std_error):
        std_error = float('nan')
    
    # CVaR₉₅ — average loss in worst 5% of outcomes (tail-risk measure)
    sorted_payoffs = np.sort(payoffs_for_pricing)
    n_tail = max(1, int(0.05 * n_paths))
    cvar_95 = float(np.mean(sorted_payoffs[:n_tail]))
    
    # 95% confidence interval
    ci_lower = price - 1.96 * std_error if np.isfinite(std_error) else float('nan')
    ci_upper = price + 1.96 * std_error if np.isfinite(std_error) else float('nan')
    
    # Payoff distribution stats (model-pure, no execution price)
    # Used by trader layer to compute PnL-space G/L for EU scoring
    pos_mask = payoffs_for_pricing > 0
    neg_mask = ~pos_mask
    payoff_mean_pos = float(np.mean(payoffs_for_pricing[pos_mask])) if np.any(pos_mask) else 0.0
    payoff_mean_zero = float(np.mean(np.abs(payoffs_for_pricing[neg_mask]))) if np.any(neg_mask) else 0.0
    payoff_frac_pos = float(np.sum(pos_mask) / n_paths)

    elapsed_time = time.time() - start_time
    
    # Build result dictionary
    result = {
        'price': price,
        'std_error': std_error,
        'cvar_95': cvar_95,
        'payoff_mean_pos': payoff_mean_pos,
        'payoff_mean_zero': payoff_mean_zero,
        'payoff_frac_pos': payoff_frac_pos,
        'confidence_interval': (ci_lower, ci_upper),
        'n_paths': n_paths,
        'elapsed_time': elapsed_time,
    }
    
    if use_control_variate:
        result['variance_reduction_factor'] = vr_factor
        result['beta'] = beta
    
    return result


def price_option_suite(S_paths, strikes, T, r, option_type='call',
                      use_control_variate=False, sigma_BS=None,
                      dt_array=None, Z1=None):
    results = {}
    
    for K in strikes:
        result = price_european_option(
            S_paths=S_paths,
            K=K,
            T=T,
            r=r,
            option_type=option_type,
            use_control_variate=use_control_variate,
            sigma_BS=sigma_BS,
            dt_array=dt_array,
            Z1=Z1
        )
        results[K] = result
    
    return results


def compare_to_black_scholes(S0, K, T, r, sigma, option_type, S_paths,
                             dt_array=None, Z1=None):
    # Black-Scholes analytical
    bs_price = black_scholes(S0, K, T, r, sigma, option_type)
    
    # Combined model (no control variate)
    result_no_cv = price_european_option(
        S_paths=S_paths,
        K=K,
        T=T,
        r=r,
        option_type=option_type,
        use_control_variate=False
    )
    model_price = result_no_cv['price']
    
    # Combined model (with control variate if possible)
    if dt_array is not None and Z1 is not None:
        result_cv = price_european_option(
            S_paths=S_paths,
            K=K,
            T=T,
            r=r,
            option_type=option_type,
            use_control_variate=True,
            sigma_BS=sigma,
            dt_array=dt_array,
            Z1=Z1
        )
        model_price_cv = result_cv['price']
        vr_factor = result_cv['variance_reduction_factor']
    else:
        model_price_cv = None
        vr_factor = None
    
    # Compute differences
    difference = model_price - bs_price
    percent_diff = (difference / bs_price * 100) if bs_price != 0 else 0.0
    
    comparison = {
        'bs_price': bs_price,
        'model_price': model_price,
        'model_price_cv': model_price_cv,
        'difference': difference,
        'percent_difference': percent_diff,
        'variance_reduction_factor': vr_factor,
    }
    
    return comparison