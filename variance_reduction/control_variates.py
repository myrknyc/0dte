import numpy as np
from pricing.black_scholes import black_scholes


def apply_control_variate(payoffs_complex, payoffs_control, control_analytical_value):
    covariance = np.cov(payoffs_complex, payoffs_control)[0, 1]
    variance_control = np.var(payoffs_control)
    
    if variance_control > 1e-12:
        beta = covariance / variance_control
    else:
        beta = 0.0
    
    mean_control = np.mean(payoffs_control)
    payoffs_adjusted = payoffs_complex - beta * (payoffs_control - control_analytical_value)
    
    var_before = np.var(payoffs_complex)
    var_after = np.var(payoffs_adjusted)
    
    if var_after > 1e-12:
        variance_reduction_factor = var_before / var_after
    else:
        variance_reduction_factor = 1.0
    
    return payoffs_adjusted, beta, variance_reduction_factor


def simulate_black_scholes_control(S0, K, T, r, sigma, Z, dt_array, option_type='call'):
    n_paths, n_steps = Z.shape
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    
    for i in range(n_steps):
        dt = dt_array[i]
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, i]
        log_return = drift + diffusion
        S_paths[:, i + 1] = S_paths[:, i] * np.exp(log_return)
    
    S_T = S_paths[:, -1]
    
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    discount_factor = np.exp(-r * T)
    control_payoffs = discount_factor * payoffs
    
    analytical_value = black_scholes(S0, K, T, r, sigma, option_type)
    
    return control_payoffs, analytical_value


def price_with_control_variate(S_paths, K, T, r, sigma_control, Z, dt_array, option_type='call'):
    S0 = S_paths[0, 0]
    S_T = S_paths[:, -1]
    
    if option_type == 'call':
        payoffs_complex = np.maximum(S_T - K, 0)
    else:
        payoffs_complex = np.maximum(K - S_T, 0)
    
    discount_factor = np.exp(-r * T)
    discounted_payoffs_complex = discount_factor * payoffs_complex
    
    control_payoffs, analytical_value = simulate_black_scholes_control(
        S0, K, T, r, sigma_control, Z, dt_array, option_type
    )
    
    payoffs_adjusted, beta, vr_factor = apply_control_variate(
        discounted_payoffs_complex,
        control_payoffs,
        analytical_value
    )
    
    price = np.mean(payoffs_adjusted)
    std_error = np.std(payoffs_adjusted, ddof=1) / np.sqrt(len(payoffs_adjusted))
    
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error
    
    result = {
        'price': price,
        'std_error': std_error,
        'confidence_interval': (ci_lower, ci_upper),
        'beta': beta,
        'variance_reduction_factor': vr_factor,
        'control_analytical': analytical_value,
    }
    
    return result