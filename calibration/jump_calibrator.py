import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import yfinance as yf

from config import JUMP_THRESHOLD, TRADING_DAYS_PER_YEAR


def calibrate_jumps_live(ticker='SPY', lookback_days=60):
    print(f"Analyzing {ticker} returns for jump detection...")
    
    try:
        # Download historical data
        hist_data = yf.download(ticker, period=f"{lookback_days}d", 
                               interval='1d', progress=False)
        
        if hist_data.empty or len(hist_data) < 20:
            raise ValueError(f"Insufficient data: only {len(hist_data)} days")
        
        prices = hist_data['Close'].values
        returns = np.diff(np.log(prices))
        
        # Use existing calibration function
        params = calibrate_from_returns(
            returns,
            dt=1/252,
            threshold=3.0,
            min_jumps=3
        )
        
        # Return only the key jump parameters
        return {
            'lambda_jump': params['lambda_jump'],
            'mu_jump': params['mu_jump'],
            'sigma_jump': params['sigma_jump']
        }
        
    except Exception as e:
        print(f"Error in jump calibration: {e}")
        print("Using default jump parameters")
        return {
            'lambda_jump': 2.0,
            'mu_jump': -0.01,
            'sigma_jump': 0.02
        }





def detect_jumps_threshold(returns, threshold=JUMP_THRESHOLD, dt=1/252):
    returns = np.array(returns)
    
    # Compute threshold
    sigma = np.std(returns)
    jump_threshold = threshold * sigma
    
    # Detect jumps
    jump_mask = np.abs(returns) > jump_threshold
    
    jump_returns = returns[jump_mask]
    normal_returns = returns[~jump_mask]
    
    return jump_mask, jump_returns, normal_returns


def calibrate_from_returns(returns, dt=1/252, threshold=JUMP_THRESHOLD, 
                           min_jumps=5):
    returns = np.array(returns)
    n_periods = len(returns)
    
    print(f"Calibrating jumps from {n_periods} returns (threshold={threshold}σ)...")
    
    # Detect jumps
    jump_mask, jump_returns, normal_returns = detect_jumps_threshold(
        returns, threshold=threshold, dt=dt
    )
    
    n_jumps = len(jump_returns)
    
    print(f"  Detected {n_jumps} jumps ({n_jumps/n_periods*100:.2f}% of returns)")
    
    # Check if enough jumps
    if n_jumps < min_jumps:
        warnings.warn(
            f"Only {n_jumps} jumps detected (minimum {min_jumps} recommended). "
            f"Using default parameters."
        )
        return {
            'lambda_jump': 0.5,  # Conservative default
            'mu_jump': 0.0,
            'sigma_jump': 0.02,
            'n_jumps': n_jumps,
            'jump_probability': 0.0
        }
    
    # [1] Estimate jump intensity (λ)
    # λ = (# jumps) / (total time in years)
    total_time_years = n_periods * dt
    lambda_jump = n_jumps / total_time_years
    
    # [2] Estimate jump size distribution (μⱼ, σⱼ)
    mu_jump = np.mean(jump_returns)
    sigma_jump = np.std(jump_returns, ddof=1)
    
    # [3] Compute jump probability per period
    jump_prob_per_period = n_jumps / n_periods
    
    # Validation
    if lambda_jump > 100:
        warnings.warn(f"Very high jump intensity ({lambda_jump:.1f}/year). Check threshold.")
    
    if sigma_jump < 0.005:
        warnings.warn(f"Very low jump volatility ({sigma_jump:.4f}). May need more data.")
    
    params = {
        'lambda_jump': lambda_jump,
        'mu_jump': mu_jump,
        'sigma_jump': sigma_jump,
        'n_jumps': int(n_jumps),
        'jump_probability': jump_prob_per_period,
    }
    
    # Report results
    print(f"\nCalibration results:")
    print(f"  λ (intensity): {lambda_jump:.2f} jumps/year")
    print(f"  μⱼ (mean jump): {mu_jump:.4f} ({mu_jump*100:.2f}%)")
    print(f"  σⱼ (jump vol): {sigma_jump:.4f} ({sigma_jump*100:.2f}%)")
    print(f"  P(jump per period): {jump_prob_per_period:.4f}")
    
    # Statistics on jump direction
    positive_jumps = np.sum(jump_returns > 0)
    negative_jumps = np.sum(jump_returns < 0)
    print(f"  Jump direction: {positive_jumps} up, {negative_jumps} down")
    
    if n_jumps > 0:
        avg_up_jump = np.mean(jump_returns[jump_returns > 0]) if positive_jumps > 0 else 0
        avg_down_jump = np.mean(jump_returns[jump_returns < 0]) if negative_jumps > 0 else 0
        print(f"  Avg up jump: {avg_up_jump:.2%}")
        print(f"  Avg down jump: {avg_down_jump:.2%}")
    
    return params

def calibrate_from_returns_mle(returns, dt=1/252, initial_guess=None):
    returns = np.array(returns)
    n = len(returns)
    
    print(f"Calibrating jumps using MLE on {n} returns...")
    
    # Initialize with threshold method
    if initial_guess is None:
        threshold_params = calibrate_from_returns(returns, dt=dt, threshold=3.0)
        
        # Estimate diffusion parameters from non-jump returns
        _, _, normal_returns = detect_jumps_threshold(returns, threshold=3.0)
        mu_d = np.mean(normal_returns)
        sigma_d = np.std(normal_returns, ddof=1)
        
        initial_guess = [
            threshold_params['lambda_jump'],
            threshold_params['mu_jump'],
            threshold_params['sigma_jump'],
            mu_d,
            sigma_d
        ]
    
    # Negative log-likelihood function
    def neg_log_likelihood(x):
        lambda_jump, mu_jump, sigma_jump, mu_d, sigma_d = x
        
        # Parameter constraints
        if lambda_jump <= 0 or sigma_jump <= 0 or sigma_d <= 0:
            return 1e10
        
        if lambda_jump > 100:  # Unreasonably high
            return 1e10
        
        # Jump probability per period
        p = lambda_jump * dt
        p = np.clip(p, 0, 0.99)  # Can't exceed 99%
        
        # Mixture density for each return
        normal_pdf = norm.pdf(returns, mu_d, sigma_d)
        jump_pdf = norm.pdf(returns, mu_jump, sigma_jump)
        
        mixture_pdf = (1 - p) * normal_pdf + p * jump_pdf
        
        # Avoid log(0)
        mixture_pdf = np.maximum(mixture_pdf, 1e-300)
        
        # Negative log-likelihood
        nll = -np.sum(np.log(mixture_pdf))
        
        return nll
    
    # Bounds
    bounds = [
        (0.01, 50),      # lambda: 0.01 to 50 jumps/year
        (-0.2, 0.2),     # mu_jump: -20% to +20%
        (0.001, 0.5),    # sigma_jump: 0.1% to 50%
        (-0.1, 0.1),     # mu_d: -10% to +10%
        (0.001, 0.5),    # sigma_d: 0.1% to 50%
    ]
    
    # Optimize
    print("Running MLE optimization...")
    result = minimize(
        neg_log_likelihood,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )
    
    if not result.success:
        warnings.warn(f"MLE optimization did not converge: {result.message}")
    
    # Extract parameters
    lambda_jump, mu_jump, sigma_jump, mu_d, sigma_d = result.x
    
    params = {
        'lambda_jump': lambda_jump,
        'mu_jump': mu_jump,
        'sigma_jump': sigma_jump,
        'mu_diffusion': mu_d,
        'sigma_diffusion': sigma_d,
        'log_likelihood': -result.fun,
    }
    
    print(f"\nMLE Calibration results:")
    print(f"  λ: {lambda_jump:.2f} jumps/year")
    print(f"  μⱼ: {mu_jump:.4f}")
    print(f"  σⱼ: {sigma_jump:.4f}")
    print(f"  Diffusion μ: {mu_d:.4f}")
    print(f"  Diffusion σ: {sigma_d:.4f}")
    print(f"  Log-likelihood: {-result.fun:.2f}")
    
    return params, result

def adjust_for_risk_neutral(mu_jump, sigma_jump, lambda_jump=None):
    # Simplified risk-neutral adjustment
    mu_jump_rn = mu_jump - (sigma_jump**2) / 2.0
    
    params_rn = {
        'lambda_jump': lambda_jump if lambda_jump is not None else None,
        'mu_jump': mu_jump_rn,
        'sigma_jump': sigma_jump,  # Unchanged
    }
    
    print(f"Risk-neutral adjustment:")
    print(f"  Real-world μⱼ: {mu_jump:.4f}")
    print(f"  Risk-neutral μⱼ: {mu_jump_rn:.4f}")
    print(f"  Adjustment: {mu_jump - mu_jump_rn:.4f}")
    
    return params_rn


def analyze_jump_characteristics(returns, params, dt=1/252, threshold=3.0):

    jump_mask, jump_returns, normal_returns = detect_jumps_threshold(
        returns, threshold=threshold, dt=dt
    )
    
    analysis = {
        'n_total': len(returns),
        'n_jumps': len(jump_returns),
        'jump_frequency': len(jump_returns) / len(returns),
        'positive_jumps': np.sum(jump_returns > 0),
        'negative_jumps': np.sum(jump_returns < 0),
    }
    
    if len(jump_returns) > 0:
        analysis['mean_jump_size'] = np.mean(np.abs(jump_returns))
        analysis['largest_up_jump'] = np.max(jump_returns) if len(jump_returns) > 0 else 0
        analysis['largest_down_jump'] = np.min(jump_returns) if len(jump_returns) > 0 else 0
        
        # Compare to calibrated distribution
        theoretical_mean = params['mu_jump']
        theoretical_std = params['sigma_jump']
        empirical_mean = np.mean(jump_returns)
        empirical_std = np.std(jump_returns, ddof=1)
        
        analysis['calibration_quality'] = {
            'mean_error': abs(empirical_mean - theoretical_mean),
            'std_error': abs(empirical_std - theoretical_std),
        }
    
    return analysis


def simulate_jump_process(lambda_jump, mu_jump, sigma_jump, T=1.0, n_steps=252):

    dt = T / n_steps
    
    # Poisson number of jumps
    expected_jumps = lambda_jump * T
    n_jumps = np.random.poisson(expected_jumps)
    
    # Jump times (uniformly distributed)
    jump_times = np.random.uniform(0, T, n_jumps)
    jump_indices = (jump_times / dt).astype(int)
    jump_indices = np.clip(jump_indices, 0, n_steps - 1)
    
    # Jump sizes
    jump_sizes = np.random.normal(mu_jump, sigma_jump, n_jumps)
    
    # Aggregate jumps
    jumps = np.zeros(n_steps)
    for idx, size in zip(jump_indices, jump_sizes):
        jumps[idx] += size
    
    return jumps, n_jumps


def get_jump_summary(params):
    
    lines = []
    lines.append("="*50)
    lines.append("JUMP PARAMETERS SUMMARY")
    lines.append("="*50)
    
    lambda_jump = params['lambda_jump']
    mu_jump = params['mu_jump']
    sigma_jump = params['sigma_jump']
    
    lines.append(f"Jump Intensity: {lambda_jump:.2f} per year")
    lines.append(f"  → Expected jumps per day: {lambda_jump/252:.3f}")
    lines.append(f"  → Average days between jumps: {252/lambda_jump:.1f}")
    
    lines.append(f"\nJump Size Distribution:")
    lines.append(f"  Mean (μⱼ): {mu_jump:.4f} ({mu_jump*100:.2f}%)")
    lines.append(f"  Std (σⱼ): {sigma_jump:.4f} ({sigma_jump*100:.2f}%)")
    
    # Expected jump magnitude
    expected_abs_jump = sigma_jump * np.sqrt(2/np.pi) + abs(mu_jump)
    lines.append(f"  Expected |jump|: {expected_abs_jump:.2%}")
    
    # Probability ranges
    prob_large_jump = 1 - norm.cdf(0.05, mu_jump, sigma_jump)
    lines.append(f"\nJump Probabilities:")
    lines.append(f"  P(jump > 5%): {prob_large_jump:.2%}")
    
    lines.append("="*50)
    
    return "\n".join(lines)