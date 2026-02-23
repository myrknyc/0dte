import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

from config import TRADING_DAYS_PER_YEAR, MINUTES_PER_TRADING_DAY
from models.mean_reversion import compute_half_life



def calibrate_from_intraday(price_data, vwap=None, dt=None):
    # Extract prices and volume
    if isinstance(price_data, pd.DataFrame):
        if 'Close' in price_data.columns:
            prices = price_data['Close'].values
        else:
            prices = price_data.values[:, 0]
        
        # Compute VWAP if volume available
        if vwap is None and 'Volume' in price_data.columns:
            volume = price_data['Volume'].values
            vwap = np.sum(prices * volume) / np.sum(volume)
            print(f"Computed VWAP from volume data: ${vwap:.2f}")
    else:
        prices = np.array(price_data)
    
    # Default VWAP if not provided
    if vwap is None:
        vwap = np.mean(prices)
        print(f"Using mean price as VWAP: ${vwap:.2f}")
    
    # Default time step (1 minute)
    if dt is None:
        dt = 1.0 / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    n = len(prices)
    print(f"Calibrating mean reversion from {n} intraday observations...")
    print(f"Time step: {dt * TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY:.1f} minutes")
    
    # Compute price changes
    price_changes = np.diff(prices)  # ΔS[i] = S[i+1] - S[i]
    lagged_prices = prices[:-1]      # S[i]
    
    # OLS Regression: ΔS = a + b·S + ε
    n_obs = len(price_changes)
    
    # Design matrix
    X = np.column_stack([np.ones(n_obs), lagged_prices])
    Y = price_changes
    
    # OLS: β = (X'X)^(-1) X'Y
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    a, b = beta
    
    # Recover OU parameters
    theta_mr = -b / dt
    
    # mu_0: prefer the passed-in VWAP over the OLS-derived mean level,
    # because the OLS estimate (-a/b) is unreliable when R² is low
    if vwap is not None:
        mu_0 = vwap
    elif abs(b) > 1e-10:
        mu_0 = -a / b
    else:
        mu_0 = np.mean(prices)
    
    # Ensure positive theta
    if theta_mr <= 0:
        warnings.warn(
            f"Estimated θ={theta_mr:.2f} is non-positive. "
            f"Using default θ=5.0. Check if data shows mean reversion."
        )
        theta_mr = 5.0
    
    # Compute goodness of fit
    Y_pred = a + b * lagged_prices
    residuals = Y - Y_pred
    
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - SS_res / SS_tot if SS_tot > 0 else 0
    
    std_residuals = np.std(residuals, ddof=2)
    
    # Compute half-life
    half_life_dict = compute_half_life(theta_mr)
    
    # Build result
    params = {
        'theta_mr': theta_mr,
        'mu_0': mu_0,
        'half_life_years': half_life_dict['years'],
        'half_life_days': half_life_dict['days'],
        'half_life_hours': half_life_dict['hours'],
        'half_life_minutes': half_life_dict['minutes'],
        'r_squared': r_squared,
        'std_residuals': std_residuals,
        'vwap': vwap,
        'n_observations': n_obs,
    }
    
    # Report results
    print(f"\nCalibration results:")
    print(f"  θ_mr (speed): {theta_mr:.2f} per year")
    print(f"  μ₀ (VWAP): ${mu_0:.2f}")
    print(f"  Half-life: {half_life_dict['hours']:.1f} hours ({half_life_dict['minutes']:.0f} minutes)")
    print(f"  R²: {r_squared:.4f}")
    print(f"  Residual std: ${std_residuals:.4f}")
    
    # Interpretation
    if theta_mr < 2:
        print(f"  → Weak mean reversion (slow return to VWAP)")
    elif theta_mr < 5:
        print(f"  → Moderate mean reversion (typical for 0DTE)")
    elif theta_mr < 10:
        print(f"  → Strong mean reversion (fast return to VWAP)")
    else:
        print(f"  → Very strong mean reversion (may be overfit)")
    
    # Check if price is far from VWAP
    current_price = prices[-1]
    deviation_pct = (current_price - vwap) / vwap * 100
    print(f"\n  Current price: ${current_price:.2f}")
    print(f"  Deviation from VWAP: {deviation_pct:+.2f}%")
    
    if abs(deviation_pct) > 1:
        direction = "below" if deviation_pct < 0 else "above"
        print(f"  → Price is {abs(deviation_pct):.1f}% {direction} VWAP")
        print(f"  → Model predicts mean reversion toward ${vwap:.2f}")
    
    return params

def calibrate_from_intraday_mle(price_data, vwap=None, dt=None, initial_guess=None):
    # Extract prices
    if isinstance(price_data, pd.DataFrame):
        prices = price_data['Close'].values if 'Close' in price_data.columns else price_data.values[:, 0]
    else:
        prices = np.array(price_data)
    
    if vwap is None:
        vwap = np.mean(prices)
    
    if dt is None:
        dt = 1.0 / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    print(f"Calibrating with MLE from {len(prices)} observations...")
    
    # Initialize with OLS if no guess provided
    if initial_guess is None:
        ols_params = calibrate_from_intraday(price_data, vwap=vwap, dt=dt)
        initial_guess = [
            ols_params['theta_mr'],
            ols_params['mu_0'],
            ols_params['std_residuals'] / np.sqrt(dt)  # Convert to continuous-time σ
        ]
    
    # Negative log-likelihood
    def neg_log_likelihood(x):
        theta, mu, sigma = x
        
        # Parameter constraints
        if theta <= 0 or sigma <= 0:
            return 1e10
        
        # Conditional moments for OU process
        exp_neg_theta_dt = np.exp(-theta * dt)
        
        nll = 0
        for i in range(len(prices) - 1):
            # Conditional mean
            mu_cond = mu + (prices[i] - mu) * exp_neg_theta_dt
            
            # Conditional variance
            var_cond = (sigma**2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
            std_cond = np.sqrt(var_cond)
            
            # Log-likelihood contribution
            ll_i = norm.logpdf(prices[i+1], mu_cond, std_cond)
            nll -= ll_i
        
        return nll
    
    # Bounds
    bounds = [
        (0.1, 50),           # theta: 0.1 to 50 per year
        (vwap * 0.9, vwap * 1.1),  # mu: ±10% of VWAP
        (0.001, 1.0),        # sigma: 0.1% to 100%
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
        warnings.warn(f"MLE did not converge: {result.message}")
    
    # Extract parameters
    theta_mr, mu_0, sigma_ou = result.x
    
    half_life_dict = compute_half_life(theta_mr)
    
    params = {
        'theta_mr': theta_mr,
        'mu_0': mu_0,
        'sigma_ou': sigma_ou,
        'half_life_hours': half_life_dict['hours'],
        'half_life_minutes': half_life_dict['minutes'],
        'log_likelihood': -result.fun,
        'vwap': vwap,
    }
    
    print(f"\nMLE Results:")
    print(f"  θ_mr: {theta_mr:.2f}")
    print(f"  μ₀: ${mu_0:.2f}")
    print(f"  σ: {sigma_ou:.4f}")
    print(f"  Half-life: {half_life_dict['hours']:.1f} hours")
    print(f"  Log-likelihood: {-result.fun:.2f}")
    
    return params, result



def calibrate_from_autocorrelation(price_data, vwap=None, dt=None, lag=1):

    # Extract prices
    if isinstance(price_data, pd.DataFrame):
        prices = price_data['Close'].values if 'Close' in price_data.columns else price_data.values[:, 0]
    else:
        prices = np.array(price_data)
    
    if vwap is None:
        vwap = np.mean(prices)
    
    if dt is None:
        dt = 1.0 / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    print(f"Calibrating from autocorrelation ({len(prices)} obs)...")
    
    # Deviations from VWAP
    deviations = prices - vwap
    
    # Compute autocorrelation at lag
    rho = np.corrcoef(deviations[:-lag], deviations[lag:])[0, 1]
    
    # Recover theta
    if 0 < rho < 1:
        theta_mr = -np.log(rho) / (lag * dt)
    else:
        warnings.warn(f"Invalid autocorrelation {rho:.3f}, using default θ=5")
        theta_mr = 5.0
    
    half_life_dict = compute_half_life(theta_mr)
    
    params = {
        'theta_mr': theta_mr,
        'mu_0': vwap,
        'autocorrelation': rho,
        'half_life_hours': half_life_dict['hours'],
        'half_life_minutes': half_life_dict['minutes'],
        'vwap': vwap,
    }
    
    print(f"  Lag-{lag} autocorrelation: {rho:.4f}")
    print(f"  θ_mr: {theta_mr:.2f}")
    print(f"  Half-life: {half_life_dict['hours']:.1f} hours")
    
    return params



def validate_mean_reversion(price_data, params, dt=None):
    
    if isinstance(price_data, pd.DataFrame):
        prices = price_data['Close'].values if 'Close' in price_data.columns else price_data.values[:, 0]
    else:
        prices = np.array(price_data)
    
    if dt is None:
        dt = 1.0 / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    theta_mr = params['theta_mr']
    mu_0 = params['mu_0']
    
    # [1] Check if deviations decay
    deviations = prices - mu_0
    abs_deviations = np.abs(deviations)
    
    # Correlation between |deviation| and future return
    price_changes = np.diff(prices)
    
    # When far from VWAP, should move toward VWAP
    sign_deviations = np.sign(deviations[:-1])
    sign_changes = np.sign(price_changes)
    
    # Mean reversion implies: negative correlation
    reversion_correlation = np.corrcoef(sign_deviations, sign_changes)[0, 1]
    
    # [2] Half-distance decay test
    # After half-life, deviation should be ~50% of original
    half_life_steps = int(params['half_life_minutes'] / (dt * TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY * 60))
    
    if half_life_steps < len(prices):
        initial_dev = abs_deviations[:-half_life_steps]
        final_dev = abs_deviations[half_life_steps:]
        
        mean_decay_ratio = np.mean(final_dev / (initial_dev + 1e-10))
    else:
        mean_decay_ratio = None
    
    # [3] Excursion probability
    # Price should spend ~50% time above and below VWAP
    above_vwap = np.sum(prices > mu_0) / len(prices)
    
    validation = {
        'reversion_correlation': reversion_correlation,
        'mean_decay_ratio': mean_decay_ratio,
        'time_above_vwap': above_vwap,
        'mean_abs_deviation': np.mean(abs_deviations),
        'max_deviation': np.max(abs_deviations),
    }
    
    print("\n" + "="*50)
    print("MEAN REVERSION VALIDATION")
    print("="*50)
    print(f"Reversion correlation: {reversion_correlation:.4f}")
    print(f"  (Negative = mean reverting)")
    
    if mean_decay_ratio is not None:
        print(f"Decay ratio after half-life: {mean_decay_ratio:.2f}")
        print(f"  (Should be ~0.5 for true OU)")
    
    print(f"Time above VWAP: {above_vwap:.1%}")
    print(f"  (Should be ~50%)")
    
    print(f"Mean |deviation|: ${validation['mean_abs_deviation']:.2f}")
    print(f"Max |deviation|: ${validation['max_deviation']:.2f}")
    
    # Overall assessment
    is_mean_reverting = reversion_correlation < -0.1
    print(f"\nMean reversion detected: {'✓ YES' if is_mean_reverting else '✗ NO'}")
    
    return validation


def plot_mean_reversion_diagnostics(price_data, params):
    if isinstance(price_data, pd.DataFrame):
        prices = price_data['Close'].values if 'Close' in price_data.columns else price_data.values[:, 0]
    else:
        prices = np.array(price_data)
    
    mu_0 = params['mu_0']
    
    diagnostics = {
        'prices': prices,
        'vwap': mu_0,
        'deviations': prices - mu_0,
        'abs_deviations': np.abs(prices - mu_0),
    }
    
    print("\nDiagnostic plots would show:")
    print("  1. Price vs VWAP over time")
    print("  2. Deviations from VWAP")
    print("  3. Autocorrelation function")
    print("  4. Residual distribution")
    
    return diagnostics


def get_mean_reversion_summary(params):
    
    lines = []
    lines.append("="*60)
    lines.append("MEAN REVERSION PARAMETERS SUMMARY")
    lines.append("="*60)
    
    theta = params['theta_mr']
    mu = params['mu_0']
    
    lines.append(f"Mean Reversion Speed: θ = {theta:.2f} per year")
    lines.append(f"Mean Level (VWAP): μ₀ = ${mu:.2f}")
    lines.append(f"\nHalf-Life:")
    lines.append(f"  {params['half_life_hours']:.1f} hours")
    lines.append(f"  {params['half_life_minutes']:.0f} minutes")
    
    if 'r_squared' in params:
        lines.append(f"\nGoodness of Fit:")
        lines.append(f"  R² = {params['r_squared']:.4f}")
    
    lines.append(f"\nInterpretation:")
    if theta < 2:
        lines.append("  → Weak mean reversion")
        lines.append("  → Price drifts slowly back to VWAP")
    elif theta < 5:
        lines.append("  → Moderate mean reversion (typical for SPY)")
        lines.append("  → Good for 0DTE modeling")
    elif theta < 10:
        lines.append("  → Strong mean reversion")
        lines.append("  → Price quickly returns to VWAP")
    else:
        lines.append("  → Very strong mean reversion")
        lines.append("  → May be overfit - use with caution")
    
    lines.append("="*60)
    
    return "\n".join(lines)