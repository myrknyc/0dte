import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
import yfinance as yf

from pricing.black_scholes import black_scholes
from config import HESTON_BOUNDS, RISK_FREE_RATE


def calibrate_heston_live(ticker='SPY', lookback_days=60):
    print(f"Fetching {lookback_days} days of {ticker} data...")
    
    try:
        # Download historical data
        hist_data = yf.download(ticker, period=f"{lookback_days}d", 
                               interval='1d', progress=False)
        
        if hist_data.empty or len(hist_data) < 20:
            raise ValueError(f"Insufficient data: only {len(hist_data)} days")
        
        prices = hist_data['Close'].values
        print(f"Retrieved {len(prices)} price points")
        
        # Use existing calibration function
        params = calibrate_to_realized_vol(prices)
        return params
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using default Heston parameters")
        return {
            'kappa': 2.0,
            'theta_v': 0.04,
            'sigma_v': 0.3,
            'rho': -0.7,
            'v0': 0.04
        }



def estimate_realized_variance(prices, window=20, periods_per_year=252):
    returns = np.diff(np.log(prices))
    
    # Rolling variance
    realized_var = []
    for i in range(len(returns) - window + 1):
        window_returns = returns[i:i+window]
        var = np.var(window_returns, ddof=1) * periods_per_year
        realized_var.append(var)
    
    return np.array(realized_var)


def estimate_price_vol_correlation(prices, window=60):
    returns = np.diff(np.log(prices))
    
    # Estimate rolling volatility
    vol_estimates = []
    for i in range(20, len(returns)):
        vol = np.std(returns[i-20:i]) * np.sqrt(252)
        vol_estimates.append(vol)
    
    vol_changes = np.diff(vol_estimates)
    aligned_returns = returns[20:-1]
    
    # Correlation between returns and vol changes
    if len(aligned_returns) >= window:
        rho = np.corrcoef(aligned_returns[-window:], vol_changes[-window:])[0, 1]
    else:
        rho = -0.7  # Default leverage effect
    
    # Ensure reasonable bounds
    rho = np.clip(rho, -0.99, -0.1)
    
    return rho


def calibrate_to_realized_vol(price_history, method='moment_matching',
                               periods_per_year=252):
  
    # Extract prices
    if hasattr(price_history, 'values'):
        prices = price_history.values
    else:
        prices = np.array(price_history)
    
    if len(prices) < 60:
        warnings.warn(f"Only {len(prices)} data points. Recommend 60+ for reliable calibration.")
    
    print(f"Calibrating Heston to {len(prices)} data points (annualization={periods_per_year})...")
    
    # [1] Estimate current variance (v₀)
    # Use larger window for intraday data (120 periods ≈ 2 hours of 1-min data)
    v0_window = min(121, len(prices))  # 120 returns
    recent_returns = np.diff(np.log(prices[-v0_window:]))
    v0 = np.var(recent_returns, ddof=1) * periods_per_year
    
    # [2] Estimate long-term variance (θ_v)
    all_returns = np.diff(np.log(prices))
    theta_v = np.var(all_returns, ddof=1) * periods_per_year
    
    # Floor v0: instantaneous vol shouldn't be less than half of long-term vol
    # (if it is, it's likely a quiet-window artifact, not real low vol)
    v0_floor = 0.5 * theta_v
    if v0 < v0_floor:
        print(f"  ⚠ v0={np.sqrt(v0):.2%} suspiciously low vs θ_v={np.sqrt(theta_v):.2%}. "
              f"Flooring to {np.sqrt(v0_floor):.2%}")
        v0 = v0_floor
    
    # [3] Estimate variance mean reversion speed (κ)
    # For κ, downsample to 15-min intervals — AR(1) on 1-min rolling windows
    # is far too noisy and always hits the 10.0 ceiling.
    downsample_factor = 15 if periods_per_year > 1000 else 1  # only for intraday
    prices_for_kappa = prices[::downsample_factor] if downsample_factor > 1 else prices
    kappa_periods_per_year = periods_per_year // downsample_factor if downsample_factor > 1 else periods_per_year
    
    realized_vars = estimate_realized_variance(
        prices_for_kappa, window=20, periods_per_year=kappa_periods_per_year
    )
    
    if len(realized_vars) > 1:
        # Fit AR(1): v[t] = α + β·v[t-1]
        # Then: κ = -log(β) × 252 (convert to continuous time)
        X = realized_vars[:-1]
        Y = realized_vars[1:]
        
        # OLS
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        beta = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
        
        # Convert to continuous time mean reversion
        if 0 < beta < 1:
            kappa = -np.log(beta) * kappa_periods_per_year
            kappa = np.clip(kappa, 0.1, 10.0)
        else:
            kappa = 2.0  # Default
    else:
        kappa = 2.0  # Default if insufficient data
    
    # [4] Estimate volatility of volatility (σ_v)
    if len(realized_vars) > 1:
        vol_of_vol = np.std(np.diff(realized_vars), ddof=1) * np.sqrt(kappa_periods_per_year)
        sigma_v = np.clip(vol_of_vol, 0.1, 2.0)
    else:
        sigma_v = 0.3  # Default
    
    # [5] Estimate price-vol correlation (ρ)
    rho = estimate_price_vol_correlation(prices)
    
    # Build parameter dictionary
    params = {
        'kappa': kappa,
        'theta_v': theta_v,
        'sigma_v': sigma_v,
        'rho': rho,
        'v0': v0,
    }
    
    # Check Feller condition
    feller_lhs = 2 * kappa * theta_v
    feller_rhs = sigma_v**2
    feller_satisfied = feller_lhs >= feller_rhs
    
    print(f"Calibration results:")
    print(f"  v₀ (current vol): {np.sqrt(v0):.2%}")
    print(f"  θ_v (long-term vol): {np.sqrt(theta_v):.2%}")
    print(f"  κ (mean reversion): {kappa:.2f}")
    print(f"  σ_v (vol of vol): {sigma_v:.2f}")
    print(f"  ρ (correlation): {rho:.2f}")
    print(f"  Feller condition: {'✓ Satisfied' if feller_satisfied else '✗ Violated'}")
    
    return params


def calibrate_to_market_prices(options_data, S0, r=RISK_FREE_RATE, 
                               initial_guess=None, bounds=None):
    if bounds is None:
        bounds = HESTON_BOUNDS
    
    # Compute market prices and vegas
    market_prices = []
    market_vegas = []
    strikes = []
    times = []
    option_types = []
    
    for opt in options_data:
        K = opt['strike']
        T = opt['T']
        opt_type = opt['type']
        iv = opt['implied_vol']
        
        # Market price
        market_price = opt.get('price')
        if market_price is None:
            # Compute from IV if not provided
            market_price = black_scholes(S0, K, T, r, iv, opt_type)
        
        # Market vega (from Black-Scholes)
        d1 = (np.log(S0/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        
        market_prices.append(market_price)
        market_vegas.append(vega)
        strikes.append(K)
        times.append(T)
        option_types.append(opt_type)
    
    market_prices = np.array(market_prices)
    market_vegas = np.array(market_vegas)
    
    # Vega weights (normalized)
    total_vega = np.sum(market_vegas)
    weights = market_vegas / total_vega if total_vega > 0 else np.ones(len(market_vegas)) / len(market_vegas)
    
    print(f"Calibrating to {len(market_prices)} option prices...")
    print(f"Strike range: ${min(strikes):.0f} - ${max(strikes):.0f}")
    
    # Objective function: Price-based error
    def objective(x):
        kappa, theta_v, sigma_v, rho, v0 = x
        
        # Check Feller condition (soft penalty)
        feller_lhs = 2 * kappa * theta_v
        feller_rhs = sigma_v**2
        if feller_lhs < feller_rhs:
            return 1e10  # Large penalty
        
        total_error = 0.0
        
        for i in range(len(market_prices)):
            # Price option with Heston (simplified - would use full MC)
            # For now, use Black-Scholes with adjusted vol as placeholder
            # TODO: Replace with actual Heston pricing
            model_vol = np.sqrt(theta_v)  # Simplified
            model_price = black_scholes(S0, strikes[i], times[i], r, model_vol, option_types[i])
            
            # Vega-weighted price error
            price_error = model_price - market_prices[i]
            weighted_error = (price_error / market_vegas[i])**2
            
            total_error += weights[i] * weighted_error
        
        return total_error
    
    # Initial guess
    if initial_guess is None:
        # Use realized vol as starting point
        recent_vol = np.sqrt(np.mean([opt['implied_vol']**2 for opt in options_data]))
        initial_guess = [
            2.0,                    # kappa
            recent_vol**2,          # theta_v
            0.3,                    # sigma_v
            -0.7,                   # rho
            recent_vol**2           # v0
        ]
    
    # Optimize
    print("Running optimization...")
    result = minimize(
        objective,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )
    
    if not result.success:
        warnings.warn(f"Optimization did not converge: {result.message}")
        print("Trying global optimization...")
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            disp=False
        )
    
    # Extract parameters
    kappa, theta_v, sigma_v, rho, v0 = result.x
    
    params = {
        'kappa': kappa,
        'theta_v': theta_v,
        'sigma_v': sigma_v,
        'rho': rho,
        'v0': v0,
    }
    
    # Report results
    print(f"\nCalibration complete:")
    print(f"  κ = {kappa:.2f}")
    print(f"  θ_v = {theta_v:.4f} (vol: {np.sqrt(theta_v):.2%})")
    print(f"  σ_v = {sigma_v:.2f}")
    print(f"  ρ = {rho:.2f}")
    print(f"  v₀ = {v0:.4f} (vol: {np.sqrt(v0):.2%})")
    print(f"  Final objective: {result.fun:.6f}")
    
    # Check Feller
    feller_satisfied = 2 * kappa * theta_v >= sigma_v**2
    print(f"  Feller condition: {'✓' if feller_satisfied else '✗'}")
    
    return params, result

def compare_calibration_methods(price_history, options_data, S0, r=RISK_FREE_RATE):
    print("="*60)
    print("CALIBRATION METHOD COMPARISON")
    print("="*60)
    
    # Method 1: Real-world (realized vol)
    print("\n[1] Real-World Calibration (Realized Volatility)")
    print("-"*60)
    params_rw = calibrate_to_realized_vol(price_history)
    
    # Method 2: Risk-neutral (market prices)
    print("\n[2] Risk-Neutral Calibration (Market Prices)")
    print("-"*60)
    params_rn, _ = calibrate_to_market_prices(options_data, S0, r)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Parameter':<15} {'Real-World':<15} {'Risk-Neutral':<15} {'Difference'}")
    print("-"*60)
    
    for key in ['kappa', 'theta_v', 'sigma_v', 'rho', 'v0']:
        val_rw = params_rw[key]
        val_rn = params_rn[key]
        diff = abs(val_rw - val_rn)
        
        if key in ['theta_v', 'v0']:
            # Show as volatility
            vol_rw = np.sqrt(val_rw)
            vol_rn = np.sqrt(val_rn)
            print(f"{key:<15} {vol_rw:<15.2%} {vol_rn:<15.2%} {abs(vol_rw-vol_rn):.2%}")
        else:
            print(f"{key:<15} {val_rw:<15.3f} {val_rn:<15.3f} {diff:.3f}")
    
    comparison = {
        'real_world': params_rw,
        'risk_neutral': params_rn,
    }
    
    return comparison


def validate_heston_params(params):
    issues = []
    
    kappa = params['kappa']
    theta_v = params['theta_v']
    sigma_v = params['sigma_v']
    rho = params['rho']
    v0 = params['v0']
    
    # Check positivity
    if kappa <= 0:
        issues.append(f"kappa must be positive, got {kappa}")
    if theta_v <= 0:
        issues.append(f"theta_v must be positive, got {theta_v}")
    if sigma_v <= 0:
        issues.append(f"sigma_v must be positive, got {sigma_v}")
    if v0 <= 0:
        issues.append(f"v0 must be positive, got {v0}")
    
    # Check correlation bounds
    if not -1 < rho < 1:
        issues.append(f"rho must be in (-1, 1), got {rho}")
    
    # Check Feller condition
    feller_lhs = 2 * kappa * theta_v
    feller_rhs = sigma_v**2
    if feller_lhs < feller_rhs:
        issues.append(f"Feller condition violated: 2κθ={feller_lhs:.4f} < σ_v²={feller_rhs:.4f}")
    
    # Check reasonable ranges
    if kappa > 20:
        issues.append(f"kappa very high ({kappa:.2f}), may cause numerical issues")
    if sigma_v > 3:
        issues.append(f"sigma_v very high ({sigma_v:.2f}), may cause instability")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues