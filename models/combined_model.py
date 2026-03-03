import numpy as np
from config import MIN_VARIANCE, MIN_PRICE, RISK_FREE_RATE

from models.heston import simulate_variance_paths_fast as simulate_variance_paths
from models.jump_diffusion import generate_jumps_vectorized, compute_jump_compensator
from models.mean_reversion import compute_mu_t, compute_mean_reversion_drift


def simulate_combined_paths_fast(S0, v0, params, times, dt_array, Z1, Z2=None,
                                 use_jumps=True, use_mean_reversion=None, seed=None,
                                 verbose=False, diurnal_weights=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_paths, n_steps = Z1.shape
    
    # Validate measure
    measure = params.get('measure', 'risk_neutral')
    if measure not in ['risk_neutral', 'real_world']:
        raise ValueError(f"measure must be 'risk_neutral' or 'real_world', got {measure}")
    
    # Determine mean reversion
    if use_mean_reversion is None:
        use_mean_reversion = (measure == 'real_world')
    
    # Extract parameters
    kappa = params['kappa']
    theta_v = params['theta_v']
    sigma_v = params['sigma_v']
    r = params.get('r', RISK_FREE_RATE)
    
    if use_mean_reversion:
        theta_mr = params['theta_mr']
        mu_0 = params['mu_0']
        if mu_0 is None:
            raise ValueError("mu_0 must be specified for real_world measure")
    
    # Initialize
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    
    # [1] Variance paths
    v_paths = simulate_variance_paths(
        v0=v0, kappa=kappa, theta_v=theta_v, sigma_v=sigma_v,
        times=times, dt_array=dt_array, n_paths=n_paths, seed=seed
    )
    
    # --- Variance path diagnostic ---
    if verbose:
        v_mean = np.mean(v_paths)
        v_min = np.min(v_paths)
        v_max = np.max(v_paths)
        print(f"  [V DIAG] v0_in={v0:.6f} ({np.sqrt(v0):.2%}), "
              f"theta_v={theta_v:.6f}, kappa={kappa:.2f}, sigma_v={sigma_v:.4f}")
        print(f"  [V DIAG] v_paths: mean={v_mean:.6f} ({np.sqrt(v_mean):.2%}), "
              f"min={v_min:.6f}, max={v_max:.6f}")
    
    # [2] Jumps
    if use_jumps:
        lambda_jump = params['lambda_jump']
        mu_jump = params['mu_jump']
        sigma_jump = params['sigma_jump']
        jump_log_sizes = generate_jumps_vectorized(
            lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump
        )
    else:
        jump_log_sizes = np.zeros((n_paths, n_steps))
    
    # Compute jump compensator for risk-neutral drift (Merton's correction)
    # k = E[e^J - 1], so drift becomes r - 0.5v - λk
    jump_compensator = 0.0
    if use_jumps:
        jump_compensator = params['lambda_jump'] * compute_jump_compensator(
            params['mu_jump'], params['sigma_jump']
        )
    
    # [3] Price paths (vectorized over paths)
    for i in range(n_steps):
        t = times[i]
        dt = dt_array[i]
        
        # Floor variance before use (prevent sqrt of negative)
        v_i = np.maximum(v_paths[:, i], MIN_VARIANCE)
        
        # Apply intraday volatility seasonality (U-shape scaling)
        if diurnal_weights is not None:
            v_i = v_i * diurnal_weights[i]
        
        # Drift (vectorized): risk-neutral with jump compensator
        drift = (r - 0.5 * v_i - jump_compensator) * dt
        
        if use_mean_reversion:
            mu_t = compute_mu_t(mu_0, r, t)
            drift += -theta_mr * (S_paths[:, i] - mu_t) / S_paths[:, i] * dt
        
        # Diffusion (vectorized)
        diffusion = np.sqrt(v_i) * np.sqrt(dt) * Z1[:, i]
        
        # Jumps (already vectorized)
        jump = jump_log_sizes[:, i]
        
        # Update (vectorized)
        log_return = drift + diffusion + jump
        S_paths[:, i + 1] = S_paths[:, i] * np.exp(log_return)
        
        # Guard NaN/Inf — do NOT clip tails
        bad_mask = ~np.isfinite(S_paths[:, i + 1])
        S_paths[bad_mask, i + 1] = S_paths[bad_mask, i]
        S_paths[:, i + 1] = np.maximum(S_paths[:, i + 1], MIN_PRICE)
    
    v_paths = np.maximum(v_paths, MIN_VARIANCE)
    
    # --- Martingale assertion (risk-neutral only: E[S_T] should ≈ S0·e^(rT)) ---
    if measure == 'risk_neutral' and not use_mean_reversion:
        T_total = float(np.sum(dt_array))
        S_T = S_paths[:, -1]
        expected_ratio = np.exp(r * T_total)
        actual_ratio = np.mean(S_T) / S0
        se_ratio = np.std(S_T / S0) / np.sqrt(n_paths)
        if abs(actual_ratio - expected_ratio) > 3 * se_ratio and se_ratio > 0:
            print(f"⚠ MARTINGALE VIOLATION: E[S_T]/S0={actual_ratio:.6f}, "
                  f"expected={expected_ratio:.6f}, 3σ={3*se_ratio:.6f}")
    
    # --- One-shot simulation diagnostic ---
    if verbose:
        S_T = S_paths[:, -1]
        logret = np.log(S_T / S0)
        avg_v = np.mean(v_paths[:, :-1])
        T_total = float(np.sum(dt_array))
        expected_logret_std = np.sqrt(avg_v) * np.sqrt(T_total)
        print(f"\n  [SIM DIAG] S0={S0:.2f}, mean(S_T)={S_T.mean():.2f}, std(S_T)={S_T.std():.4f}")
        print(f"  [SIM DIAG] logret: mean={logret.mean():.6f}, std={logret.std():.6f}")
        print(f"  [SIM DIAG] expected logret std (sqrt(avg_v)*sqrt(T))={expected_logret_std:.6f}")
        print(f"  [SIM DIAG] avg_v={avg_v:.6f} (vol={np.sqrt(avg_v):.2%}), T={T_total:.6f}")
        print(f"  [SIM DIAG] P(S_T>S0)={np.mean(S_T > S0):.2%}, "  
              f"min(S_T)={S_T.min():.2f}, max(S_T)={S_T.max():.2f}")
    
    return S_paths, v_paths


def get_model_description(params):
    measure = params.get('measure', 'risk_neutral')
    
    lines = []
    lines.append("="*60)
    lines.append("COMBINED MODEL CONFIGURATION")
    lines.append("="*60)
    
    # Measure
    lines.append(f"\nMeasure: {measure}")
    if measure == 'risk_neutral':
        lines.append("  → Arbitrage-free pricing (no mean reversion)")
        lines.append("  → Use for: hedging, market making, fair value")
    else:
        lines.append("  → Statistical model (includes mean reversion)")
        lines.append("  → Use for: alpha signals, proprietary trading")
    
    # Components
    lines.append("\nActive Components:")
    lines.append(f"  ✓ Heston stochastic volatility (κ={params['kappa']:.2f})")
    
    if params.get('lambda_jump', 0) > 0:
        lines.append(f"  ✓ Jump-diffusion (λ={params['lambda_jump']:.2f}/year)")
    else:
        lines.append(f"  ✗ Jump-diffusion (disabled)")
    
    if measure == 'real_world' and params.get('theta_mr', 0) > 0:
        lines.append(f"  ✓ Mean reversion (θ={params['theta_mr']:.2f})")
    else:
        lines.append(f"  ✗ Mean reversion (disabled)")
    
    lines.append("="*60)
    
    return "\n".join(lines)