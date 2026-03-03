"""
Heston Characteristic Function Pricing & IV Surface Calibration

Semi-analytical Heston pricing via Gil-Pelaez inversion of the
characteristic function.  Used to calibrate (κ, θ_v, σ_v, ρ, v0)
against the observed option chain.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
import warnings

from pricing.black_scholes import black_scholes
from config import RISK_FREE_RATE, HESTON_BOUNDS


# ────────────────────────────────────────────────────────────────
#  Heston characteristic function  (Albrecher rotation-count form)
# ────────────────────────────────────────────────────────────────

def _heston_cf(u, S0, K, T, r, kappa, theta_v, sigma_v, rho, v0):
    """Heston characteristic function φ(u) for log-price."""
    iu = 1j * u

    # d = sqrt( (ρσ_v iu - κ)² + σ_v²(iu + u²) )
    a = rho * sigma_v * iu - kappa
    b = sigma_v ** 2 * (iu + u ** 2)
    d = np.sqrt(a ** 2 + b)

    # g = (κ - ρσ_v iu - d) / (κ - ρσ_v iu + d)
    g_num = kappa - rho * sigma_v * iu - d
    g_den = kappa - rho * sigma_v * iu + d

    # Guard division by zero
    if abs(g_den) < 1e-30:
        g_den = 1e-30

    g = g_num / g_den

    exp_neg_dT = np.exp(-d * T)

    # C and D coefficients
    C = (kappa * theta_v / (sigma_v ** 2)) * (
        g_num * T - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )

    D = (g_num / (sigma_v ** 2)) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )

    # φ(u) = exp(C + D·v₀ + iu·ln(S₀·e^{rT}))
    phi = np.exp(C + D * v0 + iu * np.log(S0 * np.exp(r * T)))

    return phi


def _integrand_P(u, j, S0, K, T, r, kappa, theta_v, sigma_v, rho, v0):
    """Integrand for Gil-Pelaez P₁ and P₂."""
    iu = 1j * u

    if j == 1:
        # P₁ integrand: use φ(u - i) / φ(-i)
        phi_u_mi = _heston_cf(u - 1j, S0, K, T, r, kappa, theta_v, sigma_v, rho, v0)
        phi_mi = _heston_cf(-1j, S0, K, T, r, kappa, theta_v, sigma_v, rho, v0)
        if abs(phi_mi) < 1e-30:
            return 0.0
        cf_val = phi_u_mi / phi_mi
    else:
        # P₂ integrand: use φ(u) directly
        cf_val = _heston_cf(u, S0, K, T, r, kappa, theta_v, sigma_v, rho, v0)

    integrand = np.real(np.exp(-iu * np.log(K)) * cf_val / (iu))
    return integrand


# ────────────────────────────────────────────────────────────────
#  Call/put pricing
# ────────────────────────────────────────────────────────────────

def heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0,
                    option_type='call'):
    """Price a European option using Heston's semi-analytical CF.

    Parameters
    ----------
    S, K, T, r : float
        Spot, strike, time-to-expiry (years), risk-free rate.
    kappa, theta_v, sigma_v, rho, v0 : float
        Heston parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float  — option price
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # Gil-Pelaez: P_j = 0.5 + (1/π) ∫₀^∞ Re[...] du
    args = (S, K, T, r, kappa, theta_v, sigma_v, rho, v0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        int1, _ = quad(lambda u: _integrand_P(u, 1, *args),
                       1e-8, 200, limit=200)
        int2, _ = quad(lambda u: _integrand_P(u, 2, *args),
                       1e-8, 200, limit=200)

    P1 = 0.5 + int1 / np.pi
    P2 = 0.5 + int2 / np.pi

    # Clamp probabilities to [0, 1]
    P1 = np.clip(float(np.real(P1)), 0.0, 1.0)
    P2 = np.clip(float(np.real(P2)), 0.0, 1.0)

    call_price = S * P1 - K * np.exp(-r * T) * P2
    call_price = max(call_price, 0.0)

    if option_type == 'call':
        return call_price
    else:
        # Put via put-call parity
        put_price = call_price - S + K * np.exp(-r * T)
        # Floor at discounted intrinsic for deep ITM puts
        put_intrinsic = max(K - S, 0.0) * np.exp(-r * T)
        return max(put_price, put_intrinsic)


def heston_cf_price_chain(S, strikes, T, r, kappa, theta_v, sigma_v,
                          rho, v0, option_type='call'):
    """Price a chain of options at once."""
    return np.array([
        heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0,
                        option_type)
        for K in strikes
    ])


# ────────────────────────────────────────────────────────────────
#  IV Surface Calibration
# ────────────────────────────────────────────────────────────────

def calibrate_to_iv_surface(option_chain, S0, T, r=RISK_FREE_RATE,
                            moneyness_range=(0.97, 1.03),
                            min_strikes=5,
                            verbose=True):
    """Calibrate Heston params by fitting the semi-analytical CF to the
    observed option chain.

    Parameters
    ----------
    option_chain : pandas.DataFrame
        Must have columns: 'strike', 'bid', 'ask', 'impliedVolatility'.
    S0 : float
        Current spot price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    moneyness_range : tuple
        (lo, hi) moneyness K/S range to include.
    min_strikes : int
        Minimum number of valid strikes required.
    verbose : bool

    Returns
    -------
    dict  — calibrated params  {'kappa', 'theta_v', 'sigma_v', 'rho', 'v0'}
    dict  — fit quality  {'rmse', 'n_strikes', 'residuals'}
    """
    import pandas as pd
    from scipy.stats import norm

    # ── Filter to liquid, near-ATM options ──
    df = option_chain.copy()
    df['moneyness'] = df['strike'] / S0
    df['mid'] = (df['bid'] + df['ask']) / 2.0
    df = df[
        (df['moneyness'] >= moneyness_range[0]) &
        (df['moneyness'] <= moneyness_range[1]) &
        (df['bid'] > 0) &
        (df['ask'] > 0) &
        (df['mid'] > 0.01)
    ].copy()

    if len(df) < min_strikes:
        raise ValueError(
            f"Only {len(df)} liquid strikes in moneyness range "
            f"{moneyness_range}; need {min_strikes}"
        )

    strikes = df['strike'].values.astype(float)
    market_prices = df['mid'].values.astype(float)

    # Vega weights (BS vega at market IV)
    ivs = df['impliedVolatility'].values.astype(float)
    ivs = np.clip(ivs, 0.05, 3.0)
    sqrt_T = np.sqrt(max(T, 1e-8))
    d1 = (np.log(S0 / strikes) + (r + 0.5 * ivs ** 2) * T) / (ivs * sqrt_T)
    vegas = S0 * norm.pdf(d1) * sqrt_T
    weights = vegas / np.sum(vegas)

    # ── Objective: vega-weighted RMSE ──
    def objective(x):
        kappa, theta_v, sigma_v, rho, v0 = x

        # Feller soft penalty
        if 2 * kappa * theta_v < sigma_v ** 2:
            return 1e6

        try:
            model_prices = heston_cf_price_chain(
                S0, strikes, T, r, kappa, theta_v, sigma_v, rho, v0, 'call'
            )
        except Exception:
            return 1e6

        residuals = (model_prices - market_prices) ** 2
        return float(np.sum(weights * residuals))

    # ── Global search + polish ──
    bounds = HESTON_BOUNDS

    if verbose:
        print(f"  IV surface: fitting {len(strikes)} strikes "
              f"(moneyness {moneyness_range})...")

    result = differential_evolution(
        objective, bounds, maxiter=60, tol=1e-6,
        seed=42, disp=False
    )

    # Polish with L-BFGS-B
    result2 = minimize(
        objective, result.x, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 100}
    )
    if result2.fun < result.fun:
        result = result2

    kappa, theta_v, sigma_v, rho, v0 = result.x

    params = {
        'kappa': float(kappa),
        'theta_v': float(theta_v),
        'sigma_v': float(sigma_v),
        'rho': float(rho),
        'v0': float(v0),
    }

    # ── Fit quality ──
    model_prices = heston_cf_price_chain(
        S0, strikes, T, r, kappa, theta_v, sigma_v, rho, v0, 'call'
    )
    residuals = model_prices - market_prices
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    quality = {
        'rmse': rmse,
        'n_strikes': len(strikes),
        'residuals': residuals.tolist(),
        'objective': float(result.fun),
    }

    if verbose:
        print(f"  IV surface fit: RMSE=${rmse:.4f} over {len(strikes)} strikes")
        print(f"    κ={kappa:.2f}  θ_v={theta_v:.4f} ({np.sqrt(theta_v):.1%})  "
              f"σ_v={sigma_v:.2f}  ρ={rho:.2f}  v0={v0:.4f} ({np.sqrt(v0):.1%})")

    return params, quality
