"""
Regime Detector — canonical regime classification and adaptive parameters.

Shared source of truth for regime buckets used by:
  - backtest_metrics.py (analytics)
  - paper_trader.py (decision-layer thresholds)
  - trading_system.py (jump param blending)
"""

import math
from typing import Dict, Tuple

# ── Canonical regime buckets (intraday_move_pct) ─────────────
# These cutoffs MUST match analytics. Do not redefine elsewhere.
REGIME_BUCKETS = [
    ('calm',     0.0,  0.5),
    ('normal',   0.5,  1.0),
    ('volatile', 1.0,  2.0),
    ('extreme',  2.0,  float('inf')),
]


def classify(intraday_move_pct: float) -> str:
    """Classify current regime from absolute intraday spot move %.

    Args:
        intraday_move_pct: absolute % move from open (e.g. 0.8 = 0.8%)

    Returns:
        One of 'calm', 'normal', 'volatile', 'extreme', or 'unknown'.
    """
    if intraday_move_pct is None or not math.isfinite(intraday_move_pct):
        return 'unknown'
    move = abs(intraday_move_pct)
    for name, lo, hi in REGIME_BUCKETS:
        if lo <= move < hi:
            return name
    return 'unknown'


# ── Regime-conditioned filter thresholds (H4) ────────────────
_REGIME_THRESHOLDS = {
    'calm':     {'min_edge': 0.015, 'min_confidence': 0.45, 'max_cvar_loss': -1.50, 'min_eu': 0.00},
    'normal':   {'min_edge': 0.020, 'min_confidence': 0.50, 'max_cvar_loss': -2.00, 'min_eu': 0.00},
    'volatile': {'min_edge': 0.030, 'min_confidence': 0.60, 'max_cvar_loss': -2.50, 'min_eu': 0.02},
    'extreme':  {'min_edge': 0.050, 'min_confidence': 0.70, 'max_cvar_loss': -3.00, 'min_eu': 0.05},
}

# ── Regime-conditioned exit parameters (H4 extension) ────────
# Wider TP/SL in volatile regimes to avoid noise stops on 0DTE.
_REGIME_EXIT_PARAMS = {
    'calm':     {'tp_pct': 0.15, 'sl_pct': 0.12},
    'normal':   {'tp_pct': 0.20, 'sl_pct': 0.15},
    'volatile': {'tp_pct': 0.30, 'sl_pct': 0.25},
    'extreme':  {'tp_pct': 0.40, 'sl_pct': 0.35},
}


def get_adjusted_thresholds(regime: str) -> Dict[str, float]:
    """Return filter threshold overrides for the given regime.

    Returns a dict of filter keys → values.  Caller should overlay these
    on top of the base filter config.
    """
    return dict(_REGIME_THRESHOLDS.get(regime, _REGIME_THRESHOLDS['normal']))


def get_exit_params(regime: str,
                    default_tp: float = 0.20,
                    default_sl: float = 0.15) -> Dict[str, float]:
    """Return regime-conditioned TP/SL percentages.

    Falls back to caller-supplied defaults for unknown regimes.

    Args:
        regime: current regime string
        default_tp: fallback take-profit percentage
        default_sl: fallback stop-loss percentage

    Returns:
        {'tp_pct': float, 'sl_pct': float}
    """
    params = _REGIME_EXIT_PARAMS.get(regime)
    if params:
        return dict(params)
    return {'tp_pct': default_tp, 'sl_pct': default_sl}


# ── Regime-keyed jump priors (#5) ────────────────────────────
_REGIME_JUMP_PRIORS = {
    'calm':     {'lambda_jump': 2.0, 'mu_jump': -0.01, 'sigma_jump': 0.02},
    'normal':   {'lambda_jump': 5.0, 'mu_jump': -0.02, 'sigma_jump': 0.03},
    'volatile': {'lambda_jump': 10.0, 'mu_jump': -0.03, 'sigma_jump': 0.05},
    'extreme':  {'lambda_jump': 20.0, 'mu_jump': -0.05, 'sigma_jump': 0.08},
}


def get_jump_prior(regime: str) -> Dict[str, float]:
    """Return regime-keyed jump parameter priors for blending."""
    return dict(_REGIME_JUMP_PRIORS.get(regime, _REGIME_JUMP_PRIORS['normal']))


def regime_blend_weight(n_returns: int, variance_stability: float) -> float:
    """Deterministic blend weight for calibrated params vs regime prior.

    w → 1 when calibration is strong (many returns, stable variance).
    w → 0 when calibration is weak.

    Args:
        n_returns: number of intraday returns used in latest calibration
        variance_stability: ratio of recent variance to long-run variance
                           (≈1.0 means stable)

    Returns:
        w ∈ [0.2, 0.9]
    """
    # Sample size component: saturates at n=50
    w_sample = min(1.0, max(0.0, n_returns / 50.0))

    # Stability component: penalize when variance ratio is far from 1×
    w_stable = max(0.0, 1.0 - abs(variance_stability - 1.0))

    # Geometric mean — both conditions must hold
    w = (w_sample * w_stable) ** 0.5

    # Clamp: never fully trust either source
    return max(0.2, min(0.9, w))


def blend_jump_params(calibrated: Dict[str, float],
                      regime: str,
                      n_returns: int,
                      variance_stability: float) -> Dict[str, float]:
    """Blend calibrated jump params with regime prior using shrinkage weight.

    Args:
        calibrated: {'lambda_jump': ..., 'mu_jump': ..., 'sigma_jump': ...}
        regime: current regime string
        n_returns: calibration sample size
        variance_stability: variance ratio (recent / long-run)

    Returns:
        Blended params dict.
    """
    prior = get_jump_prior(regime)
    w = regime_blend_weight(n_returns, variance_stability)

    return {
        'lambda_jump': w * calibrated.get('lambda_jump', prior['lambda_jump'])
                        + (1 - w) * prior['lambda_jump'],
        'mu_jump':     w * calibrated.get('mu_jump', prior['mu_jump'])
                        + (1 - w) * prior['mu_jump'],
        'sigma_jump':  w * calibrated.get('sigma_jump', prior['sigma_jump'])
                        + (1 - w) * prior['sigma_jump'],
    }
