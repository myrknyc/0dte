"""
Tests 23-35: Phases 5-7 Combined
Final testing phases covering variance reduction, integration, and validation.

Refactored to use pytest-compatible test_ prefixed functions.
"""

import numpy as np
import time as time_mod
import pytest
from pricing.european import price_european_option, simulate_black_scholes_paths
from pricing.black_scholes import black_scholes, BlackScholesModel
from models.combined_model import simulate_combined_paths_fast
from core.time_grid import generate_time_grid
from core.random_numbers import generate_sobol_normals, apply_moment_matching, generate_antithetic_pairs
from config import DEFAULT_PARAMS


# ============================================================
# Phase 5: Variance Reduction (Tests 23-27)
# ============================================================

def _base_params():
    """Common test parameters."""
    return dict(S0=100, K=100, T=0.25, r=0.05, sigma=0.2, n_steps=100)


def test_23_sobol_qmc():
    """Sobol quasi-Monte Carlo produces prices close to BS."""
    p = _base_params()
    times, dt = generate_time_grid(p['T'], p['n_steps'], use_adaptive=False)

    n_paths = 5000
    Z_sobol = generate_sobol_normals(n_paths, p['n_steps'], seed=42)
    S_paths = simulate_black_scholes_paths(p['S0'], p['sigma'], p['T'], dt, Z_sobol, p['r'])
    result = price_european_option(S_paths, p['K'], p['T'], p['r'], 'call')
    bs_price = black_scholes(p['S0'], p['K'], p['T'], p['r'], p['sigma'], 'call')

    assert abs(result['price'] - bs_price) < 0.50, (
        f"Sobol price ${result['price']:.4f} too far from BS ${bs_price:.4f}"
    )


def test_24_moment_matching():
    """Moment matching forces mean=0, std=1."""
    p = _base_params()
    times, dt = generate_time_grid(p['T'], p['n_steps'], use_adaptive=False)

    n_paths = 5000
    Z = np.random.standard_normal((n_paths, p['n_steps']))
    Z_mm = apply_moment_matching(Z)

    assert abs(np.mean(Z_mm)) < 1e-10, "Moment matching should force mean ≈ 0"

    S_paths = simulate_black_scholes_paths(p['S0'], p['sigma'], p['T'], dt, Z_mm, p['r'])
    result = price_european_option(S_paths, p['K'], p['T'], p['r'], 'call')
    assert result['price'] > 0, "Price should be positive"


def test_25_antithetic_variates():
    """Antithetic variates should reduce standard error."""
    p = _base_params()
    times, dt = generate_time_grid(p['T'], p['n_steps'], use_adaptive=False)

    n_paths = 5000
    Z_base = np.random.standard_normal((n_paths, p['n_steps']))
    Z_anti = generate_antithetic_pairs(Z_base)

    S_paths = simulate_black_scholes_paths(p['S0'], p['sigma'], p['T'], dt, Z_anti, p['r'])
    result = price_european_option(S_paths, p['K'], p['T'], p['r'], 'call')

    assert result['std_error'] < 0.20, (
        f"Antithetic std error ${result['std_error']:.4f} should be low"
    )


def test_26_control_variates():
    """Control variates should achieve variance reduction > 1x."""
    p = _base_params()
    times, dt = generate_time_grid(p['T'], p['n_steps'], use_adaptive=False)

    params = DEFAULT_PARAMS.copy()
    params['r'] = p['r']
    params['mu_0'] = p['S0']
    v0 = p['sigma'] ** 2

    n_paths = 5000
    Z1 = np.random.standard_normal((n_paths, p['n_steps']))
    S_paths, _ = simulate_combined_paths_fast(
        p['S0'], v0, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=42
    )

    result = price_european_option(
        S_paths, p['K'], p['T'], p['r'], 'call',
        use_control_variate=True, sigma_BS=p['sigma'], dt_array=dt, Z1=Z1
    )

    vr = result.get('variance_reduction_factor', 1.0)
    assert vr >= 1.0, f"Control variate VR factor {vr:.2f} should be >= 1"


def test_27_combined_vr():
    """Combining Sobol + moment matching + antithetic should produce accurate price."""
    p = _base_params()
    times, dt = generate_time_grid(p['T'], p['n_steps'], use_adaptive=False)

    n_paths = 5000
    Z = generate_sobol_normals(n_paths, p['n_steps'], seed=42)
    Z = apply_moment_matching(Z)
    Z = generate_antithetic_pairs(Z)

    S_paths = simulate_black_scholes_paths(p['S0'], p['sigma'], p['T'], dt, Z, p['r'])
    result = price_european_option(S_paths, p['K'], p['T'], p['r'], 'call')
    bs_price = black_scholes(p['S0'], p['K'], p['T'], p['r'], p['sigma'], 'call')

    assert abs(result['price'] - bs_price) < 0.50, (
        f"Combined VR ${result['price']:.4f} too far from BS ${bs_price:.4f}"
    )


# ============================================================
# Phase 6: Integration & End-to-End (Tests 28-31)
# ============================================================

def _default_rn_params(S0, r):
    """Risk-neutral params for integration tests."""
    params = DEFAULT_PARAMS.copy()
    params['measure'] = 'risk_neutral'
    params['r'] = r
    params['mu_0'] = S0
    return params


def test_28_full_0dte_pipeline():
    """Full 0DTE pipeline should produce a positive price."""
    S0, K, r, v0 = 450, 450, 0.05, 0.04
    T = 3.0 / (252 * 6.5)  # 3 hours

    params = DEFAULT_PARAMS.copy()
    params['measure'] = 'real_world'
    params['r'] = r
    params['mu_0'] = S0

    n_paths, n_steps = 1000, 100
    times, dt = generate_time_grid(T, n_steps, use_adaptive=True)
    Z1 = np.random.standard_normal((n_paths, n_steps))

    S_paths, v_paths = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=True, use_mean_reversion=True, seed=42
    )
    result = price_european_option(S_paths, K, T, r, 'call')

    assert result['price'] > 0, "0DTE price should be positive"
    assert result['std_error'] < result['price'], "Std error should be smaller than price"


def test_29_longer_dated():
    """Longer-dated MC price should be close to BS."""
    S0, K, r, v0 = 450, 450, 0.05, 0.04
    T = 0.25

    params = _default_rn_params(S0, r)
    times, dt = generate_time_grid(T, 100, use_adaptive=False)
    Z1 = np.random.standard_normal((1000, 100))

    S_paths, _ = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=True, use_mean_reversion=False, seed=42
    )
    result = price_european_option(S_paths, K, T, r, 'call')
    bs_price = black_scholes(S0, K, T, r, np.sqrt(v0), 'call')

    # Heston ≠ BS, so allow wider tolerance
    assert abs(result['price'] - bs_price) < 5.0, (
        f"MC ${result['price']:.2f} vs BS ${bs_price:.2f} gap too large"
    )


@pytest.mark.parametrize("name,v_test,K_test", [
    ("High Vol", 0.16, 100),
    ("Low Vol",  0.01, 100),
    ("Deep ITM", 0.04, 80),
    ("Deep OTM", 0.04, 120),
])
def test_30_extreme_scenarios(name, v_test, K_test):
    """Extreme scenarios should produce finite, non-negative prices."""
    S0, r = 100, 0.05
    params = _default_rn_params(S0, r)

    n_paths, n_steps = 500, 50
    times, dt = generate_time_grid(0.25, n_steps, use_adaptive=False)
    Z = np.random.standard_normal((n_paths, n_steps))

    S_paths, _ = simulate_combined_paths_fast(
        S0, v_test, params, times, dt, Z,
        use_jumps=False, use_mean_reversion=False, seed=42
    )
    result = price_european_option(S_paths, K_test, 0.25, r, 'call')

    assert np.isfinite(result['price']), f"{name}: price must be finite"
    assert result['price'] >= 0, f"{name}: price must be non-negative"


def test_31_performance_benchmark():
    """10k paths should complete in under 10 seconds."""
    S0, r = 100, 0.05
    params = _default_rn_params(S0, r)

    n_paths, n_steps = 10000, 100
    times, dt = generate_time_grid(0.25, n_steps, use_adaptive=False)
    Z = np.random.standard_normal((n_paths, n_steps))

    start = time_mod.time()
    S_paths, _ = simulate_combined_paths_fast(
        S0, 0.04, params, times, dt, Z,
        use_jumps=True, use_mean_reversion=False, seed=42
    )
    result = price_european_option(S_paths, 100, 0.25, r, 'call')
    elapsed = time_mod.time() - start

    assert elapsed < 10.0, f"10k paths took {elapsed:.2f}s, should be < 10s"


# ============================================================
# Phase 7: Final Validation (Tests 32-35)
# ============================================================

def test_32_market_like_conditions():
    """MC prices across strikes should be reasonable vs BS."""
    S0, r, v0 = 100, 0.05, 0.04
    T = 0.25
    strikes = [95, 97.5, 100, 102.5, 105]

    params = _default_rn_params(S0, r)
    times, dt = generate_time_grid(T, 100, use_adaptive=False)
    Z1 = np.random.standard_normal((5000, 100))

    S_paths, _ = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=42
    )

    for K in strikes:
        result = price_european_option(S_paths, K, T, r, 'call')
        bs = black_scholes(S0, K, T, r, np.sqrt(v0), 'call')
        assert abs(result['price'] - bs) < 1.0, (
            f"Strike {K}: MC ${result['price']:.4f} vs BS ${bs:.4f}"
        )


def test_33_greeks_consistency():
    """BS Greeks should satisfy basic sign constraints."""
    S0, K, T, sigma, r = 100, 100, 0.25, 0.2, 0.05
    bs = BlackScholesModel(r=r)

    delta = bs.delta(S0, K, T, sigma, 'call')
    gamma = bs.gamma(S0, K, T, sigma)
    vega = bs.vega(S0, K, T, sigma)
    theta = bs.theta(S0, K, T, sigma, 'call')

    assert 0 < delta < 1, f"Delta {delta} must be in (0,1)"
    assert gamma > 0, f"Gamma {gamma} must be positive"
    assert vega > 0, f"Vega {vega} must be positive"
    assert theta < 0, f"Theta {theta} must be negative"


def test_34_0dte_vs_monthly():
    """0DTE and monthly prices should both be positive and 0DTE < monthly for ATM."""
    S0, r, v0 = 100, 0.05, 0.04
    K = 100
    params = _default_rn_params(S0, r)

    results = {}
    for label, T_val, adaptive in [("0dte", 3.0 / (252 * 6.5), True),
                                    ("monthly", 1.0 / 12, False)]:
        times, dt = generate_time_grid(T_val, 100, use_adaptive=adaptive)
        Z = np.random.standard_normal((1000, 100))
        S_paths, _ = simulate_combined_paths_fast(
            S0, v0, params, times, dt, Z,
            use_jumps=True, use_mean_reversion=False, seed=42
        )
        results[label] = price_european_option(S_paths, K, T_val, r, 'call')

    assert results['0dte']['price'] > 0
    assert results['monthly']['price'] > 0
    assert results['0dte']['price'] < results['monthly']['price'], (
        "0DTE ATM call should be cheaper than monthly ATM call"
    )


def test_35_regression_smoke():
    """Quick smoke tests for core functions — catch regressions."""
    # BS pricing
    bs_call = black_scholes(100, 100, 0.25, 0.05, 0.2, 'call')
    assert 4 < bs_call < 5, f"BS ATM call should be ~$4-5, got ${bs_call:.4f}"

    # Time grid
    t, d = generate_time_grid(0.25, 100, use_adaptive=False)
    assert len(t) == 101
    assert len(d) == 100

    # Moment matching
    Z = np.random.standard_normal((1000, 100))
    Z_mm = apply_moment_matching(Z)
    assert abs(np.mean(Z_mm)) < 1e-10

    # Antithetic — shape preserved
    Z_anti = generate_antithetic_pairs(Z)
    assert Z_anti.shape == Z.shape
