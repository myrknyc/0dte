"""
Tests for Phase 1 Enhancements:
  - CVaR computation
  - Vol seasonality weights
  - Adaptive path logic
  - Theta/Greeks exit
"""

import numpy as np
import sys


# ─────────────────────────────────── CVaR ───────────────────────────────────

def test_cvar_computation():
    """CVaR₉₅ should equal the mean of the worst 5% of payoffs."""
    print("\n" + "=" * 60)
    print("TEST: CVaR Computation in price_european_option")
    print("=" * 60)

    from pricing.european import price_european_option
    from models.combined_model import simulate_combined_paths_fast
    from core.time_grid import generate_time_grid
    from config import DEFAULT_PARAMS

    S0, K, T, r = 100.0, 100.0, 0.25, 0.05
    params = DEFAULT_PARAMS.copy()
    params['r'] = r

    times, dt = generate_time_grid(T, 50, use_adaptive=False)
    np.random.seed(42)
    n_paths = 5000
    Z1 = np.random.standard_normal((n_paths, 50))

    S_paths, _ = simulate_combined_paths_fast(
        S0, 0.04, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=42
    )

    result = price_european_option(S_paths, K, T, r, option_type='call')

    assert 'cvar_95' in result, "cvar_95 missing from result"
    cvar = result['cvar_95']

    # Manually compute expected CVaR
    S_T = S_paths[:, -1]
    payoffs = np.maximum(S_T - K, 0) * np.exp(-r * T)
    sorted_p = np.sort(payoffs)
    n_tail = max(1, int(0.05 * len(sorted_p)))
    expected_cvar = float(np.mean(sorted_p[:n_tail]))

    diff = abs(cvar - expected_cvar)
    ok = diff < 0.01
    print(f"  CVaR₉₅:    {cvar:.4f}")
    print(f"  Expected:  {expected_cvar:.4f}")
    print(f"  Diff:      {diff:.6f}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_cvar_is_lower_tail():
    """CVaR₉₅ should be ≤ the overall mean price."""
    print("\n" + "=" * 60)
    print("TEST: CVaR is Lower Tail")
    print("=" * 60)

    from pricing.european import price_european_option
    from models.combined_model import simulate_combined_paths_fast
    from core.time_grid import generate_time_grid
    from config import DEFAULT_PARAMS

    S0, K, T, r = 100.0, 100.0, 0.25, 0.05
    params = DEFAULT_PARAMS.copy()
    params['r'] = r

    times, dt = generate_time_grid(T, 50, use_adaptive=False)
    np.random.seed(99)
    Z1 = np.random.standard_normal((10000, 50))

    S_paths, _ = simulate_combined_paths_fast(
        S0, 0.04, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=99
    )

    result = price_european_option(S_paths, K, T, r, option_type='call')
    ok = result['cvar_95'] <= result['price']
    print(f"  CVaR₉₅={result['cvar_95']:.4f} <= price={result['price']:.4f}: {'✓' if ok else '✗'}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ────────────────────────────── Vol Seasonality ──────────────────────────────

def test_diurnal_ushape():
    """Weights at open/close should be higher than midday."""
    print("\n" + "=" * 60)
    print("TEST: Diurnal U-Shape")
    print("=" * 60)

    from calibration.vol_seasonality import compute_diurnal_weights

    fracs = np.linspace(0, 1, 100)
    w = compute_diurnal_weights(fracs)

    w_open = w[0]       # t=0 (market open)
    w_mid = w[50]        # t=0.5 (midday)
    w_close = w[-1]      # t=1 (close)

    ok1 = w_open > w_mid
    ok2 = w_close > w_mid
    print(f"  w(open)={w_open:.3f} > w(mid)={w_mid:.3f}: {'✓' if ok1 else '✗'}")
    print(f"  w(close)={w_close:.3f} > w(mid)={w_mid:.3f}: {'✓' if ok2 else '✗'}")

    ok = ok1 and ok2
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_diurnal_normalization():
    """Mean of weights should be ~1.0."""
    print("\n" + "=" * 60)
    print("TEST: Diurnal Normalization")
    print("=" * 60)

    from calibration.vol_seasonality import compute_diurnal_weights

    fracs = np.linspace(0, 1, 390)  # one per minute
    w = compute_diurnal_weights(fracs)
    mean_w = np.mean(w)

    ok = abs(mean_w - 1.0) < 0.01
    print(f"  mean(w) = {mean_w:.6f}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: |mean - 1.0| < 0.01")
    return ok


def test_session_fraction_mapping():
    """Simulation times should map to session fractions correctly."""
    print("\n" + "=" * 60)
    print("TEST: Session Fraction Mapping")
    print("=" * 60)

    from calibration.vol_seasonality import simulation_time_to_session_fraction

    # If we're at session_frac=0.5, T covers remaining half
    T = 0.005  # some trading-year duration
    times = np.linspace(0, T, 5)
    fracs = simulation_time_to_session_fraction(times, T, 0.5)

    ok1 = abs(fracs[0] - 0.5) < 0.01    # starts at current position
    ok2 = abs(fracs[-1] - 1.0) < 0.01   # ends at close
    ok3 = all(fracs[i] <= fracs[i+1] for i in range(len(fracs)-1))  # monotonic

    print(f"  fracs = {fracs}")
    print(f"  starts at ~0.5: {'✓' if ok1 else '✗'}")
    print(f"  ends at ~1.0: {'✓' if ok2 else '✗'}")
    print(f"  monotonic: {'✓' if ok3 else '✗'}")

    ok = ok1 and ok2 and ok3
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ───────────────────────────── Theta/Greeks Exit ─────────────────────────────

def test_theta_exit_fires_near_close():
    """Theta exit should fire when close to EOD and profitable."""
    print("\n" + "=" * 60)
    print("TEST: Theta Exit Fires Near Close")
    print("=" * 60)

    from trading.paper_trader import PaperTrader
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = {
        'exit_mode': 'hybrid',
        'tp_pct': 0.50,   # high TP so it doesn't fire first
        'sl_pct': 0.50,
        'exit_time_minutes': 999,
        'eod_exit_time': '15:55',
        'greeks_exit': {
            'enabled': True,
            'theta_decay_pct': 0.80,
            'lookforward_minutes': 15,
            'min_profit_pct': 0.05,
        },
    }

    class FakePaperTrader:
        def __init__(self):
            self.cfg = cfg

    pt = FakePaperTrader()
    pt._check_exit_a = PaperTrader._check_exit_a.__get__(pt, FakePaperTrader)

    trade = {
        'entry_fill': {'mid': 1.50},
        'action': 'BUY',
        'strike': 600.0,
        'option_type': 'call',
    }

    # 15:50 ET, 5 min before EOD — profitable trade with lots of time value
    ts_et = datetime(2026, 3, 3, 15, 50, tzinfo=ZoneInfo('America/New_York'))
    spot = 601.0  # slightly ITM

    # bid/ask → mid=1.80, so +20% profit
    reason = pt._check_exit_a(trade, bid=1.70, ask=1.90, hold_minutes=30,
                               quote_fresh=True, ts_et=ts_et, spot=spot)

    print(f"  Exit reason: {reason}")
    ok = reason == 'theta_decay'
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: should be theta_decay")
    return ok


def test_theta_exit_no_fire_if_losing():
    """Theta exit should NOT fire on losing trades."""
    print("\n" + "=" * 60)
    print("TEST: Theta Exit Does Not Fire if Losing")
    print("=" * 60)

    from trading.paper_trader import PaperTrader
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = {
        'exit_mode': 'hybrid', 'tp_pct': 0.50, 'sl_pct': 0.50,
        'exit_time_minutes': 999, 'eod_exit_time': '15:55',
        'greeks_exit': {'enabled': True, 'theta_decay_pct': 0.80,
                        'lookforward_minutes': 15, 'min_profit_pct': 0.05},
    }

    class FakePT:
        def __init__(self):
            self.cfg = cfg

    pt = FakePT()
    pt._check_exit_a = PaperTrader._check_exit_a.__get__(pt, FakePT)

    trade = {
        'entry_fill': {'mid': 2.00},
        'action': 'BUY', 'strike': 600.0, 'option_type': 'call',
    }

    ts_et = datetime(2026, 3, 3, 15, 50, tzinfo=ZoneInfo('America/New_York'))

    # Losing trade: bid/ask → mid=1.50 → -25%
    reason = pt._check_exit_a(trade, bid=1.40, ask=1.60, hold_minutes=30,
                               quote_fresh=True, ts_et=ts_et, spot=599.0)

    print(f"  Exit reason: {reason}")
    ok = reason != 'theta_decay'
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: should NOT be theta_decay")
    return ok


def test_theta_exit_disabled():
    """When greeks_exit disabled, theta exit should not fire."""
    print("\n" + "=" * 60)
    print("TEST: Theta Exit Disabled")
    print("=" * 60)

    from trading.paper_trader import PaperTrader
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = {
        'exit_mode': 'hybrid', 'tp_pct': 0.50, 'sl_pct': 0.50,
        'exit_time_minutes': 999, 'eod_exit_time': '15:55',
        'greeks_exit': {'enabled': False},
    }

    class FakePT:
        def __init__(self):
            self.cfg = cfg

    pt = FakePT()
    pt._check_exit_a = PaperTrader._check_exit_a.__get__(pt, FakePT)

    trade = {
        'entry_fill': {'mid': 1.50},
        'action': 'BUY', 'strike': 600.0, 'option_type': 'call',
    }

    ts_et = datetime(2026, 3, 3, 15, 50, tzinfo=ZoneInfo('America/New_York'))
    reason = pt._check_exit_a(trade, bid=1.70, ask=1.90, hold_minutes=30,
                               quote_fresh=True, ts_et=ts_et, spot=601.0)

    print(f"  Exit reason: {reason}")
    ok = reason != 'theta_decay'
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: should NOT be theta_decay")
    return ok


def run_all():
    results = []
    results.append(("CVaR Computation", test_cvar_computation()))
    results.append(("CVaR Lower Tail", test_cvar_is_lower_tail()))
    results.append(("Diurnal U-Shape", test_diurnal_ushape()))
    results.append(("Diurnal Normalization", test_diurnal_normalization()))
    results.append(("Session Fraction", test_session_fraction_mapping()))
    results.append(("Theta Exit Fires", test_theta_exit_fires_near_close()))
    results.append(("Theta No Fire Losing", test_theta_exit_no_fire_if_losing()))
    results.append(("Theta Disabled", test_theta_exit_disabled()))

    print("\n" + "=" * 60)
    print("PHASE 1 ENHANCEMENT TEST SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name:<25} {'PASS ✓' if ok else 'FAIL ✗'}")
    total = sum(ok for _, ok in results)
    print(f"\n  {total}/{len(results)} passed")
    return total == len(results)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
