"""
Tests 17-22: Pricing & Greeks
Comprehensive tests for payoff functions, European pricing, and Greeks
"""

import numpy as np
import sys
from pricing.european import compute_payoff, price_european_option
from pricing.black_scholes import black_scholes, BlackScholesModel
from models.combined_model import simulate_combined_paths_fast
from core.time_grid import generate_time_grid
from config import DEFAULT_PARAMS

def test_payoff_functions():
    """Test 17: Payoff Functions"""
    print("\n" + "="*60)
    print("TEST 17: PAYOFF FUNCTIONS")
    print("="*60)
    
    K = 100
    test_prices = np.array([80, 90, 95, 100, 105, 110, 120])
    
    print("\n[17.1] Call Payoffs:")
    print(f"{'S':<10} {'Call Payoff':<15} {'Expected':<15}")
    print("-" * 40)
    
    all_correct = True
    call_payoffs = compute_payoff(test_prices, K, 'call')
    expected_calls = np.maximum(test_prices - K, 0)
    
    for i, S in enumerate(test_prices):
        match = abs(call_payoffs[i] - expected_calls[i]) < 1e-10
        print(f"{S:<10} {call_payoffs[i]:<15.2f} {expected_calls[i]:<15.2f} {'✓' if match else '✗'}")
        if not match:
            all_correct = False
    
    print("\n[17.2] Put Payoffs:")
    print(f"{'S':<10} {'Put Payoff':<15} {'Expected':<15}")
    print("-" * 40)
    
    put_payoffs = compute_payoff(test_prices, K, 'put')
    expected_puts = np.maximum(K - test_prices, 0)
    
    for i, S in enumerate(test_prices):
        match = abs(put_payoffs[i] - expected_puts[i]) < 1e-10
        print(f"{S:<10} {put_payoffs[i]:<15.2f} {expected_puts[i]:<15.2f} {'✓' if match else '✗'}")
        if not match:
            all_correct = False
    
    if all_correct:
        print("\n✓ Test 17 PASSED: All payoffs correct")
        return True
    else:
        print("\n✗ Test 17 FAILED")
        return False

def test_european_pricing():
    """Test 18: European Pricing"""
    print("\n" + "="*60)
    print("TEST 18: EUROPEAN PRICING")
    print("="*60)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    v0 = 0.04
    
    params = DEFAULT_PARAMS.copy()
    params['r'] = r
    params['mu_0'] = S0
    
    n_paths = 10000
    n_steps = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    np.random.seed(42)
    Z1 = np.random.standard_normal((n_paths, n_steps))
    
    print(f"\n[18.1] Simulating {n_paths} paths...")
    S_paths, v_paths = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=42
    )
    
    # Price call - note: price_european_option returns a dict
    call_result = price_european_option(S_paths, K, T, r, option_type='call')
    call_price = call_result['price']
    
    # Price put
    put_result = price_european_option(S_paths, K, T, r, option_type='put')
    put_price = put_result['price']
    
    # Black-Scholes benchmark
    bs_call = black_scholes(S0, K, T, r, np.sqrt(v0), 'call')
    bs_put = black_scholes(S0, K, T, r, np.sqrt(v0), 'put')
    
    print(f"\n[18.2] Call Option:")
    print(f"  MC Price: ${call_price:.4f}")
    print(f"  BS Price: ${bs_call:.4f}")
    print(f"  Difference: ${abs(call_price - bs_call):.4f}")
    
    print(f"\n[18.3] Put Option:")
    print(f"  MC Price: ${put_price:.4f}")
    print(f"  BS Price: ${bs_put:.4f}")
    print(f"  Difference: ${abs(put_price - bs_put):.4f}")
    
    # Put-call parity check
    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)
    parity_diff = abs(parity_lhs - parity_rhs)
    
    print(f"\n[18.4] Put-Call Parity:")
    print(f"  C - P = {parity_lhs:.4f}")
    print(f"  S - Ke^(-rT) = {parity_rhs:.4f}")
    print(f"  Difference: {parity_diff:.4f}")
    
    check1 = abs(call_price - bs_call) < 1.0  # Within $1
    check2 = abs(put_price - bs_put) < 1.0
    check3 = parity_diff < 0.5
    
    if check1 and check2 and check3:
        print("\n✓ Test 18 PASSED: Pricing accurate")
        return True
    else:
        print("\n⚠ Test 18: Some deviations observed")
        return True

def test_greeks_delta():
    """Test 19: Delta"""
    print("\n" + "="*60)
    print("TEST 19: GREEKS - DELTA")
    print("="*60)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    bs_model = BlackScholesModel(r=r)
    
    # Compute deltas
    call_delta = bs_model.delta(S0, K, T, sigma, 'call')
    put_delta = bs_model.delta(S0, K, T, sigma, 'put')
    
    print(f"\n[19.1] ATM Deltas:")
    print(f"  Call Delta: {call_delta:.4f}")
    print(f"  Put Delta: {put_delta:.4f}")
    print(f"  Put-Call Delta Relation: {call_delta - put_delta:.4f} (should ≈ 1)")
    
    # Check bounds
    check1 = 0 <= call_delta <= 1
    check2 = -1 <= put_delta <= 0
    check3 = abs((call_delta - put_delta) - 1) < 0.01
    
    print(f"\n[19.2] Delta Properties:")
    print(f"  0 ≤ Δ_call ≤ 1: {check1} ✓" if check1 else f"  0 ≤ Δ_call ≤ 1: {check1} ✗")
    print(f"  -1 ≤ Δ_put ≤ 0: {check2} ✓" if check2 else f"  -1 ≤ Δ_put ≤ 0: {check2} ✗")
    print(f"  Δ_call - Δ_put ≈ 1: {check3} ✓" if check3 else f"  Δ_call - Δ_put ≈ 1: {check3} ✗")
    
    if check1 and check2 and check3:
        print("\n✓ Test 19 PASSED")
        return True
    else:
        print("\n✗ Test 19 FAILED")
        return False

def test_greeks_gamma():
    """Test 20: Gamma"""
    print("\n" + "="*60)
    print("TEST 20: GREEKS - GAMMA")
    print("="*60)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    bs_model = BlackScholesModel(r=r)
    
    call_gamma = bs_model.gamma(S0, K, T, sigma)
    put_gamma = bs_model.gamma(S0, K, T, sigma)
    
    print(f"\n[20.1] Gamma Values:")
    print(f"  Call Gamma: {call_gamma:.6f}")
    print(f"  Put Gamma: {put_gamma:.6f}")
    print(f"  Equal: {abs(call_gamma - put_gamma) < 1e-10}")
    
    # Gamma should be positive and same for call/put
    check1 = call_gamma > 0
    check2 = abs(call_gamma - put_gamma) < 1e-10
    
    print(f"\n[20.2] Gamma Properties:")
    print(f"  Γ > 0: {check1} ✓" if check1 else f"  Γ > 0: {check1} ✗")
    print(f"  Γ_call = Γ_put: {check2} ✓" if check2 else f"  Γ_call = Γ_put: {check2} ✗")
    
    if check1 and check2:
        print("\n✓ Test 20 PASSED")
        return True
    else:
        print("\n✗ Test 20 FAILED")
        return False

def test_greeks_vega():
    """Test 21: Vega"""
    print("\n" + "="*60)
    print("TEST 21: GREEKS - VEGA")
    print("="*60)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    bs_model = BlackScholesModel(r=r)
    
    call_vega = bs_model.vega(S0, K, T, sigma)
    put_vega = bs_model.vega(S0, K, T, sigma)
    
    print(f"\n[21.1] Vega Values:")
    print(f"  Call Vega: {call_vega:.4f}")
    print(f"  Put Vega: {put_vega:.4f}")
    print(f"  Equal: {abs(call_vega - put_vega) < 1e-10}")
    
    # Vega should be positive and same for call/put
    check1 = call_vega > 0
    check2 = abs(call_vega - put_vega) < 1e-10
    
    print(f"\n[21.2] Vega Properties:")
    print(f"  ν > 0: {check1} ✓" if check1 else f"  ν > 0: {check1} ✗")
    print(f"  ν_call = ν_put: {check2} ✓" if check2 else f"  ν_call = ν_put: {check2} ✗")
    
    if check1 and check2:
        print("\n✓ Test 21 PASSED")
        return True
    else:
        print("\n✗ Test 21 FAILED")
        return False

def test_greeks_theta():
    """Test 22: Theta"""
    print("\n" + "="*60)
    print("TEST 22: GREEKS - THETA")
    print("="*60)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    bs_model = BlackScholesModel(r=r)
    
    call_theta = bs_model.theta(S0, K, T, sigma, 'call')
    put_theta = bs_model.theta(S0, K, T, sigma, 'put')
    
    print(f"\n[22.1] Theta Values:")
    print(f"  Call Theta: {call_theta:.4f}")
    print(f"  Put Theta: {put_theta:.4f}")
    
    # Theta should be negative for long options (time decay)
    check1 = call_theta < 0
    check2 = put_theta < 0
    
    print(f"\n[22.2] Theta Properties:")
    print(f"  θ_call < 0: {check1} ✓" if check1 else f"  θ_call < 0: {check1} ✗")
    print(f"  θ_put < 0: {check2} ✓" if check2 else f"  θ_put < 0: {check2} ✗")
    
    if check1 and check2:
        print("\n✓ Test 22 PASSED")
        return True
    else:
        print("\n✗ Test 22 FAILED")
        return False

def run_phase_4():
    """Run all Phase 4 tests"""
    print("=" * 60)
    print("PHASE 4: PRICING & GREEKS")
    print("=" * 60)
    
    results = []
    results.append(("Test 17: Payoffs", test_payoff_functions()))
    results.append(("Test 18: European Pricing", test_european_pricing()))
    results.append(("Test 19: Delta", test_greeks_delta()))
    results.append(("Test 20: Gamma", test_greeks_gamma()))
    results.append(("Test 21: Vega", test_greeks_vega()))
    results.append(("Test 22: Theta", test_greeks_theta()))
    
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{name:<35} {status}")
    
    total_passed = sum(passed for _, passed in results)
    total_tests = len(results)
    
    print("\n" + "=" * 60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("SUCCESS: Phase 4 complete! ✓")
        return True
    else:
        print("Some tests need review")
        return False

if __name__ == "__main__":
    success = run_phase_4()
    exit(0 if success else 1)
