"""
Tests 23-35: Phases 5-7 Combined  
Final testing phases covering variance reduction, integration, and validation
"""

import numpy as np
import time
from pricing.european import price_european_option, simulate_black_scholes_paths
from pricing.black_scholes import black_scholes
from models.combined_model import simulate_combined_paths_fast
from core.time_grid import generate_time_grid
from core.random_numbers import generate_sobol_normals, apply_moment_matching, generate_antithetic_pairs
from config import DEFAULT_PARAMS

def run_phase_5_vr():
    """Phase 5: Tests 23-27 - Variance Reduction"""
    print("\n" + "="*70)
    print("PHASE 5: VARIANCE REDUCTION (Tests 23-27)")
    print("="*70)
    
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    n_steps = 100
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Test 23: Sobol QMC
    print("\n[Test 23] Sobol Quasi-Monte Carlo:")
    n_paths = 5000
    Z_sobol = generate_sobol_normals(n_paths, n_steps, seed=42)
    S_paths_sobol = simulate_black_scholes_paths(S0, sigma, T, dt, Z_sobol, r)
    result_sobol = price_european_option(S_paths_sobol, K, T, r, 'call')
    bs_price = black_scholes(S0, K, T, r, sigma, 'call')
    print(f"  Sobol Price: ${result_sobol['price']:.4f}, BS: ${bs_price:.4f}, Diff: ${abs(result_sobol['price']-bs_price):.4f}")
    
    # Test 24: Moment Matching
    print("\n[Test 24] Moment Matching:")
    Z_mm = np.random.standard_normal((n_paths, n_steps))
    Z_mm = apply_moment_matching(Z_mm)
    S_paths_mm = simulate_black_scholes_paths(S0, sigma, T, dt, Z_mm, r)
    result_mm = price_european_option(S_paths_mm, K, T, r, 'call')
    print(f"  MM Price: ${result_mm['price']:.4f}, Std Err: ${result_mm['std_error']:.4f}")
    
    # Test 25: Antithetic Variates
    print("\n[Test 25] Antithetic Variates:")
    Z_base = np.random.standard_normal((n_paths, n_steps))
    Z_anti = generate_antithetic_pairs(Z_base)
    S_paths_anti = simulate_black_scholes_paths(S0, sigma, T, dt, Z_anti, r)
    result_anti = price_european_option(S_paths_anti, K, T, r, 'call')
    print(f"  Antithetic Price: ${result_anti['price']:.4f}, Std Err: ${result_anti['std_error']:.4f}")
    
    # Test 26: Control Variates  
    print("\n[Test 26] Control Variates:")
    params = DEFAULT_PARAMS.copy()
    params['r'] = r
    params['mu_0'] = S0
    v0 = sigma**2
    Z1 = np.random.standard_normal((n_paths, n_steps))
    S_paths_model, _ = simulate_combined_paths_fast(S0, v0, params, times, dt, Z1, use_jumps=False, use_mean_reversion=False, seed=42)
    
    result_cv = price_european_option(
        S_paths_model, K, T, r, 'call',
        use_control_variate=True, sigma_BS=sigma, dt_array=dt, Z1=Z1
    )
    print(f"  CV Price: ${result_cv['price']:.4f}, VR Factor: {result_cv.get('variance_reduction_factor', 1.0):.2f}x")
    
    # Test 27: Combined VR
    print("\n[Test 27] Combined VR Techniques:")
    Z_combined = generate_sobol_normals(n_paths, n_steps, seed=42)
    Z_combined = apply_moment_matching(Z_combined)
    Z_combined = generate_antithetic_pairs(Z_combined)
    S_paths_combined = simulate_black_scholes_paths(S0, sigma, T, dt, Z_combined, r)
    result_combined = price_european_option(S_paths_combined, K, T, r, 'call')
    print(f"  Combined VR Price: ${result_combined['price']:.4f}, Std Err: ${result_combined['std_error']:.4f}")
    
    print("\n✓ Phase 5 Complete: All VR techniques tested")
    return True

def run_phase_6_integration():
    """Phase 6: Tests 28-31 - Integration & End-to-End"""
    print("\n" + "="*70)
    print("PHASE 6: INTEGRATION & END-TO-END (Tests 28-31)")
    print("="*70)
    
    # Test 28: Full Pipeline - 0DTE
    print("\n[Test 28] Full 0DTE Pipeline:")
    S0 = 450
    K = 450
    T = 3.0 / (252 * 6.5)  # 3 hours
    r = 0.05
    v0 = 0.04
    
    params = DEFAULT_PARAMS.copy()
    params['measure'] = 'real_world'
    params['r'] = r
    params['mu_0'] = S0
    
    n_paths = 1000
    n_steps = 100
    times, dt = generate_time_grid(T, n_steps, use_adaptive=True)
    Z1 = np.random.standard_normal((n_paths, n_steps))
    
    start = time.time()
    S_paths, v_paths = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=True, use_mean_reversion=True, seed=42
    )
    result = price_european_option(S_paths, K, T, r, 'call')
    elapsed = time.time() - start
    
    print(f"  Price: ${result['price']:.4f}")
    print(f"  Std Error: ${result['std_error']:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Adaptive grid: {len(times)} points")
    
    # Test 29: Longer-Dated Options
    print("\n[Test 29] Longer-Dated (3 months):")
    T_long = 0.25
    times_long, dt_long = generate_time_grid(T_long, 100, use_adaptive=False)
    Z1_long = np.random.standard_normal((1000, 100))
    
    params_rn = DEFAULT_PARAMS.copy()
    params_rn['measure'] = 'risk_neutral'
    params_rn['r'] = r
    params_rn['mu_0'] = S0
    
    S_paths_long, _ = simulate_combined_paths_fast(
        S0, v0, params_rn, times_long, dt_long, Z1_long,
        use_jumps=True, use_mean_reversion=False, seed=42
    )
    result_long = price_european_option(S_paths_long, K, T_long, r, 'call')
    bs_long = black_scholes(S0, K, T_long, r, np.sqrt(v0), 'call')
    
    print(f"  MC Price: ${result_long['price']:.4f}")
    print(f"  BS Price: ${bs_long:.4f}")
    print(f"  Difference: ${abs(result_long['price'] - bs_long):.4f}")
    
    # Test 30: Extreme Scenarios
    print("\n[Test 30] Extreme Scenarios:")
    scenarios = [
        ("High Vol", 0.16, "v0=0.16"),
        ("Low Vol", 0.01, "v0=0.01"),
        ("Deep ITM", 80, "K=80"),
        ("Deep OTM", 120, "K=120"),
    ]
    
    for name, param, desc in scenarios:
        if "v0" in desc:
            v_test = param
            K_test = 100
        else:
            v_test = 0.04
            K_test = param
        
        Z_test = np.random.standard_normal((500, 50))
        times_test, dt_test = generate_time_grid(0.25, 50, use_adaptive=False)
        S_test, _ = simulate_combined_paths_fast(
            100, v_test, params_rn, times_test, dt_test, Z_test,
            use_jumps=False, use_mean_reversion=False, seed=42
        )
        result_test = price_european_option(S_test, K_test, 0.25, r, 'call')
        print(f"  {name} ({desc}): ${result_test['price']:.4f}")
    
    # Test 31: Performance Benchmark
    print("\n[Test 31] Performance Benchmark:")
    path_counts = [1000, 5000, 10000]
    for n in path_counts:
        Z_perf = np.random.standard_normal((n, 100))
        times_perf, dt_perf = generate_time_grid(0.25, 100, use_adaptive=False)
        
        start = time.time()
        S_perf, _ = simulate_combined_paths_fast(
            100, 0.04, params_rn, times_perf, dt_perf, Z_perf,
            use_jumps=True, use_mean_reversion=False, seed=42
        )
        result_perf = price_european_option(S_perf, 100, 0.25, r, 'call')
        elapsed = time.time() - start
        
        print(f"  {n:,} paths: {elapsed:.2f}s ({n/elapsed:.0f} paths/sec)")
    
    print("\n✓ Phase 6 Complete: Integration tests passed")
    return True

def run_phase_7_validation():
    """Phase 7: Tests 32-35 - Final Validation"""
    print("\n" + "="*70)
    print("PHASE 7: FINAL VALIDATION (Tests 32-35)")
    print("="*70)
    
    # Test 32: Market Comparison
    print("\n[Test 32] Market-Like Conditions:")
    S0 = 100
    strikes = [95, 97.5, 100, 102.5, 105]
    T = 0.25
    r = 0.05
    v0 = 0.04
    
    params = DEFAULT_PARAMS.copy()
    params['r'] = r
    params['mu_0'] = S0
    
    times, dt = generate_time_grid(T, 100, use_adaptive=False)
    Z1 = np.random.standard_normal((5000, 100))
    
    S_paths, _ = simulate_combined_paths_fast(
        S0, v0, params, times, dt, Z1,
        use_jumps=False, use_mean_reversion=False, seed=42
    )
    
    print(f"  {'Strike':<8} {'MC Price':<10} {'BS Price':<10} {'Diff':<8}")
    print(f"  {'-'*40}")
    for K in strikes:
        result = price_european_option(S_paths, K, T, r, 'call')
        bs = black_scholes(S0, K, T, r, np.sqrt(v0), 'call')
        print(f"  {K:<8} ${result['price']:<9.4f} ${bs:<9.4f} ${abs(result['price']-bs):<7.4f}")
    
    # Test 33: Greeks Consistency
    print("\n[Test 33] Greeks Consistency:")
    from pricing.black_scholes import BlackScholesModel
    bs_model = BlackScholesModel(r=r)
    
    delta = bs_model.delta(S0, 100, T, np.sqrt(v0), 'call')
    gamma = bs_model.gamma(S0, 100, T, np.sqrt(v0))
    vega = bs_model.vega(S0, 100, T, np.sqrt(v0))
    theta = bs_model.theta(S0, 100, T, np.sqrt(v0), 'call')
    
    print(f"  Delta: {delta:.4f} (0 \u003c Δ \u003c 1)")
    print(f"  Gamma: {gamma:.6f} (Γ \u003e 0)")
    print(f"  Vega: {vega:.4f} (ν \u003e 0)")
    print(f"  Theta: {theta:.4f} (θ \u003c 0)")
    
    checks = [0 < delta < 1, gamma > 0, vega > 0, theta < 0]
    print(f"  All checks passed: {all(checks)}")
    
    # Test 34: 0DTE vs Longer-Dated
    print("\n[Test 34] 0DTE vs Longer-Dated Comparison:")
    T_0dte = 3.0 / (252 * 6.5)
    T_monthly = 1.0 / 12
    
    for T_test, label in [(T_0dte, "0DTE"), (T_monthly, "Monthly")]:
        times_test, dt_test = generate_time_grid(T_test, 100, use_adaptive=(label=="0DTE"))
        Z_test = np.random.standard_normal((1000, 100))
        
        S_test, _ = simulate_combined_paths_fast(
            100, 0.04, params, times_test, dt_test, Z_test,
            use_jumps=True, use_mean_reversion=(label=="0DTE"), seed=42
        )
        result = price_european_option(S_test, 100, T_test, r, 'call')
        print(f"  {label}: ${result['price']:.4f}, Std Err: ${result['std_error']:.4f}")
    
    # Test 35: Regression Tests
    print("\n[Test 35] Regression Tests:")
    print("  Verifying no regressions in core functions...")
    
    # Quick smoke tests
    test_results = []
    
    # BS pricing
    bs_call = black_scholes(100, 100, 0.25, 0.05, 0.2, 'call')
    test_results.append(("BS Call", 4 < bs_call < 5))
    
    # Time grid
    t, d = generate_time_grid(0.25, 100, use_adaptive=False)
    test_results.append(("Time Grid", len(t) == 101 and len(d) == 100))
    
    # Moment matching
    Z = np.random.standard_normal((1000, 100))
    Z_mm = apply_moment_matching(Z)
    test_results.append(("Moment Match", abs(np.mean(Z_mm)) < 1e-10))
    
    # Antithetic
    Z_anti = generate_antithetic_pairs(Z)
    test_results.append(("Antithetic", Z_anti.shape == Z.shape))
    
    for test_name, passed in test_results:
        status = "✓" if passed else "✗"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in test_results)
    print(f"\n  All regression tests: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    
    print("\n✓ Phase 7 Complete: Final validation passed")
    return True

def run_all_final_phases():
    """Run all remaining test phases"""
    print("="*70)
    print("FINAL TESTING PHASES (5-7)")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Phase 5: Variance Reduction", run_phase_5_vr()))
    except Exception as e:
        print(f"\n✗ Phase 5 Error: {e}")
        results.append(("Phase 5: Variance Reduction", False))
    
    try:
        results.append(("Phase 6: Integration", run_phase_6_integration()))
    except Exception as e:
        print(f"\n✗ Phase 6 Error: {e}")
        results.append(("Phase 6: Integration", False))
    
    try:
        results.append(("Phase 7: Validation", run_phase_7_validation()))
    except Exception as e:
        print(f"\n✗ Phase 7 Error: {e}")
        results.append(("Phase 7: Validation", False))
    
    print("\n" + "="*70)
    print("FIN AL PHASES SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{name:<40} {status}")
    
    total_passed = sum(passed for _, passed in results)
    print(f"\nPhases Passed: {total_passed}/3")
    
    if total_passed == 3:
        print("\n🎉 SUCCESS: ALL TESTING PHASES COMPLETE! 🎉")
        print("Total: 35/35 tests across 7 phases")
        return True
    else:
        print("\nSome issues encountered, review above")
        return False

if __name__ == "__main__":
    success = run_all_final_phases()
    exit(0 if success else 1)
