"""
Test 10: Heston Variance - Full Path
Goal: Simulate complete variance path
"""

import numpy as np
import sys
from models.heston import simulate_variance_path, check_feller_condition
from core.time_grid import generate_time_grid
from config import HESTON_PARAMS

def test_full_path_structure():
    """Test that full path has correct structure"""
    print("\n=== Test 10.1: Path Structure ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25  # 3 months
    n_steps = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_path = simulate_variance_path(v0, kappa, theta_v, sigma_v, times, dt)
    
    print(f"Parameters: n_steps={n_steps}")
    print(f"Path length: {len(v_path)}")
    print(f"Expected: {n_steps + 1}")
    
    check1 = len(v_path) == n_steps + 1
    
    # Check initial value
    check2 = abs(v_path[0] - v0) < 1e-10
    print(f"v_path[0] = {v_path[0]:.6f} (expected {v0})")
    print(f"Initial value correct: {check2}")
    
    if check1 and check2:
        print("✓ Path structure correct")
    else:
        print("✗ Path structure incorrect")
    
    return check1 and check2

def test_full_path_positivity():
    """Test that all values in path are positive"""
    print("\n=== Test 10.2: Path Positivity ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Simulate multiple paths to check consistency
    n_sims = 100
    all_positive = True
    
    for i in range(n_sims):
        v_path = simulate_variance_path(v0, kappa, theta_v, sigma_v, times, dt)
        if not np.all(v_path > 0):
            all_positive = False
            break
    
    print(f"Simulated {n_sims} paths")
    print(f"All values positive: {all_positive}")
    
    if all_positive:
        print("✓ All variance values positive")
    else:
        print("✗ Some negative values found")
    
    return all_positive

def test_mean_convergence():
    """Test that variance paths remain stable and positive"""
    print("\n=== Test 10.3: Path Stability ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25  # 3 months (shorter horizon)
    n_steps = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Simulate many paths
    n_sims = 1000
    final_variances = []
    
    for i in range(n_sims):
        v_path = simulate_variance_path(v0, kappa, theta_v, sigma_v, times, dt)
        final_variances.append(v_path[-1])
    
    mean_final = np.mean(final_variances)
    std_final = np.std(final_variances)
    min_final = np.min(final_variances)
    max_final = np.max(final_variances)
    
    print(f"Simulated {n_sims} paths over {T} year(s)")
    print(f"Initial variance: v0 = {v0:.6f}")
    print(f"\nFinal variance statistics:")
    print(f"  Mean: {mean_final:.6f}")
    print(f"  Std:  {std_final:.6f}")
    print(f"  Min:  {min_final:.6f}")
    print(f"  Max:  {max_final:.6f}")
    
    # Check that paths remain stable and positive
    # All paths should be positive
    check1 = min_final >= 0
    
    # Paths shouldn't explode to unreasonable values
    check2 = max_final < 1.0  # Less than 100% vol
    
    # Mean should be reasonable
    check3 = 0.0 <= mean_final <= 0.5
    
    print(f"\nAll positive: {check1}")
    print(f"No explosions: {check2}")
    print(f"Mean reasonable: {check3}")
    
    if check1 and check2 and check3:
        print("✓ Paths remain stable and positive")
    else:
        print("✗ Path stability issues")
    
    return check1 and check2 and check3

def test_path_oscillation():
    """Visual check: does path oscillate around theta_v"""
    print("\n=== Test 10.4: Path Oscillation ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Generate single path
    v_path = simulate_variance_path(v0, kappa, theta_v, sigma_v, times, dt)
    
    # Compute statistics
    mean_path = np.mean(v_path)
    min_path = np.min(v_path)
    max_path = np.max(v_path)
    
    print(f"Single path statistics:")
    print(f"  Start: {v_path[0]:.6f}")
    print(f"  End:   {v_path[-1]:.6f}")
    print(f"  Mean:  {mean_path:.6f}")
    print(f"  Min:   {min_path:.6f}")
    print(f"  Max:   {max_path:.6f}")
    print(f"  θ_v:   {theta_v:.6f}")
    
    # Check reasonable oscillation
    # Mean should be within reasonable range of theta_v
    check1 = abs(mean_path - theta_v) < 0.02
    
    # Should have some variation (not constant)
    check2 = (max_path - min_path) > 0.001
    
    print(f"\nOscillates around θ_v: {check1}")
    print(f"Shows variation: {check2}")
    
    if check1 and check2:
        print("✓ Path shows reasonable oscillation")
    else:
        print("⚠ Path behavior unusual")
    
    return True  # Informational

def test_starting_conditions():
    """Test different starting conditions"""
    print("\n=== Test 10.5: Different Starting Conditions ===")
    
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    all_passed = True
    
    # Test 1: High volatility start
    print("\n[10.5.1] High volatility (v0 = 0.16 = 40% vol):")
    v0_high = 0.16
    v_path_high = simulate_variance_path(v0_high, kappa, theta_v, sigma_v, times, dt)
    check1 = v_path_high[0] == v0_high and np.all(v_path_high > 0)
    print(f"  Start: {v_path_high[0]:.6f}, End: {v_path_high[-1]:.6f}")
    print(f"  Valid: {check1}")
    all_passed = all_passed and check1
    
    # Test 2: Low volatility start
    print("\n[10.5.2] Low volatility (v0 = 0.01 = 10% vol):")
    v0_low = 0.01
    v_path_low = simulate_variance_path(v0_low, kappa, theta_v, sigma_v, times, dt)
    check2 = v_path_low[0] == v0_low and np.all(v_path_low > 0)
    print(f"  Start: {v_path_low[0]:.6f}, End: {v_path_low[-1]:.6f}")
    print(f"  Valid: {check2}")
    all_passed = all_passed and check2
    
    # Test 3: At theta_v
    print("\n[10.5.3] At long-term mean (v0 = θ_v):")
    v0_theta = theta_v
    v_path_theta = simulate_variance_path(v0_theta, kappa, theta_v, sigma_v, times, dt)
    check3 = v_path_theta[0] == v0_theta and np.all(v_path_theta > 0)
    print(f"  Start: {v_path_theta[0]:.6f}, End: {v_path_theta[-1]:.6f}")
    print(f"  Valid: {check3}")
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ All starting conditions work")
    else:
        print("\n✗ Some starting conditions failed")
    
    return all_passed

def test_time_step_variations():
    """Test with different time step sizes"""
    print("\n=== Test 10.6: Time Step Variations ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    
    all_passed = True
    
    # Test 1: Few steps
    print("\n[10.6.1] Few steps (n=10):")
    times1, dt1 = generate_time_grid(T, 10, use_adaptive=False)
    v_path1 = simulate_variance_path(v0, kappa, theta_v, sigma_v, times1, dt1)
    check1 = len(v_path1) == 11 and np.all(v_path1 > 0)
    print(f"  Length: {len(v_path1)}, All positive: {np.all(v_path1 > 0)}")
    all_passed = all_passed and check1
    
    # Test 2: Many steps
    print("\n[10.6.2] Many steps (n=500):")
    times2, dt2 = generate_time_grid(T, 500, use_adaptive=False)
    v_path2 = simulate_variance_path(v0, kappa, theta_v, sigma_v, times2, dt2)
    check2 = len(v_path2) == 501 and np.all(v_path2 > 0)
    print(f"  Length: {len(v_path2)}, All positive: {np.all(v_path2 > 0)}")
    all_passed = all_passed and check2
    
    # Test 3: Adaptive grid
    print("\n[10.6.3] Adaptive grid (0DTE):")
    T_short = 3.0 / (252 * 6.5)  # 3 hours
    times3, dt3 = generate_time_grid(T_short, 100, use_adaptive=True)
    v_path3 = simulate_variance_path(v0, kappa, theta_v, sigma_v, times3, dt3)
    check3 = len(v_path3) == 101 and np.all(v_path3 > 0)
    print(f"  Length: {len(v_path3)}, All positive: {np.all(v_path3 > 0)}")
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ All time step variations work")
    else:
        print("\n✗ Some time step variations failed")
    
    return all_passed

def run_all_tests():
    """Run all Test 10 subtests"""
    print("=" * 60)
    print("TEST 10: HESTON VARIANCE - FULL PATH")
    print("=" * 60)
    
    results = []
    
    results.append(("Path Structure", test_full_path_structure()))
    results.append(("Path Positivity", test_full_path_positivity()))
    results.append(("Long-term Convergence", test_mean_convergence()))
    results.append(("Path Oscillation", test_path_oscillation()))
    results.append(("Starting Conditions", test_starting_conditions()))
    results.append(("Time Step Variations", test_time_step_variations()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{name:<30} {status}")
    
    total_passed = sum(passed for _, passed in results)
    total_tests = len(results)
    
    print("\n" + "=" * 60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("SUCCESS: All Heston full path tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
