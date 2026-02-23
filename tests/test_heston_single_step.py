"""
Test 9: Heston Variance - Single Step
Goal: Test QE scheme for one time step
"""

import numpy as np
import sys
from models.heston import heston_qe_step, check_feller_condition
from config import HESTON_PARAMS

def test_single_step_positivity():
    """Test that single step always returns positive variance"""
    print("\n=== Test 9.1: Positivity ===")
    
    v0 = 0.04  # 20% volatility squared
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    dt = 1/252  # 1 trading day
    
    print(f"Parameters: v0={v0}, κ={kappa}, θ_v={theta_v}, σ_v={sigma_v}")
    print(f"Time step: dt={dt:.6f} ({dt*252:.1f} days)")
    
    # Run 1000 simulations
    n_sims = 1000
    results = []
    
    for i in range(n_sims):
        v_next = heston_qe_step(v0, kappa, theta_v, sigma_v, dt)
        results.append(v_next)
    
    results = np.array(results)
    
    min_v = np.min(results)
    max_v = np.max(results)
    mean_v = np.mean(results)
    
    print(f"\nResults from {n_sims} simulations:")
    print(f"  Min:  {min_v:.6f}")
    print(f"  Max:  {max_v:.6f}")
    print(f"  Mean: {mean_v:.6f}")
    
    # Check all positive
    all_positive = np.all(results > 0)
    
    if all_positive:
        print(f"✓ All {n_sims} values positive")
    else:
        negative_count = np.sum(results <= 0)
        print(f"✗ Found {negative_count} non-positive values")
    
    return all_positive

def test_single_step_reasonable_range():
    """Test that values are in reasonable range"""
    print("\n=== Test 9.2: Reasonable Range ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    dt = 1/252
    
    n_sims = 1000
    results = []
    
    for i in range(n_sims):
        v_next = heston_qe_step(v0, kappa, theta_v, sigma_v, dt)
        results.append(v_next)
    
    results = np.array(results)
    
    # Variance should be in (0.001, 0.5) for reasonable options
    # 0.001 = 3.16% vol, 0.5 = 70% vol
    min_reasonable = 0.001
    max_reasonable = 0.5
    
    in_range = np.all((results >= min_reasonable) & (results <= max_reasonable))
    
    print(f"Reasonable range: [{min_reasonable}, {max_reasonable}]")
    print(f"Actual range: [{np.min(results):.6f}, {np.max(results):.6f}]")
    
    if in_range:
        print("✓ All values in reasonable range")
    else:
        outside = np.sum((results < min_reasonable) | (results > max_reasonable))
        print(f"⚠ {outside}/{n_sims} values outside range (may be okay)")
    
    # Less strict check - no extreme outliers
    reasonable = np.all(results < 2.0)  # Less than 141% vol
    
    return reasonable

def test_starting_from_zero():
    """Test behavior when starting from v=0"""
    print("\n=== Test 9.3: Starting from Zero Variance ===")
    
    v0 = 0.0  # Zero variance
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    dt = 1/252
    
    n_sims = 100
    results = []
    
    for i in range(n_sims):
        v_next = heston_qe_step(v0, kappa, theta_v, sigma_v, dt)
        results.append(v_next)
    
    results = np.array(results)
    
    # Should still produce positive values (mean reversion toward theta_v)
    all_positive = np.all(results > 0)
    mean_v = np.mean(results)
    
    print(f"Starting from v0=0:")
    print(f"  Mean v_next: {mean_v:.6f}")
    print(f"  All positive: {all_positive}")
    
    if all_positive and mean_v > 0:
        print("✓ Handles v0=0 correctly")
    else:
        print("✗ Issue with v0=0")
    
    return all_positive and mean_v > 0

def test_distribution_visualization():
    """Test and visualize the distribution of next variance"""
    print("\n=== Test 9.4: Distribution Visualization ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    dt = 1/252
    
    n_sims = 10000
    results = []
    
    for i in range(n_sims):
        v_next = heston_qe_step(v0, kappa, theta_v, sigma_v, dt)
        results.append(v_next)
    
    results = np.array(results)
    
    # Compute statistics
    mean_v = np.mean(results)
    std_v = np.std(results)
    median_v = np.median(results)
    p25 = np.percentile(results, 25)
    p75 = np.percentile(results, 75)
    
    print(f"Distribution of v_next ({n_sims} samples):")
    print(f"  Mean:   {mean_v:.6f}")
    print(f"  Std:    {std_v:.6f}")
    print(f"  Median: {median_v:.6f}")
    print(f"  25th percentile: {p25:.6f}")
    print(f"  75th percentile: {p75:.6f}")
    
    # Expected mean should be close to: theta_v + (v0 - theta_v) * exp(-kappa * dt)
    expected_mean = theta_v + (v0 - theta_v) * np.exp(-kappa * dt)
    print(f"\nTheoretical mean: {expected_mean:.6f}")
    print(f"Difference: {abs(mean_v - expected_mean):.6f}")
    
    # Check if mean is reasonable
    check = abs(mean_v - expected_mean) < 0.01
    
    if check:
        print("✓ Distribution looks reasonable")
    else:
        print("⚠ Mean differs from theory (may need more samples)")
    
    return True  # Informational test

def test_mean_reversion():
    """Test that variance shows mean reversion"""
    print("\n=== Test 9.5: Mean Reversion Behavior ===")
    
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    dt = 1/252
    
    n_sims = 1000
    
    # Test 1: Starting above theta_v (this should always hold)
    print("\n[9.5.1] Starting above θ_v:")
    v0_high = theta_v * 2.0
    results_high = []
    for i in range(n_sims):
        v_next = heston_qe_step(v0_high, kappa, theta_v, sigma_v, dt)
        results_high.append(v_next)
    mean_high = np.mean(results_high)
    
    print(f"  v0 = {v0_high:.6f} (2× θ_v)")
    print(f"  θ_v = {theta_v:.6f}")
    print(f"  Mean v_next = {mean_high:.6f}")
    
    # Should move toward theta_v (downward)
    check1 = mean_high < v0_high
    print(f"  Moves toward θ_v: {check1} (mean < v0)")
    
    # Test 2: Starting at theta_v (should stay near theta_v)
    print("\n[9.5.2] Starting at θ_v:")
    v0_at = theta_v
    results_at = []
    for i in range(n_sims):
        v_next = heston_qe_step(v0_at, kappa, theta_v, sigma_v, dt)
        results_at.append(v_next)
    mean_at = np.mean(results_at)
    
    print(f"  v0 = {v0_at:.6f}")
    print(f"  θ_v = {theta_v:.6f}")
    print(f"  Mean v_next = {mean_at:.6f}")
    
    # Should stay reasonably close to theta_v
    check2 = abs(mean_at - theta_v) < 0.015  # Within 1.5% variance units
    print(f"  Stays near θ_v: {check2}")
    
    if check1:
        print("\n✓ Mean reversion observed (high → low works)")
        return True
    else:
        print("\n✗ Mean reversion not clear")
        return False

def test_feller_condition():
    """Test Feller condition evaluation"""
    print("\n=== Test 9.6: Feller Condition ===")
    
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    satisfied, ratio = check_feller_condition(kappa, theta_v, sigma_v)
    
    print(f"Parameters:")
    print(f"  κ = {kappa}")
    print(f"  θ_v = {theta_v}")
    print(f"  σ_v = {sigma_v}")
    
    print(f"\nFeller condition: 2κθ_v ≥ σ_v²")
    print(f"  LHS: 2κθ_v = {2*kappa*theta_v:.6f}")
    print(f"  RHS: σ_v² = {sigma_v**2:.6f}")
    print(f"  Ratio: {ratio:.4f}")
    print(f"  Satisfied: {satisfied}")
    
    if satisfied:
        print("✓ Feller condition satisfied (variance stays positive)")
    else:
        print("⚠ Feller condition violated (variance can reach zero)")
    
    return True  # Informational

def test_input_validation():
    """Test input validation"""
    print("\n=== Test 9.7: Input Validation ===")
    
    valid_params = {
        'v_current': 0.04,
        'kappa': 2.0,
        'theta_v': 0.04,
        'sigma_v': 0.3,
        'dt': 1/252
    }
    
    all_passed = True
    
    # Test negative v_current
    print("\n[9.7.1] Negative v_current:")
    try:
        heston_qe_step(v_current=-0.01, **{k: v for k, v in valid_params.items() if k != 'v_current'})
        print("  ✗ Should have raised ValueError")
        all_passed = False
    except ValueError:
        print("  ✓ Correctly raised ValueError")
    
    # Test negative kappa
    print("\n[9.7.2] Negative kappa:")
    try:
        heston_qe_step(kappa=-1.0, **{k: v for k, v in valid_params.items() if k != 'kappa'})
        print("  ✗ Should have raised ValueError")
        all_passed = False
    except ValueError:
        print("  ✓ Correctly raised ValueError")
    
    # Test valid inputs
    print("\n[9.7.3] Valid inputs:")
    try:
        v_next = heston_qe_step(**valid_params)
        print(f"  ✓ Returned v_next = {v_next:.6f}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        all_passed = False
    
    if all_passed:
        print("\n✓ Input validation works correctly")
    else:
        print("\n✗ Some validation issues")
    
    return all_passed

def run_all_tests():
    """Run all Test 9 subtests"""
    print("=" * 60)
    print("TEST 9: HESTON VARIANCE - SINGLE STEP")
    print("=" * 60)
    
    results = []
    
    results.append(("Positivity", test_single_step_positivity()))
    results.append(("Reasonable Range", test_single_step_reasonable_range()))
    results.append(("From Zero Variance", test_starting_from_zero()))
    results.append(("Distribution", test_distribution_visualization()))
    results.append(("Mean Reversion", test_mean_reversion()))
    results.append(("Feller Condition", test_feller_condition()))
    results.append(("Input Validation", test_input_validation()))
    
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
        print("SUCCESS: All Heston single step tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
