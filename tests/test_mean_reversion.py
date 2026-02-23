"""
Test 14: Mean Reversion Utilities
Goal: Test MR helper functions
"""

import numpy as np
import sys
from models.mean_reversion import (
    compute_mu_t, 
    compute_half_life, 
    compute_mean_reversion_drift,
    estimate_theta_from_half_life
)
from config import MEAN_REVERSION_PARAMS

def test_mu_t_growth():
    """Test that μ_t grows at rate r"""
    print("\n=== Test 14.1: μ_t Growth Rate ===")
    
    mu_0 = 450.0
    r = 0.05
    
    # Test at different times
    times = [0.0, 0.25, 0.5, 1.0]
    
    print(f"μ_0 = {mu_0}, r = {r}")
    print(f"\n{'Time (years)':<15} {'μ_t':<15} {'Expected':<15}")
    print("-" * 50)
    
    all_passed = True
    for t in times:
        mu_t = compute_mu_t(mu_0, r, t)
        expected = mu_0 * np.exp(r * t)
        
        print(f"{t:<15.2f} {mu_t:<15.6f} {expected:<15.6f}")
        
        check = abs(mu_t - expected) < 1e-10
        if not check:
            all_passed = False
    
    if all_passed:
        print("\n✓ μ_t grows at rate r")
    else:
        print("\n✗ μ_t growth incorrect")
    
    return all_passed

def test_half_life():
    """Test half-life calculation"""
    print("\n=== Test 14.2: Half-Life Calculation ===")
    
    theta_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    print(f"{'θ (1/hour)':<15} {'Half-life (hours)':<20}")
    print("-" * 40)
    
    all_passed = True
    for theta in theta_values:
        half_life_dict = compute_half_life(theta)
        half_life = half_life_dict['hours']
        expected = np.log(2) / theta
        
        print(f"{theta:<15.1f} {half_life:<20.6f}")
        
        # Check formula: half-life (hours) = ln(2) / theta
        # But theta is in 1/year units, so need conversion
        expected_hours = np.log(2) / theta * (252 * 6.5)
        check = abs(half_life - expected_hours) < 1e-6
        if not check:
            all_passed = False
    
    # Special case: θ=5 → ~2.5 hours (as mentioned in spec)
    hl_dict_5 = compute_half_life(5.0)
    hl_5 = hl_dict_5['hours']
    
    print(f"\nθ=5 gives half-life ≈ {hl_5:.2f} hours")
    
    if all_passed:
        print("✓ Half-life calculation correct")
    else:
        print("✗ Half-life calculation incorrect")
    
    return all_passed

def test_mean_reversion_drift():
    """Test mean reversion drift direction"""
    print("\n=== Test 14.3: Mean Reversion Drift ===")
    
    mu_t = 450.0
    theta_mr = 5.0
    
    test_cases = [
        (455.0, "Above VWAP", "negative"),
        (450.0, "At VWAP", "zero"),
        (445.0, "Below VWAP", "positive"),
    ]
    
    print(f"μ_t = {mu_t}, θ = {theta_mr}")
    print(f"\n{'S_current':<12} {'Case':<15} {'Drift':<12} {'Expected':<12}")
    print("-" * 55)
    
    all_passed = True
    
    for S, case, expected in test_cases:
        drift = compute_mean_reversion_drift(S, mu_t, theta_mr)
        
        if expected == "negative":
            check = drift < 0
        elif expected == "zero":
            check = abs(drift) < 1e-10
        else:  # positive
            check = drift > 0
        
        status = "✓" if check else "✗"
        print(f"{S:<12.1f} {case:<15} {drift:<12.6f} {expected:<12} {status}")
        
        if not check:
            all_passed = False
    
    if all_passed:
        print("\n✓ Mean reversion drift directions correct")
    else:
        print("\n✗ Mean reversion drift incorrect")
    
    return all_passed

def test_theta_estimation():
    """Test theta estimation from half-life"""
    print("\n=== Test 14.4: Theta Estimation ===")
    
    half_lives = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"{'Half-life (hours)':<20} {'Estimated θ':<15} {'Check':<10}")
    print("-" * 50)
    
    all_passed = True
    for hl in half_lives:
        theta = estimate_theta_from_half_life(hl)
        # Verify: compute_half_life(theta)['hours'] should give back hl
        hl_dict_back = compute_half_life(theta)
        hl_back = hl_dict_back['hours']
        
        check = abs(hl_back - hl) < 1e-6
        status = "✓" if check else "✗"
        
        print(f"{hl:<20.2f} {theta:<15.6f} {status}")
        
        if not check:
            all_passed = False
    
    if all_passed:
        print("\n✓ Theta estimation correct")
    else:
        print("\n✗ Theta estimation incorrect")
    
    return all_passed

def test_mean_reversion_magnitude():
    """Test magnitude of mean reversion drift"""
    print("\n=== Test 14.5: Drift Magnitude ===")
    
    mu_t = 450.0
    theta_mr = 5.0
    
    # Test that drift increases with distance
    distances = [1, 5, 10, 20, 50]
    
    print(f"μ_t = {mu_t}, θ = {theta_mr}")
    print(f"\n{'Distance from μ':<20} {'|Drift|':<15}")
    print("-" * 40)
    
    prev_drift_mag = 0
    all_passed = True
    
    for dist in distances:
        S = mu_t + dist
        drift = compute_mean_reversion_drift(S, mu_t, theta_mr)
        drift_mag = abs(drift)
        
        print(f"{dist:<20.1f} {drift_mag:<15.6f}")
        
        # Drift magnitude should increase with distance
        if drift_mag <= prev_drift_mag and dist > distances[0]:
            all_passed = False
        
        prev_drift_mag = drift_mag
    
    if all_passed:
        print("\n✓ Drift magnitude increases with distance")
    else:
        print("\n✗ Drift magnitude behavior incorrect")
    
    return all_passed

def run_all_tests():
    """Run all Test 14 subtests"""
    print("=" * 60)
    print("TEST 14: MEAN REVERSION UTILITIES")
    print("=" * 60)
    
    results = []
    
    results.append(("μ_t Growth", test_mu_t_growth()))
    results.append(("Half-Life", test_half_life()))
    results.append(("MR Drift Direction", test_mean_reversion_drift()))
    results.append(("Theta Estimation", test_theta_estimation()))
    results.append(("Drift Magnitude", test_mean_reversion_magnitude()))
    
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
        print("SUCCESS: All mean reversion tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
