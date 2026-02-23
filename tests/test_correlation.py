"""
Test 8: Random Number Generation - Correlation
Goal: Generate correlated normals (Z1, Z2)
"""

import numpy as np
import sys
from core.random_numbers import generate_correlated_normals

def test_correlation_target():
    """Test that achieved correlation matches target"""
    print("\n=== Test 8.1: Correlation Target Matching ===")
    
    test_cases = [
        -0.9,
        -0.7,
        -0.5,
        0.0,
        0.3,
        0.5,
        0.7,
        0.9
    ]
    
    n_paths = 5000
    n_steps = 100
    
    all_passed = True
    
    print(f"\nTesting with {n_paths} paths, {n_steps} steps:")
    print(f"{'Target ρ':<12} {'Achieved ρ':<15} {'Difference':<12} {'Status'}")
    print("-" * 55)
    
    for rho_target in test_cases:
        Z1, Z2 = generate_correlated_normals(
            n_paths=n_paths,
            n_steps=n_steps,
            rho=rho_target,
            use_sobol=False,  # Use standard for reproducibility
            use_moment_matching=False,
            use_antithetic=False,
            seed=42
        )
        
        # Flatten and compute correlation
        Z1_flat = Z1.flatten()
        Z2_flat = Z2.flatten()
        
        rho_achieved = np.corrcoef(Z1_flat, Z2_flat)[0, 1]
        diff = abs(rho_achieved - rho_target)
        
        # Check within ±0.05 tolerance
        passed = diff < 0.05
        status = "✓" if passed else "✗"
        
        print(f"{rho_target:>10.2f}   {rho_achieved:>12.6f}   {diff:>10.6f}   {status}")
        
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✓ All correlations within ±0.05")
    else:
        print("\n✗ Some correlations outside tolerance")
    
    return all_passed

def test_shapes_match():
    """Test that Z1 and Z2 have matching shapes"""
    print("\n=== Test 8.2: Shapes Match ===")
    
    n_paths = 1000
    n_steps = 100
    
    Z1, Z2 = generate_correlated_normals(
        n_paths=n_paths,
        n_steps=n_steps,
        rho=-0.5,
        seed=42
    )
    
    print(f"Z1 shape: {Z1.shape}")
    print(f"Z2 shape: {Z2.shape}")
    
    check = Z1.shape == Z2.shape == (n_paths, n_steps)
    
    if check:
        print("✓ Shapes match and are correct")
    else:
        print("✗ Shapes don't match")
    
    return check

def test_correlation_formula():
    """Test that the correlation formula is correctly applied"""
    print("\n=== Test 8.3: Correlation Formula ===")
    
    # Manual implementation
    n_paths = 5000
    n_steps = 100
    rho = 0.6
    
    np.random.seed(42)
    Z1_manual = np.random.standard_normal((n_paths, n_steps))
    Z_indep = np.random.standard_normal((n_paths, n_steps))
    Z2_manual = rho * Z1_manual + np.sqrt(1 - rho**2) * Z_indep
    
    # Using function
    Z1_func, Z2_func = generate_correlated_normals(
        n_paths=n_paths,
        n_steps=n_steps,
        rho=rho,
        use_sobol=False,
        use_moment_matching=False,
        use_antithetic=False,
        seed=42
    )
    
    # Check correlation
    corr_manual = np.corrcoef(Z1_manual.flatten(), Z2_manual.flatten())[0, 1]
    corr_func = np.corrcoef(Z1_func.flatten(), Z2_func.flatten())[0, 1]
    
    print(f"Manual correlation: {corr_manual:.6f}")
    print(f"Function correlation: {corr_func:.6f}")
    print(f"Target: {rho:.6f}")
    
    # Both should be close to target
    check1 = abs(corr_manual - rho) < 0.05
    check2 = abs(corr_func - rho) < 0.05
    
    if check1 and check2:
        print("✓ Correlation formula correct")
    else:
        print("✗ Correlation formula incorrect")
    
    return check1 and check2

def test_variance_reduction_options():
    """Test with different variance reduction options"""
    print("\n=== Test 8.4: Variance Reduction Options ===")
    
    n_paths = 5000
    n_steps = 100
    rho = -0.7
    
    # Test all combinations
    configs = [
        ("No VR", False, False, False),
        ("Sobol only", True, False, False),
        ("Moment only", False, True, False),
        ("Antithetic only", False, False, True),
        ("All VR", True, True, True),
    ]
    
    all_passed = True
    
    print(f"\nTesting configurations (target ρ={rho}):")
    print(f"{'Config':<20} {'Z1 Mean':<12} {'Z1 Std':<12} {'Achieved ρ':<15}")
    print("-" * 65)
    
    for name, sobol, moment, anti in configs:
        Z1, Z2 = generate_correlated_normals(
            n_paths=n_paths,
            n_steps=n_steps,
            rho=rho,
            use_sobol=sobol,
            use_moment_matching=moment,
            use_antithetic=anti,
            seed=42
        )
        
        mean = np.mean(Z1)
        std = np.std(Z1)
        corr = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
        
        print(f"{name:<20} {mean:>10.6f}  {std:>10.6f}  {corr:>12.6f}")
        
        # Check correlation is close
        check = abs(corr - rho) < 0.1  # Wider tolerance for different configs
        all_passed = all_passed and check
    
    if all_passed:
        print("\n✓ All configurations work")
    else:
        print("\n✗ Some configurations failed")
    
    return all_passed

def test_extreme_correlations():
    """Test extreme correlation values"""
    print("\n=== Test 8.5: Extreme Correlations ===")
    
    n_paths = 5000
    n_steps = 100
    
    # Near -1
    print("\n[8.5.1] Near -1 (ρ=-0.99):")
    Z1, Z2 = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=-0.99,
        use_sobol=False, use_moment_matching=False,
        use_antithetic=False, seed=42
    )
    corr_neg = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
    print(f"  Achieved: {corr_neg:.6f}")
    check1 = abs(corr_neg - (-0.99)) < 0.05
    
    # Near +1
    print("\n[8.5.2] Near +1 (ρ=0.99):")
    Z1, Z2 = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=0.99,
        use_sobol=False, use_moment_matching=False,
        use_antithetic=False, seed=42
    )
    corr_pos = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
    print(f"  Achieved: {corr_pos:.6f}")
    check2 = abs(corr_pos - 0.99) < 0.05
    
    # Zero correlation
    print("\n[8.5.3] Zero correlation (ρ=0.0):")
    Z1, Z2 = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=0.0,
        use_sobol=False, use_moment_matching=False,
        use_antithetic=False, seed=42
    )
    corr_zero = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
    print(f"  Achieved: {corr_zero:.6f}")
    check3 = abs(corr_zero) < 0.05
    
    if check1 and check2 and check3:
        print("\n✓ Extreme correlations work")
    else:
        print("\n✗ Some extreme correlations failed")
    
    return check1 and check2 and check3

def test_invalid_correlation():
    """Test that invalid correlations raise errors"""
    print("\n=== Test 8.6: Invalid Correlation Values ===")
    
    invalid_values = [-1.5, -1.0, 1.0, 1.5, 2.0]
    
    all_passed = True
    
    for rho in invalid_values:
        try:
            Z1, Z2 = generate_correlated_normals(
                n_paths=100, n_steps=10, rho=rho, seed=42
            )
            print(f"  ρ={rho}: ✗ Should have raised ValueError")
            all_passed = False
        except ValueError as e:
            print(f"  ρ={rho}: ✓ Correctly raised ValueError")
    
    if all_passed:
        print("\n✓ Invalid values properly rejected")
    else:
        print("\n✗ Some invalid values not rejected")
    
    return all_passed

def test_deterministic():
    """Test that same seed produces same results"""
    print("\n=== Test 8.7: Deterministic Behavior ===")
    
    n_paths = 500
    n_steps = 50
    rho = -0.6
    
    # Generate twice with same seed
    Z1_a, Z2_a = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=rho,
        use_sobol=False, seed=123
    )
    
    Z1_b, Z2_b = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=rho,
        use_sobol=False, seed=123
    )
    
    # Should be identical
    check1 = np.allclose(Z1_a, Z1_b, rtol=1e-15, atol=1e-15)
    check2 = np.allclose(Z2_a, Z2_b, rtol=1e-15, atol=1e-15)
    
    print(f"Z1 identical: {check1}")
    print(f"Z2 identical: {check2}")
    
    # Different seed should give different results
    Z1_c, Z2_c = generate_correlated_normals(
        n_paths=n_paths, n_steps=n_steps, rho=rho,
        use_sobol=False, seed=456
    )
    
    check3 = not np.allclose(Z1_a, Z1_c, rtol=1e-3, atol=1e-3)
    
    print(f"Different seed gives different results: {check3}")
    
    if check1 and check2 and check3:
        print("✓ Deterministic behavior correct")
    else:
        print("✗ Deterministic behavior incorrect")
    
    return check1 and check2 and check3

def run_all_tests():
    """Run all Test 8 subtests"""
    print("=" * 60)
    print("TEST 8: RANDOM NUMBER GENERATION - CORRELATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Correlation Target", test_correlation_target()))
    results.append(("Shapes Match", test_shapes_match()))
    results.append(("Correlation Formula", test_correlation_formula()))
    results.append(("VR Options", test_variance_reduction_options()))
    results.append(("Extreme Correlations", test_extreme_correlations()))
    results.append(("Invalid Values", test_invalid_correlation()))
    results.append(("Deterministic", test_deterministic()))
    
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
        print("SUCCESS: All correlated normals tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
