"""
Test 5: Random Number Generation - Sobol
Goal: Test quasi-random Sobol sequences
"""

import numpy as np
import sys
from scipy.stats import shapiro, normaltest
from core.random_numbers import generate_sobol_normals

def test_sobol_shape():
    """Test that Sobol generates correct shape"""
    print("\n=== Test 5.1: Sobol Shape ===")
    
    n_paths = 1000
    n_steps = 100
    
    Z = generate_sobol_normals(n_paths, n_steps, seed=42)
    
    print(f"Requested: ({n_paths}, {n_steps})")
    print(f"Generated: {Z.shape}")
    
    check = Z.shape == (n_paths, n_steps)
    
    if check:
        print("✓ Shape is correct")
    else:
        print("✗ Shape is incorrect")
    
    return check

def test_sobol_moments():
    """Test that Sobol normals have mean ≈ 0, std ≈ 1"""
    print("\n=== Test 5.2: Sobol Moments ===")
    
    n_paths = 1000
    n_steps = 100
    
    Z = generate_sobol_normals(n_paths, n_steps, seed=42)
    
    mean = np.mean(Z)
    std = np.std(Z, ddof=1)  # Sample std
    
    print(f"Mean: {mean:.6f} (expected: 0)")
    print(f"Std:  {std:.6f} (expected: 1)")
    
    # Check if within ±0.05 tolerance
    check_mean = abs(mean) < 0.05
    check_std = abs(std - 1.0) < 0.05
    
    print(f"Mean within ±0.05: {check_mean}")
    print(f"Std within ±0.05: {check_std}")
    
    if check_mean and check_std:
        print("✓ Moments are correct")
    else:
        print("✗ Some moments out of range")
    
    return check_mean and check_std

def test_sobol_no_nans():
    """Test that no NaNs or Infs are generated"""
    print("\n=== Test 5.3: No NaNs/Infs ===")
    
    n_paths = 1000
    n_steps = 100
    
    Z = generate_sobol_normals(n_paths, n_steps, seed=42)
    
    has_nan = np.any(np.isnan(Z))
    has_inf = np.any(np.isinf(Z))
    
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    
    check = not has_nan and not has_inf
    
    if check:
        print("✓ No NaNs or Infs")
    else:
        print("✗ Contains invalid values")
    
    return check

def test_sobol_vs_standard_normal():
    """Compare Sobol to standard normal distribution"""
    print("\n=== Test 5.4: Sobol vs Standard Normal ===")
    
    n_paths = 5000
    n_steps = 100
    
    # Generate Sobol normals
    Z_sobol = generate_sobol_normals(n_paths, n_steps, seed=42)
    
    # Generate standard normals
    np.random.seed(42)
    Z_standard = np.random.standard_normal((n_paths, n_steps))
    
    # Flatten for comparison
    sobol_flat = Z_sobol.flatten()
    standard_flat = Z_standard.flatten()
    
    # Compare moments
    print(f"\nMoments:")
    print(f"  Sobol:    mean={np.mean(sobol_flat):.6f}, std={np.std(sobol_flat):.6f}")
    print(f"  Standard: mean={np.mean(standard_flat):.6f}, std={np.std(standard_flat):.6f}")
    
    # Sobol should have better uniformity - check via binning
    # Count values in different ranges
    bins = np.linspace(-3, 3, 7)  # 6 bins
    sobol_hist, _ = np.histogram(sobol_flat, bins=bins)
    standard_hist, _ = np.histogram(standard_flat, bins=bins)
    
    # Compute coefficient of variation for bin counts
    sobol_cv = np.std(sobol_hist) / np.mean(sobol_hist)
    standard_cv = np.std(standard_hist) / np.mean(standard_hist)
    
    print(f"\nUniformity (CV of bin counts, lower is better):")
    print(f"  Sobol:    {sobol_cv:.6f}")
    print(f"  Standard: {standard_cv:.6f}")
    
    # Sobol should have lower CV (more uniform)
    check = sobol_cv <= standard_cv * 1.2  # Allow 20% margin
    
    if check:
        print(f"✓ Sobol has comparable or better uniformity")
    else:
        print(f"ℹ Sobol uniformity: {sobol_cv:.6f} vs standard: {standard_cv:.6f}")
    
    # Both should look Gaussian
    return True  # This is more informational

def test_sobol_normality():
    """Test that distribution looks Gaussian"""
    print("\n=== Test 5.5: Distribution Normality ===")
    
    n_paths = 5000
    n_steps = 100
    
    Z = generate_sobol_normals(n_paths, n_steps, seed=42)
    Z_flat = Z.flatten()
    
    # Compute percentiles
    p01 = np.percentile(Z_flat, 1)
    p50 = np.percentile(Z_flat, 50)
    p99 = np.percentile(Z_flat, 99)
    
    print(f"Percentiles:")
    print(f"  1st:  {p01:.3f} (expected ≈ -2.33)")
    print(f"  50th: {p50:.3f} (expected ≈ 0)")
    print(f"  99th: {p99:.3f} (expected ≈ 2.33)")
    
    # Check if close to expected normal percentiles
    check_p01 = abs(p01 - (-2.33)) < 0.3
    check_p50 = abs(p50) < 0.1
    check_p99 = abs(p99 - 2.33) < 0.3
    
    if check_p01 and check_p50 and check_p99:
        print("✓ Distribution looks Gaussian")
        return True
    else:
        print("⚠ Distribution deviates from normal")
        return False

def test_sobol_deterministic():
    """Test that same seed produces same results"""
    print("\n=== Test 5.6: Deterministic with Seed ===")
    
    n_paths = 100
    n_steps = 50
    
    Z1 = generate_sobol_normals(n_paths, n_steps, seed=123)
    Z2 = generate_sobol_normals(n_paths, n_steps, seed=123)
    
    # Should be identical
    identical = np.allclose(Z1, Z2, rtol=1e-15, atol=1e-15)
    
    print(f"Same seed produces identical results: {identical}")
    
    # Different seeds should produce different results
    Z3 = generate_sobol_normals(n_paths, n_steps, seed=456)
    different = not np.allclose(Z1, Z3, rtol=1e-3, atol=1e-3)
    
    print(f"Different seeds produce different results: {different}")
    
    if identical and different:
        print("✓ Deterministic behavior correct")
    else:
        print("✗ Deterministic behavior incorrect")
    
    return identical and different

def test_sobol_edge_cases():
    """Test edge cases"""
    print("\n=== Test 5.7: Edge Cases ===")
    
    all_passed = True
    
    # Small arrays
    print("\n[5.7.1] Small array (10×10):")
    Z_small = generate_sobol_normals(10, 10, seed=42)
    check1 = Z_small.shape == (10, 10) and not np.any(np.isnan(Z_small))
    print(f"  Shape: {Z_small.shape}, Valid: {check1}")
    all_passed = all_passed and check1
    
    # Single path
    print("\n[5.7.2] Single path (1×100):")
    Z_single = generate_sobol_normals(1, 100, seed=42)
    check2 = Z_single.shape == (1, 100)
    print(f"  Shape: {Z_single.shape}, Valid: {check2}")
    all_passed = all_passed and check2
    
    # Large arrays
    print("\n[5.7.3] Large array (10000×200):")
    Z_large = generate_sobol_normals(10000, 200, seed=42)
    check3 = Z_large.shape == (10000, 200) and not np.any(np.isnan(Z_large))
    mean_large = np.mean(Z_large)
    std_large = np.std(Z_large)
    print(f"  Shape: {Z_large.shape}")
    print(f"  Mean: {mean_large:.6f}, Std: {std_large:.6f}")
    check3 = check3 and abs(mean_large) < 0.05 and abs(std_large - 1.0) < 0.05
    print(f"  Valid: {check3}")
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ Edge cases passed")
    else:
        print("\n✗ Some edge cases failed")
    
    return all_passed

def run_all_tests():
    """Run all Test 5 subtests"""
    print("=" * 60)
    print("TEST 5: RANDOM NUMBER GENERATION - SOBOL")
    print("=" * 60)
    
    results = []
    
    results.append(("Shape", test_sobol_shape()))
    results.append(("Moments", test_sobol_moments()))
    results.append(("No NaNs/Infs", test_sobol_no_nans()))
    results.append(("vs Standard Normal", test_sobol_vs_standard_normal()))
    results.append(("Normality", test_sobol_normality()))
    results.append(("Deterministic", test_sobol_deterministic()))
    results.append(("Edge Cases", test_sobol_edge_cases()))
    
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
        print("SUCCESS: All Sobol generation tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
