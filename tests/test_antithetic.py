"""
Test 7: Random Number Generation - Antithetic
Goal: Test antithetic pairs
"""

import numpy as np
import sys
from core.random_numbers import generate_antithetic_pairs

def test_antithetic_shape():
    """Test that antithetic pairs produce correct shape"""
    print("\n=== Test 7.1: Antithetic Shape ===")
    
    # Even number of paths
    print("\n[7.1.1] Even paths (1000):")
    np.random.seed(42)
    Z = np.random.standard_normal((1000, 100))
    Z_anti = generate_antithetic_pairs(Z)
    
    print(f"  Input:  {Z.shape}")
    print(f"  Output: {Z_anti.shape}")
    check1 = Z_anti.shape == (1000, 100)
    print(f"  Shape preserved: {check1}")
    
    # Odd number of paths
    print("\n[7.1.2] Odd paths (1001):")
    np.random.seed(42)
    Z = np.random.standard_normal((1001, 100))
    Z_anti = generate_antithetic_pairs(Z)
    
    print(f"  Input:  {Z.shape}")
    print(f"  Output: {Z_anti.shape}")
    check2 = Z_anti.shape == (1001, 100)
    print(f"  Shape preserved: {check2}")
    
    if check1 and check2:
        print("\n✓ Shape handling correct")
    else:
        print("\n✗ Shape handling incorrect")
    
    return check1 and check2

def test_antithetic_correlation():
    """Test that first and second halves are perfectly negatively correlated"""
    print("\n=== Test 7.2: Antithetic Correlation ===")
    
    n_paths = 1000  # Even number
    n_steps = 100
    
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths, n_steps))
    Z_anti = generate_antithetic_pairs(Z)
    
    # Split into halves
    half = n_paths // 2
    Z_first = Z_anti[:half, :]
    Z_second = Z_anti[half:2*half, :]
    
    print(f"First half shape:  {Z_first.shape}")
    print(f"Second half shape: {Z_second.shape}")
    
    # Check: Z_second ≈ -Z_first
    print(f"\nChecking Z_second ≈ -Z_first:")
    difference = Z_second + Z_first
    max_diff = np.max(np.abs(difference))
    print(f"  Max |Z_second + Z_first|: {max_diff:.2e}")
    
    check1 = max_diff < 1e-10
    
    # Compute correlation
    corr_values = []
    for col in range(n_steps):
        corr = np.corrcoef(Z_first[:, col], Z_second[:, col])[0, 1]
        corr_values.append(corr)
    
    mean_corr = np.mean(corr_values)
    min_corr = np.min(corr_values)
    max_corr = np.max(corr_values)
    
    print(f"\nCorrelation statistics:")
    print(f"  Mean: {mean_corr:.10f}")
    print(f"  Min:  {min_corr:.10f}")
    print(f"  Max:  {max_corr:.10f}")
    
    # Should be very close to -1.0
    check2 = -1.001 <= mean_corr <= -0.999
    
    if check1 and check2:
        print("✓ Antithetic correlation is -1.0")
    else:
        print("✗ Antithetic correlation incorrect")
    
    return check1 and check2

def test_odd_paths_handling():
    """Test how odd number of paths is handled"""
    print("\n=== Test 7.3: Odd Paths Handling ===")
    
    n_paths = 1001  # Odd
    n_steps = 100
    
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths, n_steps))
    Z_anti = generate_antithetic_pairs(Z)
    
    half = n_paths // 2  # 500
    
    # First 500 and next 500 should be antithetic
    Z_first = Z_anti[:half, :]
    Z_second = Z_anti[half:2*half, :]
    
    difference = Z_second + Z_first
    max_diff = np.max(np.abs(difference))
    
    print(f"First {half} and second {half} are antithetic:")
    print(f"  Max |Z_second + Z_first|: {max_diff:.2e}")
    
    check1 = max_diff < 1e-10
    
    # The 1001st path should be independent
    Z_extra = Z_anti[-1, :]
    print(f"\nExtra path (1001st) added:")
    print(f"  Shape: {Z_extra.shape}")
    print(f"  Not equal to -first path: {not np.allclose(Z_extra, -Z_anti[0, :])}")
    
    check2 = not np.allclose(Z_extra, -Z_anti[0, :], atol=1e-5)
    
    if check1 and check2:
        print("✓ Odd paths handled correctly")
    else:
        print("✗ Odd paths handling incorrect")
    
    return check1 and check2

def test_moments_preserved():
    """Test that antithetic pairs preserve mean and variance"""
    print("\n=== Test 7.4: Moments Preserved ===")
    
    n_paths = 1000
    n_steps = 100
    
    np.random.seed(42)
    Z_original = np.random.standard_normal((n_paths, n_steps))
    
    # Apply moment matching to original first
    from core.random_numbers import apply_moment_matching
    Z_original = apply_moment_matching(Z_original)
    
    mean_before = np.mean(Z_original)
    std_before = np.std(Z_original, ddof=0)
    
    # Apply antithetic
    Z_anti = generate_antithetic_pairs(Z_original)
    
    mean_after = np.mean(Z_anti)
    std_after = np.std(Z_anti, ddof=0)
    
    print(f"Before antithetic:")
    print(f"  Mean: {mean_before:.10e}")
    print(f"  Std:  {std_before:.10f}")
    
    print(f"\nAfter antithetic:")
    print(f"  Mean: {mean_after:.10e}")
    print(f"  Std:  {std_after:.10f}")
    
    # Mean should be exactly 0 (sum of antithetic pairs = 0)
    check_mean = abs(mean_after) < 1e-10
    
    # Variance should be preserved or very close
    check_std = abs(std_after - 1.0) < 0.01
    
    print(f"\nMean ≈ 0: {check_mean}")
    print(f"Std ≈ 1: {check_std}")
    
    if check_mean and check_std:
        print("✓ Moments preserved")
    else:
        print("✗ Moments not preserved")
    
    return check_mean and check_std

def test_variance_reduction_effect():
    """Demonstrate variance reduction effect"""
    print("\n=== Test 7.5: Variance Reduction Effect ===")
    
    n_paths = 1000
    n_steps = 100
    n_simulations = 100
    
    # Function to price a simple option using paths
    def price_option(Z):
        # Simple geometric Brownian motion
        S0 = 100
        r = 0.05
        sigma = 0.2
        T = 0.25
        K = 100
        
        dt = T / n_steps
        S = S0 * np.exp(np.cumsum(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
        ))
        S_T = S[:, -1]
        payoffs = np.maximum(S_T - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
    
    # Run simulations without antithetic
    prices_standard = []
    for i in range(n_simulations):
        np.random.seed(i)
        Z = np.random.standard_normal((n_paths, n_steps))
        price = price_option(Z)
        prices_standard.append(price)
    
    # Run simulations with antithetic
    prices_anti = []
    for i in range(n_simulations):
        np.random.seed(i)
        Z = np.random.standard_normal((n_paths, n_steps))
        Z_anti = generate_antithetic_pairs(Z)
        price = price_option(Z_anti)
        prices_anti.append(price)
    
    std_standard = np.std(prices_standard)
    std_anti = np.std(prices_anti)
    
    variance_reduction = (std_standard**2) / (std_anti**2)
    
    print(f"\nStd of prices (standard): {std_standard:.6f}")
    print(f"Std of prices (antithetic): {std_anti:.6f}")
    print(f"Variance reduction factor: {variance_reduction:.2f}×")
    
    # Antithetic should have lower or similar variance
    check = std_anti <= std_standard * 1.1  # Allow 10% margin
    
    if variance_reduction >= 1.0:
        print(f"✓ Antithetic reduces variance by {variance_reduction:.2f}×")
    else:
        print(f"ℹ Variance reduction: {variance_reduction:.2f}× (expected ≥ 1)")
    
    return check

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 7.6: Edge Cases ===")
    
    all_passed = True
    
    # Small even
    print("\n[7.6.1] Small even (10×10):")
    np.random.seed(42)
    Z = np.random.standard_normal((10, 10))
    Z_anti = generate_antithetic_pairs(Z)
    check1 = Z_anti.shape == (10, 10)
    corr = np.corrcoef(Z_anti[0, :], Z_anti[5, :])[0, 1]
    print(f"  Shape: {Z_anti.shape}, Correlation: {corr:.4f}")
    check1 = check1 and corr < -0.99
    all_passed = all_passed and check1
    
    # Small odd
    print("\n[7.6.2] Small odd (11×10):")
    np.random.seed(42)
    Z = np.random.standard_normal((11, 10))
    Z_anti = generate_antithetic_pairs(Z)
    check2 = Z_anti.shape == (11, 10)
    print(f"  Shape: {Z_anti.shape}, Valid: {check2}")
    all_passed = all_passed and check2
    
    # Single step
    print("\n[7.6.3] Single step (1000×1):")
    np.random.seed(42)
    Z = np.random.standard_normal((1000, 1))
    Z_anti = generate_antithetic_pairs(Z)
    check3 = Z_anti.shape == (1000, 1)
    corr = np.corrcoef(Z_anti[:500, 0], Z_anti[500:1000, 0])[0, 1]
    print(f"  Shape: {Z_anti.shape}, Correlation: {corr:.4f}")
    check3 = check3 and corr < -0.99
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ Edge cases passed")
    else:
        print("\n✗ Some edge cases failed")
    
    return all_passed

def run_all_tests():
    """Run all Test 7 subtests"""
    print("=" * 60)
    print("TEST 7: RANDOM NUMBER GENERATION - ANTITHETIC")
    print("=" * 60)
    
    results = []
    
    results.append(("Shape", test_antithetic_shape()))
    results.append(("Correlation = -1", test_antithetic_correlation()))
    results.append(("Odd Paths Handling", test_odd_paths_handling()))
    results.append(("Moments Preserved", test_moments_preserved()))
    results.append(("Variance Reduction", test_variance_reduction_effect()))
    results.append(("Edge Cases", test_edge_cases()))
    
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
        print("SUCCESS: All antithetic variates tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
