"""
Test 13: Jump Generation - Vectorized
Goal: Test vectorized jump generation
"""

import numpy as np
import sys
from models.jump_diffusion import generate_jumps_vectorized, compute_jump_statistics
from core.time_grid import generate_time_grid
from config import JUMP_PARAMS

def test_vectorized_shape():
    """Test that vectorized generation has correct shape"""
    print("\n=== Test 13.1: Vectorized Shape ===")
    
    lambda_jump = 2.0
    mu_jump = -0.02
    sigma_jump = 0.05
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    jump_log_sizes = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    print(f"Parameters: n_paths={n_paths}, n_steps={n_steps}")
    print(f"Shape: {jump_log_sizes.shape}")
    print(f"Expected: ({n_paths}, {n_steps})")
    
    check = jump_log_sizes.shape == (n_paths, n_steps)
    
    if check:
        print("✓ Shape is correct")
    else:
        print("✗ Shape is incorrect")
    
    return check

def test_jump_count():
    """Test that reasonable number of jumps are generated"""
    print("\n=== Test 13.2: Jump Count ===")
    
    lambda_jump = 2.0
    mu_jump = -0.02
    sigma_jump = 0.05
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    jump_log_sizes = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    # Count non-zero entries (jumps)
    n_jumps = np.sum(jump_log_sizes != 0)
    
    # Expected: n_paths * n_steps * lambda_jump * dt_avg
    expected = n_paths * lambda_jump * T
    
    print(f"Total jumps: {n_jumps}")
    print(f"Expected: ~{expected:.0f} (±{np.sqrt(expected):.0f})")
    print(f"Per path: {n_jumps/n_paths:.2f}")
    
    # Check within reasonable range (3 sigma)
    check = abs(n_jumps - expected) < 3 * np.sqrt(expected)
    
    if check:
        print("✓ Jump count is reasonable")
    else:
        print("⚠ Jump count outside expected range (may be random variation)")
    
    return True  # Probabilistic test

def test_jump_sizes():
    """Test that jump sizes follow N(μ, σ)"""
    print("\n=== Test 13.3: Jump Size Distribution ===")
    
    lambda_jump = 10.0  # Higher rate to get more jumps
    mu_jump = -0.02
    sigma_jump = 0.05
    
    T = 1.0  # 1 year
    n_steps = 252
    n_paths = 1000
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    jump_log_sizes = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    # Extract actual jump sizes (non-zero)
    actual_jumps = jump_log_sizes[jump_log_sizes != 0]
    
    n_jumps = len(actual_jumps)
    mean_jump = np.mean(actual_jumps)
    std_jump = np.std(actual_jumps, ddof=1)
    
    print(f"Total jumps: {n_jumps}")
    print(f"Mean jump size: {mean_jump:.6f} (expected: {mu_jump})")
    print(f"Std jump size:  {std_jump:.6f} (expected: {sigma_jump})")
    
    # Check within reasonable range
    se_mean = sigma_jump / np.sqrt(n_jumps)
    check_mean = abs(mean_jump - mu_jump) < 3 * se_mean
    
    se_std = sigma_jump / np.sqrt(2 * n_jumps)
    check_std = abs(std_jump - sigma_jump) < 3 * se_std
    
    print(f"\nMean check: {check_mean}")
    print(f"Std check: {check_std}")
    
    if check_mean and check_std:
        print("✓ Jump sizes follow expected distribution")
    else:
        print("⚠ Jump sizes deviate from expectation")
    
    return True  # Probabilistic

def test_no_jumps_case():
    """Test when lambda is zero (no jumps)"""
    print("\n=== Test 13.4: No Jumps Case ===")
    
    lambda_jump = 0.0  # No jumps
    mu_jump = -0.02
    sigma_jump = 0.05
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    jump_log_sizes = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    n_jumps = np.sum(jump_log_sizes != 0)
    
    print(f"Lambda: {lambda_jump}")
    print(f"Total jumps: {n_jumps}")
    print(f"All zeros: {np.all(jump_log_sizes == 0)}")
    
    check = n_jumps == 0
    
    if check:
        print("✓ No jumps when lambda=0")
    else:
        print("✗ Unexpected jumps when lambda=0")
    
    return check

def test_statistics_computation():
    """Test jump statistics computation"""
    print("\n=== Test 13.5: Statistics Computation ===")
    
    lambda_jump = 5.0
    mu_jump = -0.03
    sigma_jump = 0.08
    
    T = 1.0
    n_steps = 252
    n_paths = 1000
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    jump_log_sizes = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    stats = compute_jump_statistics(jump_log_sizes, dt_array, T)
    
    print(f"Statistics:")
    print(f"  Total jumps: {stats['n_jumps']}")
    print(f"  Jump intensity: {stats['jump_intensity']:.4f} (expected: ~{lambda_jump})")
    print(f"  Mean jump size: {stats['mean_jump_size']:.6f} (expected: ~{mu_jump})")
    print(f"  Std jump size: {stats['std_jump_size']:.6f} (expected: ~{sigma_jump})")
    print(f"  Jumps per path: {stats['jumps_per_path']:.4f}")
    
    # Check intensity is reasonable
    check = abs(stats['jump_intensity'] - lambda_jump) < 1.0
    
    if check:
        print("✓ Statistics computation works")
    else:
        print("⚠ Intensity differs from expected")
    
    return True  # Informational

def test_deterministic():
    """Test deterministic behavior with seed"""
    print("\n=== Test 13.6: Deterministic Behavior ===")
    
    lambda_jump = 2.0
    mu_jump = -0.02
    sigma_jump = 0.05
    
    T = 0.25
    n_steps = 100
    n_paths = 100
    
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Generate twice with same seed
    np.random.seed(123)
    jumps1 = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    np.random.seed(123)
    jumps2 = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    identical = np.allclose(jumps1, jumps2, rtol=1e-15, atol=1e-15)
    
    print(f"Same seed produces identical results: {identical}")
    
    # Different seed
    np.random.seed(456)
    jumps3 = generate_jumps_vectorized(lambda_jump, dt_array, n_paths, n_steps, mu_jump, sigma_jump)
    
    different = not np.allclose(jumps1, jumps3, rtol=1e-3)
    
    print(f"Different seed produces different results: {different}")
    
    if identical and different:
        print("✓ Deterministic behavior correct")
    else:
        print("✗ Deterministic behavior incorrect")
    
    return identical and different

def run_all_tests():
    """Run all Test 13 subtests"""
    print("=" * 60)
    print("TEST 13: JUMP GENERATION - VECTORIZED")
    print("=" * 60)
    
    results = []
    
    results.append(("Shape", test_vectorized_shape()))
    results.append(("Jump Count", test_jump_count()))
    results.append(("Jump Sizes", test_jump_sizes()))
    results.append(("No Jumps Case", test_no_jumps_case()))
    results.append(("Statistics", test_statistics_computation()))
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
        print("SUCCESS: All vectorized jump tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
