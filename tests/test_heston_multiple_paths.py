"""
Test 11: Heston Variance - Multiple Paths
Goal: Test vectorized simulation
"""

import numpy as np
import sys
from models.heston import simulate_variance_paths
from core.time_grid import generate_time_grid
from config import HESTON_PARAMS

def test_multiple_paths_shape():
    """Test that multiple paths have correct shape"""
    print("\n=== Test 11.1: Multiple Paths Shape ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    print(f"Parameters: n_paths={n_paths}, n_steps={n_steps}")
    print(f"Shape: {v_paths.shape}")
    print(f"Expected: ({n_paths}, {n_steps + 1})")
    
    check = v_paths.shape == (n_paths, n_steps + 1)
    
    if check:
        print("✓ Shape is correct")
    else:
        print("✗ Shape is incorrect")
    
    return check

def test_all_paths_start_at_v0():
    """Test that all paths start at v0"""
    print("\n=== Test 11.2: All Paths Start at v0 ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    # All paths should start at v0
    initial_values = v_paths[:, 0]
    
    all_match = np.all(initial_values == v0)
    
    print(f"All paths start at v0={v0}: {all_match}")
    print(f"Min initial: {np.min(initial_values):.6f}")
    print(f"Max initial: {np.max(initial_values):.6f}")
    
    if all_match:
        print("✓ All paths start correctly")
    else:
        print("✗ Some paths don't start at v0")
    
    return all_match

def test_all_values_positive():
    """Test that all variance values are positive"""
    print("\n=== Test 11.3: All Values Positive ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    all_positive = np.all(v_paths >= 0)
    
    min_value = np.min(v_paths)
    max_value = np.max(v_paths)
    
    print(f"Total elements: {v_paths.size}")
    print(f"All positive: {all_positive}")
    print(f"Min value: {min_value:.6f}")
    print(f"Max value: {max_value:.6f}")
    
    if all_positive:
        print("✓ All variance values are positive")
    else:
        negative_count = np.sum(v_paths < 0)
        print(f"✗ Found {negative_count} negative values")
    
    return all_positive

def test_paths_are_different():
    """Test that paths are not identical (randomness working)"""
    print("\n=== Test 11.4: Path Diversity ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    # Check that paths are different
    # Compare first path to others
    path_0 = v_paths[0, :]
    
    identical_count = 0
    for i in range(1, min(100, n_paths)):
        if np.allclose(path_0, v_paths[i, :], rtol=1e-10):
            identical_count += 1
    
    print(f"Compared first path to {min(99, n_paths-1)} others")
    print(f"Identical paths found: {identical_count}")
    
    # Also check variance across paths at final time
    final_values = v_paths[:, -1]
    std_final = np.std(final_values)
    
    print(f"\nStd of final variances: {std_final:.6f}")
    
    check = (identical_count == 0) and (std_final > 0.0001)
    
    if check:
        print("✓ Paths show diversity (randomness working)")
    else:
        print("✗ Paths appear too similar")
    
    return check

def test_statistics_across_paths():
    """Test aggregate statistics across paths"""
    print("\n=== Test 11.5: Aggregate Statistics ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 10000  # More paths for better statistics
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    # Compute statistics at final time
    final_values = v_paths[:, -1]
    
    mean_final = np.mean(final_values)
    std_final = np.std(final_values)
    median_final = np.median(final_values)
    
    print(f"Statistics from {n_paths} paths:")
    print(f"  Mean:   {mean_final:.6f}")
    print(f"  Std:    {std_final:.6f}")
    print(f"  Median: {median_final:.6f}")
    print(f"  Min:    {np.min(final_values):.6f}")
    print(f"  Max:    {np.max(final_values):.6f}")
    
    # All should be reasonable
    check1 = 0.0 <= mean_final <= 0.5
    check2 = std_final > 0
    check3 = median_final >= 0
    
    if check1 and check2 and check3:
        print("✓ Statistics are reasonable")
    else:
        print("✗ Some statistics are unreasonable")
    
    return check1 and check2 and check3

def test_deterministic_with_seed():
    """Test that same seed produces same results"""
    print("\n=== Test 11.6: Deterministic with Seed ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 100
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Generate twice with same seed
    v_paths_1 = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=123)
    v_paths_2 = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=123)
    
    # Should be identical
    identical = np.allclose(v_paths_1, v_paths_2, rtol=1e-15, atol=1e-15)
    
    print(f"Same seed produces identical results: {identical}")
    
    # Different seed should give different results
    v_paths_3 = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=456)
    different = not np.allclose(v_paths_1, v_paths_3, rtol=1e-3, atol=1e-3)
    
    print(f"Different seed produces different results: {different}")
    
    if identical and different:
        print("✓ Deterministic behavior correct")
    else:
        print("✗ Deterministic behavior incorrect")
    
    return identical and different

def test_sample_paths_visualization():
    """Visualize sample paths"""
    print("\n=== Test 11.7: Sample Paths Visualization ===")
    
    v0 = 0.04
    kappa = HESTON_PARAMS['kappa']
    theta_v = HESTON_PARAMS['theta_v']
    sigma_v = HESTON_PARAMS['sigma_v']
    
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    v_paths = simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt, n_paths, seed=42)
    
    # Show statistics for first 10 paths
    print(f"\nSample of 10 paths (final values):")
    for i in range(10):
        print(f"  Path {i}: start={v_paths[i, 0]:.6f}, end={v_paths[i, -1]:.6f}")
    
    # Compute percentile bands
    p10 = np.percentile(v_paths, 10, axis=0)
    p50 = np.percentile(v_paths, 50, axis=0)
    p90 = np.percentile(v_paths, 90, axis=0)
    
    print(f"\nPercentile bands at final time:")
    print(f"  10th: {p10[-1]:.6f}")
    print(f"  50th: {p50[-1]:.6f}")
    print(f"  90th: {p90[-1]:.6f}")
    
    # Check reasonable spread
    check = p90[-1] > p10[-1]
    
    if check:
        print("✓ Paths show reasonable spread")
    else:
        print("✗ Paths appear degenerate")
    
    return check

def run_all_tests():
    """Run all Test 11 subtests"""
    print("=" * 60)
    print("TEST 11: HESTON VARIANCE - MULTIPLE PATHS")
    print("=" * 60)
    
    results = []
    
    results.append(("Shape", test_multiple_paths_shape()))
    results.append(("Start at v0", test_all_paths_start_at_v0()))
    results.append(("All Positive", test_all_values_positive()))
    results.append(("Path Diversity", test_paths_are_different()))
    results.append(("Aggregate Statistics", test_statistics_across_paths()))
    results.append(("Deterministic", test_deterministic_with_seed()))
    results.append(("Sample Paths", test_sample_paths_visualization()))
    
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
        print("SUCCESS: All multiple paths tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
