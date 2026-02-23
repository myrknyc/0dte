"""
Test 4: Time Grid Generation
Goal: Verify time grid creates valid arrays
"""

import numpy as np
import sys
from core.time_grid import (
    generate_time_grid, 
    uniform_grid, 
    adaptive_exponential_grid,
    get_grid_statistics
)

def test_uniform_grid():
    """Test uniform grid generation"""
    print("\n=== Test 4.1: Uniform Grid Generation ===")
    
    T = 0.25  # 3 months
    n_steps = 100
    
    # Generate uniform grid
    times, dt = generate_time_grid(T, n_steps, use_adaptive=False)
    
    print(f"Parameters: T={T}, n_steps={n_steps}")
    print(f"Generated {len(times)} time points")
    print(f"Generated {len(dt)} time steps")
    
    # Check: len(times) == n_steps + 1
    check1 = len(times) == n_steps + 1
    print(f"len(times) == n_steps + 1: {check1} ({len(times)} == {n_steps + 1})")
    
    # Check: times[0] == 0, times[-1] == T
    check2 = abs(times[0]) < 1e-10
    check3 = abs(times[-1] - T) < 1e-10
    print(f"times[0] == 0: {check2} (times[0] = {times[0]})")
    print(f"times[-1] == T: {check3} (times[-1] = {times[-1]}, T = {T})")
    
    # Check: all dt > 0
    check4 = np.all(dt > 0)
    min_dt = np.min(dt)
    max_dt = np.max(dt)
    print(f"all dt > 0: {check4} (min_dt = {min_dt:.6f})")
    
    # Check: sum(dt) ≈ T
    dt_sum = np.sum(dt)
    check5 = abs(dt_sum - T) < 1e-10
    print(f"sum(dt) ≈ T: {check5} (sum = {dt_sum:.10f}, T = {T})")
    
    # For uniform grid, all dt should be equal
    dt_std = np.std(dt)
    check6 = dt_std < 1e-10
    print(f"Uniform spacing: {check6} (std(dt) = {dt_std:.2e})")
    
    all_passed = check1 and check2 and check3 and check4 and check5 and check6
    
    if all_passed:
        print("✓ Uniform grid generation passed")
    else:
        print("✗ Some uniform grid checks failed")
    
    return all_passed

def test_adaptive_grid():
    """Test adaptive grid generation"""
    print("\n=== Test 4.2: Adaptive Grid Generation ===")
    
    # For 0DTE, use short time to expiry (e.g., 3 hours)
    T = 3.0 / (252 * 6.5)  # 3 hours in years
    n_steps = 100
    density_factor = 3.0
    
    # Generate adaptive grid
    times, dt = generate_time_grid(T, n_steps, density_factor=density_factor, use_adaptive=True)
    
    print(f"Parameters: T={T:.6f} years ({T*252*6.5:.1f} hours)")
    print(f"n_steps={n_steps}, density_factor={density_factor}")
    print(f"Generated {len(times)} time points")
    
    # Check: len(times) == n_steps + 1
    check1 = len(times) == n_steps + 1
    print(f"len(times) == n_steps + 1: {check1}")
    
    # Check: times[0] == 0, times[-1] == T
    check2 = abs(times[0]) < 1e-10
    check3 = abs(times[-1] - T) < 1e-10
    print(f"times[0] == 0: {check2}")
    print(f"times[-1] == T: {check3}")
    
    # Check: all dt > 0
    check4 = np.all(dt > 0)
    min_dt = np.min(dt)
    max_dt = np.max(dt)
    print(f"all dt > 0: {check4} (min = {min_dt:.2e}, max = {max_dt:.2e})")
    
    # Check: sum(dt) ≈ T
    dt_sum = np.sum(dt)
    check5 = abs(dt_sum - T) < 1e-10
    print(f"sum(dt) ≈ T: {check5} (diff = {abs(dt_sum - T):.2e})")
    
    # Verify adaptive grid has more resolution near T=0
    # First 10 steps should be smaller than last 10 steps
    first_10_avg = np.mean(dt[:10])
    last_10_avg = np.mean(dt[-10:])
    check6 = first_10_avg < last_10_avg
    
    print(f"\nAdaptive density check:")
    print(f"  First 10 steps avg: {first_10_avg:.2e}")
    print(f"  Last 10 steps avg: {last_10_avg:.2e}")
    print(f"  Ratio (last/first): {last_10_avg/first_10_avg:.2f}×")
    print(f"  Denser at start: {check6}")
    
    # Get statistics
    stats = get_grid_statistics(times, dt)
    print(f"\nGrid statistics:")
    print(f"  Clustering ratio: {stats['ratio']:.2f}×")
    
    all_passed = check1 and check2 and check3 and check4 and check5 and check6
    
    if all_passed:
        print("✓ Adaptive grid generation passed")
    else:
        print("✗ Some adaptive grid checks failed")
    
    return all_passed

def test_adaptive_vs_uniform():
    """Compare adaptive vs uniform grids"""
    print("\n=== Test 4.3: Adaptive vs Uniform Comparison ===")
    
    T = 3.0 / (252 * 6.5)  # 3 hours
    n_steps = 100
    
    # Generate both grids
    times_uniform, dt_uniform = generate_time_grid(T, n_steps, use_adaptive=False)
    times_adaptive, dt_adaptive = generate_time_grid(T, n_steps, use_adaptive=True)
    
    # Both should have same total time
    check1 = abs(times_uniform[-1] - times_adaptive[-1]) < 1e-10
    print(f"Same total time T: {check1}")
    
    # Both should have same number of steps
    check2 = len(dt_uniform) == len(dt_adaptive)
    print(f"Same number of steps: {check2}")
    
    # Adaptive should have more variability
    std_uniform = np.std(dt_uniform)
    std_adaptive = np.std(dt_adaptive)
    check3 = std_adaptive > std_uniform
    
    print(f"\nVariability:")
    print(f"  Uniform std: {std_uniform:.2e}")
    print(f"  Adaptive std: {std_adaptive:.2e}")
    print(f"  Adaptive more variable: {check3}")
    
    # Adaptive should have smaller minimum dt
    min_uniform = np.min(dt_uniform)
    min_adaptive = np.min(dt_adaptive)
    check4 = min_adaptive < min_uniform
    
    print(f"\nMinimum time step:")
    print(f"  Uniform: {min_uniform:.2e}")
    print(f"  Adaptive: {min_adaptive:.2e}")
    print(f"  Adaptive smaller: {check4}")
    
    all_passed = check1 and check2 and check3 and check4
    
    if all_passed:
        print("✓ Adaptive vs uniform comparison passed")
    else:
        print("✗ Some comparison checks failed")
    
    return all_passed

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 4.4: Edge Cases ===")
    
    all_passed = True
    
    # Test 1: Very small T
    print("\n[4.4.1] Very small T (1 minute):")
    T_small = 1.0 / (252 * 6.5 * 60)  # 1 minute
    times, dt = generate_time_grid(T_small, 50, use_adaptive=True)
    check1 = abs(times[-1] - T_small) < 1e-15 and np.all(dt > 0)
    print(f"  T = {T_small:.2e}, times[-1] = {times[-1]:.2e}")
    print(f"  Valid: {check1}")
    all_passed = all_passed and check1
    
    # Test 2: Very large T
    print("\n[4.4.2] Large T (1 year):")
    T_large = 1.0
    times, dt = generate_time_grid(T_large, 100, use_adaptive=False)
    check2 = abs(times[-1] - T_large) < 1e-10 and np.all(dt > 0)
    print(f"  T = {T_large}, times[-1] = {times[-1]:.10f}")
    print(f"  Valid: {check2}")
    all_passed = all_passed and check2
    
    # Test 3: Few steps (n_steps=5)
    print("\n[4.4.3] Few steps (n_steps=5):")
    times, dt = generate_time_grid(0.1, 5, use_adaptive=False)
    check3 = len(times) == 6 and len(dt) == 5
    print(f"  len(times) = {len(times)}, len(dt) = {len(dt)}")
    print(f"  Valid: {check3}")
    all_passed = all_passed and check3
    
    # Test 4: Many steps (n_steps=1000)
    print("\n[4.4.4] Many steps (n_steps=1000):")
    times, dt = generate_time_grid(0.25, 1000, use_adaptive=False)
    check4 = len(times) == 1001 and len(dt) == 1000 and np.all(dt > 0)
    print(f"  len(times) = {len(times)}, len(dt) = {len(dt)}")
    print(f"  All dt > 0: {np.all(dt > 0)}")
    print(f"  Valid: {check4}")
    all_passed = all_passed and check4
    
    if all_passed:
        print("\n✓ Edge cases passed")
    else:
        print("\n✗ Some edge cases failed")
    
    return all_passed

def test_numerical_stability():
    """Test numerical stability and floating point precision"""
    print("\n=== Test 4.5: Numerical Stability ===")
    
    T = 0.25
    n_steps = 100
    
    # Generate grid multiple times
    times1, dt1 = generate_time_grid(T, n_steps, use_adaptive=False)
    times2, dt2 = generate_time_grid(T, n_steps, use_adaptive=False)
    
    # Should be identical (deterministic)
    check1 = np.allclose(times1, times2, rtol=1e-15, atol=1e-15)
    check2 = np.allclose(dt1, dt2, rtol=1e-15, atol=1e-15)
    
    print(f"Deterministic (uniform): {check1 and check2}")
    
    # Test adaptive grid
    times3, dt3 = generate_time_grid(T, n_steps, use_adaptive=True, density_factor=3.0)
    times4, dt4 = generate_time_grid(T, n_steps, use_adaptive=True, density_factor=3.0)
    
    check3 = np.allclose(times3, times4, rtol=1e-15, atol=1e-15)
    print(f"Deterministic (adaptive): {check3}")
    
    # Test that boundaries are exact
    check4 = times1[0] == 0.0 and times1[-1] == T
    check5 = times3[0] == 0.0 and times3[-1] == T
    
    print(f"Exact boundaries (uniform): {check4}")
    print(f"Exact boundaries (adaptive): {check5}")
    
    all_passed = check1 and check2 and check3 and check4 and check5
    
    if all_passed:
        print("✓ Numerical stability passed")
    else:
        print("✗ Some stability checks failed")
    
    return all_passed

def run_all_tests():
    """Run all Test 4 subtests"""
    print("=" * 60)
    print("TEST 4: TIME GRID GENERATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Uniform Grid", test_uniform_grid()))
    results.append(("Adaptive Grid", test_adaptive_grid()))
    results.append(("Adaptive vs Uniform", test_adaptive_vs_uniform()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Numerical Stability", test_numerical_stability()))
    
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
        print("SUCCESS: All time grid generation tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
