"""
Test 6: Random Number Generation - Moment Matching
Goal: Verify exact moment correction
"""

import numpy as np
import sys
from core.random_numbers import apply_moment_matching

def test_exact_mean():
    """Test that moment matching produces exact mean = 0"""
    print("\n=== Test 6.1: Exact Mean = 0 ===")
    
    # Generate random normals (not centered)
    np.random.seed(42)
    Z = np.random.standard_normal((1000, 100))
    
    mean_before = np.mean(Z)
    print(f"Mean before: {mean_before:.10f}")
    
    # Apply moment matching
    Z_matched = apply_moment_matching(Z)
    
    mean_after = np.mean(Z_matched)
    print(f"Mean after:  {mean_after:.15e}")
    
    # Check if exactly 0 (or extremely close)
    check = abs(mean_after) < 1e-10
    
    if check:
        print(f"✓ Mean is exactly 0 (< 1e-10)")
    else:
        print(f"✗ Mean not exactly 0: {mean_after}")
    
    return check

def test_exact_std():
    """Test that moment matching produces exact std = 1"""
    print("\n=== Test 6.2: Exact Std = 1 ===")
    
    # Generate random normals
    np.random.seed(42)
    Z = np.random.standard_normal((1000, 100))
    
    std_before = np.std(Z, ddof=0)  # Population std
    print(f"Std before: {std_before:.10f}")
    
    # Apply moment matching
    Z_matched = apply_moment_matching(Z)
    
    std_after = np.std(Z_matched, ddof=0)
    print(f"Std after:  {std_after:.15e}")
    
    # Check if exactly 1 (or extremely close)
    check = abs(std_after - 1.0) < 1e-10
    
    if check:
        print(f"✓ Std is exactly 1 (< 1e-10)")
    else:
        print(f"✗ Std not exactly 1: {std_after}")
    
    return check

def test_preserves_shape():
    """Test that moment matching preserves array shape"""
    print("\n=== Test 6.3: Preserves Shape ===")
    
    shapes_to_test = [
        (100, 50),
        (1000, 100),
        (10000, 200),
    ]
    
    all_passed = True
    
    for shape in shapes_to_test:
        np.random.seed(42)
        Z = np.random.standard_normal(shape)
        Z_matched = apply_moment_matching(Z)
        
        check = Z_matched.shape == shape
        print(f"  Shape {shape}: {check}")
        all_passed = all_passed and check
    
    if all_passed:
        print("✓ All shapes preserved")
    else:
        print("✗ Some shapes not preserved")
    
    return all_passed

def test_large_arrays():
    """Test moment matching works for large arrays"""
    print("\n=== Test 6.4: Large Arrays (10,000+ elements) ===")
    
    np.random.seed(42)
    Z = np.random.standard_normal((10000, 100))  # 1,000,000 elements
    
    print(f"Array size: {Z.size:,} elements")
    
    Z_matched = apply_moment_matching(Z)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    
    print(f"Mean: {mean:.15e}")
    print(f"Std:  {std:.15e}")
    
    check_mean = abs(mean) < 1e-10
    check_std = abs(std - 1.0) < 1e-10
    
    if check_mean and check_std:
        print("✓ Large array moments exact")
    else:
        print("✗ Large array moments not exact")
    
    return check_mean and check_std

def test_different_distributions():
    """Test on different starting distributions"""
    print("\n=== Test 6.5: Different Starting Distributions ===")
    
    all_passed = True
    
    # Test 1: Uniform distribution
    print("\n[6.5.1] Starting from Uniform(-3, 3):")
    np.random.seed(42)
    Z_uniform = np.random.uniform(-3, 3, size=(1000, 100))
    Z_matched = apply_moment_matching(Z_uniform)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}")
    
    check1 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Exact moments: {check1}")
    all_passed = all_passed and check1
    
    # Test 2: Exponential distribution
    print("\n[6.5.2] Starting from Exponential(λ=1):")
    np.random.seed(42)
    Z_exp = np.random.exponential(1.0, size=(1000, 100))
    Z_matched = apply_moment_matching(Z_exp)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}")
    
    check2 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Exact moments: {check2}")
    all_passed = all_passed and check2
    
    # Test 3: Already matched distribution
    print("\n[6.5.3] Starting from perfect N(0,1):")
    np.random.seed(42)
    Z_perfect = np.random.standard_normal((10000, 100))
    # Artificially make it perfect
    Z_perfect = (Z_perfect - np.mean(Z_perfect)) / np.std(Z_perfect, ddof=0)
    
    # Apply again
    Z_matched = apply_moment_matching(Z_perfect)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}")
    
    check3 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Exact moments: {check3}")
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ Works for different distributions")
    else:
        print("\n✗ Some distributions failed")
    
    return all_passed

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 6.6: Edge Cases ===")
    
    all_passed = True
    
    # Test 1: Small array
    print("\n[6.6.1] Very small array (10×10):")
    np.random.seed(42)
    Z_small = np.random.standard_normal((10, 10))
    Z_matched = apply_moment_matching(Z_small)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    check1 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}, Valid: {check1}")
    all_passed = all_passed and check1
    
    # Test 2: 1D array
    print("\n[6.6.2] 1D array (1000 elements):")
    np.random.seed(42)
    Z_1d = np.random.standard_normal(1000)
    Z_matched = apply_moment_matching(Z_1d)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    check2 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}, Valid: {check2}")
    all_passed = all_passed and check2
    
    # Test 3: Non-square array
    print("\n[6.6.3] Non-square array (500×200):")
    np.random.seed(42)
    Z_rect = np.random.standard_normal((500, 200))
    Z_matched = apply_moment_matching(Z_rect)
    
    mean = np.mean(Z_matched)
    std = np.std(Z_matched, ddof=0)
    check3 = abs(mean) < 1e-10 and abs(std - 1.0) < 1e-10
    print(f"  Mean: {mean:.2e}, Std: {std:.10f}, Valid: {check3}")
    all_passed = all_passed and check3
    
    if all_passed:
        print("\n✓ Edge cases passed")
    else:
        print("\n✗ Some edge cases failed")
    
    return all_passed

def test_deterministic():
    """Test that moment matching is deterministic"""
    print("\n=== Test 6.7: Deterministic Behavior ===")
    
    np.random.seed(42)
    Z = np.random.standard_normal((1000, 100))
    
    # Apply twice
    Z_matched1 = apply_moment_matching(Z.copy())
    Z_matched2 = apply_moment_matching(Z.copy())
    
    # Should be identical
    identical = np.allclose(Z_matched1, Z_matched2, rtol=1e-15, atol=1e-15)
    
    print(f"Same input produces identical output: {identical}")
    
    if identical:
        print("✓ Deterministic behavior confirmed")
    else:
        print("✗ Non-deterministic behavior detected")
    
    return identical

def run_all_tests():
    """Run all Test 6 subtests"""
    print("=" * 60)
    print("TEST 6: RANDOM NUMBER GENERATION - MOMENT MATCHING")
    print("=" * 60)
    
    results = []
    
    results.append(("Exact Mean = 0", test_exact_mean()))
    results.append(("Exact Std = 1", test_exact_std()))
    results.append(("Preserves Shape", test_preserves_shape()))
    results.append(("Large Arrays", test_large_arrays()))
    results.append(("Different Distributions", test_different_distributions()))
    results.append(("Edge Cases", test_edge_cases()))
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
        print("SUCCESS: All moment matching tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
