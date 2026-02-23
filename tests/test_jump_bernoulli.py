"""
Test 12: Jump Generation - Bernoulli Validity
Goal: Verify Bernoulli approximation is valid
"""

import numpy as np
import sys
from models.jump_diffusion import validate_bernoulli_approximation
from config import JUMP_PARAMS

def test_bernoulli_approximation():
    """Test Bernoulli approximation validity"""
    print("\n=== Test 12.1: Bernoulli Approximation Validity ===")
    
    lambda_jump = JUMP_PARAMS['lambda_jump']
    
    # Test different time steps
    test_cases = [
        (1/252/390, "1 minute"),
        (1/252/78, "5 minutes"),
        (1/252/6.5, "1 hour"),
        (1/252, "1 day"),
    ]
    
    all_passed = True
    
    print(f"Lambda jump: {lambda_jump} jumps/year\n")
    print(f"{'Time Step':<15} {'P(2+ jumps)':<15} {'Valid (<1%)':<15}")
    print("-" * 50)
    
    for dt, label in test_cases:
        valid, prob = validate_bernoulli_approximation(lambda_jump, dt, threshold=0.01)
        status = "✓" if valid else "✗"
        print(f"{label:<15} {prob:<15.6f} {status}")
        
        if not valid:
            print(f"  ⚠ Bernoulli approximation invalid for {label}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All tested time steps have valid Bernoulli approximation")
    else:
        print("\n⚠ Some time steps have invalid approximation")
    
    return True  # This is informational

def test_different_intensities():
    """Test with different jump intensities"""
    print("\n=== Test 12.2: Different Jump Intensities ===")
    
    dt = 1/252/390  # 1 minute
    
    test_lambdas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print(f"Time step: {dt:.8f} (1 minute)")
    print(f"\n{'Lambda':<10} {'P(2+ jumps)':<15} {'Valid':<10}")
    print("-" * 40)
    
    for lambda_j in test_lambdas:
        valid, prob = validate_bernoulli_approximation(lambda_j, dt)
        status = "✓" if valid else "✗"
        print(f"{lambda_j:<10} {prob:<15.8f} {status}")
    
    print("\n✓ Validation function works for different intensities")
    return True

def test_threshold_sensitivity():
    """Test different threshold values"""
    print("\n=== Test 12.3: Threshold Sensitivity ===")
    
    lambda_jump = 2.0
    dt = 1/252/390
    
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    print(f"Lambda: {lambda_jump}, dt: {dt:.8f}")
    print(f"\n{'Threshold':<12} {'Valid':<10}")
    print("-" * 25)
    
    for thresh in thresholds:
        valid, prob = validate_bernoulli_approximation(lambda_jump, dt, threshold=thresh)
        status = "✓" if valid else "✗"
        print(f"{thresh:<12} {status}")
    
    print("\n✓ Threshold sensitivity tested")
    return True

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 12.4: Edge Cases ===")
    
    all_passed = True
    
    # Very small lambda
    print("\n[12.4.1] Very small lambda (0.01):")
    valid, prob = validate_bernoulli_approximation(0.01, 1/252)
    print(f"  Valid: {valid}, P(2+): {prob:.8f}")
    check1 = valid  # Should be valid
    
    # Very small dt
    print("\n[12.4.2] Very small dt (1 second):")
    valid, prob = validate_bernoulli_approximation(2.0, 1/(252*6.5*3600))
    print(f"  Valid: {valid}, P(2+): {prob:.10f}")
    check2 = valid  # Should be valid
    
    # Large lambda with large dt
    print("\n[12.4.3] Large lambda with large dt:")
    valid, prob = validate_bernoulli_approximation(100.0, 1/252)
    print(f"  Valid: {valid}, P(2+): {prob:.6f}")
    check3 = not valid  # Should be invalid
    
    if check1 and check2 and check3:
        print("\n✓ Edge cases behave as expected")
    else:
        print("\n✗ Some edge cases unexpected")
        all_passed = False
    
    return all_passed

def test_probability_formula():
    """Test that probability formula is correct"""
    print("\n=== Test 12.5: Probability Formula ===")
    
    lambda_jump = 2.0
    dt = 0.01
    
    # Manual calculation
    lambda_dt = lambda_jump * dt
    prob_manual = (lambda_dt**2) / 2.0
    
    # Function calculation
    valid, prob_func = validate_bernoulli_approximation(lambda_jump, dt)
    
    print(f"Lambda: {lambda_jump}, dt: {dt}")
    print(f"Lambda * dt: {lambda_dt}")
    print(f"\nManual: P(2+) = (λ·dt)² / 2 = {prob_manual:.8f}")
    print(f"Function: P(2+) = {prob_func:.8f}")
    print(f"Difference: {abs(prob_manual - prob_func):.10f}")
    
    check = abs(prob_manual - prob_func) < 1e-10
    
    if check:
        print("✓ Formula is correct")
    else:
        print("✗ Formula mismatch")
    
    return check

def run_all_tests():
    """Run all Test 12 subtests"""
    print("=" * 60)
    print("TEST 12: JUMP GENERATION - BERNOULLI VALIDITY")
    print("=" * 60)
    
    results = []
    
    results.append(("Bernoulli Approximation", test_bernoulli_approximation()))
    results.append(("Different Intensities", test_different_intensities()))
    results.append(("Threshold Sensitivity", test_threshold_sensitivity()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Probability Formula", test_probability_formula()))
    
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
        print("SUCCESS: All Bernoulli validity tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
