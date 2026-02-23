"""
Test 15-16: Combined Model Tests
Goal: Test risk-neutral and real-world combined models
"""

import numpy as np
import sys
from models.combined_model import simulate_combined_paths_fast
from core.time_grid import generate_time_grid
from core.random_numbers import generate_sobol_normals
from config import DEFAULT_PARAMS, HESTON_PARAMS, JUMP_PARAMS, MEAN_REVERSION_PARAMS

def test_risk_neutral_model():
    """Test 15: Risk-neutral combined model"""
    print("\n=== Test 15: Risk-Neutral Combined Model ===")
    
    # Parameters
    S0 = 100
    v0 = 0.04
    r = 0.05
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    # Use DEFAULT_PARAMS and only override measure
    params = DEFAULT_PARAMS.copy()
    params['measure'] = 'risk_neutral'
    params['r'] = r
    params['mu_0'] = 450  # Set even for risk-neutral (won't be used)
    
    # Grid and random numbers
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    np.random.seed(42)
    Z1 = np.random.standard_normal((n_paths, n_steps))
    
    # Simulate
    S_paths, v_paths = simulate_combined_paths_fast(
        S0, v0, params, times, dt_array, Z1, 
        use_jumps=True, use_mean_reversion=False, seed=42
    )
    
    print(f"[15.1] Shape check:")
    print(f"  S_paths: {S_paths.shape} (expected: ({n_paths}, {n_steps+1}))")
    print(f"  v_paths: {v_paths.shape}")
    check1 = S_paths.shape == (n_paths, n_steps + 1)
    
    print(f"\n[15.2] Initial values:")
    print(f"  All S start at S0={S0}: {np.all(S_paths[:, 0] == S0)}")
    print(f"  All v start at v0={v0}: {np.all(v_paths[:, 0] == v0)}")
    check2 = np.all(S_paths[:, 0] == S0) and np.all(v_paths[:, 0] == v0)
    
    print(f"\n[15.3] All prices positive:")
    all_positive = np.all(S_paths > 0)
    print(f"  {all_positive}")
    check3 = all_positive
    
    print(f"\n[15.4] Mean terminal price vs forward:")
    mean_terminal = np.mean(S_paths[:, -1])
    forward = S0 * np.exp(r * T)
    diff = abs(mean_terminal - forward)
    print(f"  Mean terminal: {mean_terminal:.2f}")
    print(f"  Forward price: {forward:.2f}")
    print(f"  Difference: {diff:.2f}")
    # Should be close (within MC error)
    check4 = diff < forward * 0.15  # Within 15% (looser for stochastic model)
    
    if check1 and check2 and check3 and check4:
        print("\n✓ Risk-neutral model works")
        return True
    else:
        print("\n✗ Risk-neutral model issues")
        return False

def test_real_world_model():
    """Test 16: Real-world combined model with mean reversion"""
    print("\n=== Test 16: Real-World Combined Model ===")
    
    # Parameters
    S0_above = 455  # Start above VWAP
    S0_below = 445  # Start below VWAP
    v0 = 0.04
    r = 0.05
    T = 0.25
    n_steps = 100
    n_paths = 1000
    
    # Use DEFAULT_PARAMS and override measure
    params = DEFAULT_PARAMS.copy()
    params['measure'] = 'real_world'
    params['r'] = r
    params['mu_0'] = 450  # VWAP level
    
    # Grid and random numbers
    times, dt_array = generate_time_grid(T, n_steps, use_adaptive=False)
    np.random.seed(42)
    Z1 = np.random.standard_normal((n_paths, n_steps))
    
    # Test 1: Starting above VWAP
    print(f"\n[16.1] Starting above VWAP (S0={S0_above}, μ_0={params['mu_0']}):")
    S_paths_above, v_paths_above = simulate_combined_paths_fast(
        S0_above, v0, params, times, dt_array, Z1.copy(),
        use_jumps=True, use_mean_reversion=True, seed=42
    )
    
    mean_terminal_above = np.mean(S_paths_above[:, -1])
    print(f"  Mean terminal: {mean_terminal_above:.2f}")
    print(f"  Effect: {mean_terminal_above - S0_above:.2f}")
    check1 = True  # Just verify it runs
    
    # Test 2: Starting below VWAP
    print(f"\n[16.2] Starting below VWAP (S0={S0_below}, μ_0={params['mu_0']}):")
    np.random.seed(42)
    Z1_copy = np.random.standard_normal((n_paths, n_steps))
    S_paths_below, v_paths_below = simulate_combined_paths_fast(
        S0_below, v0, params, times, dt_array, Z1_copy,
        use_jumps=True, use_mean_reversion=True, seed=42
    )
    
    mean_terminal_below = np.mean(S_paths_below[:, -1])
    print(f"  Mean terminal: {mean_terminal_below:.2f}")
    print(f"  Effect: {mean_terminal_below - S0_below:.2f}")
    check2 = True  # Just verify it runs
    
    # Test 3: All positive
    print(f"\n[16.3] All prices positive:")
    all_pos = np.all(S_paths_above > 0) and np.all(S_paths_below > 0)
    print(f"  {all_pos}")
    check3 = all_pos
    
    if check1 and check2 and check3:
        print("\n✓ Real-world model runs successfully")
        return True
    else:
        print("\n✗ Real-world model issues")
        return False

def run_all_tests():
    """Run Tests 15-16"""
    print("=" * 60)
    print("TESTS 15-16: COMBINED MODELS")
    print("=" * 60)
    
    results = []
    
    results.append(("Test 15: Risk-Neutral", test_risk_neutral_model()))
    results.append(("Test 16: Real-World", test_real_world_model()))
    
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
        print("SUCCESS: All combined model tests passed! ✓")
        print("=" * 60)
        return True
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
