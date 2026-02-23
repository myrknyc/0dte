"""
Test 3: Black-Scholes Baseline
===============================
Goal: Ensure pricing baseline works
"""

import sys
import numpy as np

print("="*60)
print("TEST 3: BLACK-SCHOLES BASELINE")
print("="*60)

# Import black_scholes function
try:
    from pricing.black_scholes import black_scholes
    print("✓ Successfully imported black_scholes function")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

tests_passed = 0
tests_failed = 0
failures = []

# Test 1: Price a simple ATM call
print("\n[1] Pricing ATM call option...")
try:
    S = 100
    K = 100
    T = 0.25  # 3 months
    r = 0.05
    sigma = 0.2
    
    call_price = black_scholes(S, K, T, r, sigma, 'call')
    
    print(f"    S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"    Call price: ${call_price:.4f}")
    
    # Expected range: ~$5-6 for these parameters
    if 4.5 <= call_price <= 6.5:
        print(f"    ✓ Price is reasonable (expected ~$5-6)")
        tests_passed += 1
    else:
        print(f"    ✗ Price outside expected range [4.5, 6.5]")
        tests_failed += 1
        failures.append(f"Call price {call_price:.4f} outside reasonable range")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"ATM call pricing error: {e}")

# Test 2: Price corresponding put
print("\n[2] Pricing ATM put option...")
try:
    put_price = black_scholes(S, K, T, r, sigma, 'put')
    print(f"    Put price: ${put_price:.4f}")
    
    # Put should also be reasonable
    if 3.0 <= put_price <= 5.0:
        print(f"    ✓ Price is reasonable")
        tests_passed += 1
    else:
        print(f"    ✗ Price outside expected range [3.0, 5.0]")
        tests_failed += 1
        failures.append(f"Put price {put_price:.4f} outside reasonable range")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"ATM put pricing error: {e}")

# Test 3: Put-Call Parity
print("\n[3] Testing put-call parity...")
try:
    # C - P = S - K*e^(-rT)
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    
    print(f"    C - P = {lhs:.6f}")
    print(f"    S - K*e^(-rT) = {rhs:.6f}")
    print(f"    Difference: {abs(lhs - rhs):.8f}")
    
    if abs(lhs - rhs) < 1e-6:
        print(f"    ✓ Put-call parity holds")
        tests_passed += 1
    else:
        print(f"    ✗ Put-call parity violated")
        tests_failed += 1
        failures.append(f"Put-call parity error: {abs(lhs - rhs):.8f}")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"Put-call parity test error: {e}")

# Test 4: Deep ITM call (should be worth ~intrinsic value)
print("\n[4] Testing deep ITM call...")
try:
    K_itm = 80
    call_itm = black_scholes(S, K_itm, T, r, sigma, 'call')
    intrinsic = S - K_itm
    
    print(f"    K={K_itm}, Call price: ${call_itm:.4f}")
    print(f"    Intrinsic value: ${intrinsic:.4f}")
    
    # Deep ITM call should be close to intrinsic value
    if call_itm >= intrinsic and call_itm <= intrinsic + 3:
        print(f"    ✓ ITM call >= intrinsic value")
        tests_passed += 1
    else:
        print(f"    ✗ ITM call price not reasonable")
        tests_failed += 1
        failures.append(f"ITM call price {call_itm:.4f} vs intrinsic {intrinsic:.4f}")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"ITM call test error: {e}")

# Test 5: Deep OTM call (should be worth very little)
print("\n[5] Testing deep OTM call...")
try:
    K_otm = 120
    call_otm = black_scholes(S, K_otm, T, r, sigma, 'call')
    
    print(f"    K={K_otm}, Call price: ${call_otm:.4f}")
    
    # Deep OTM call should be worth < $2
    if 0 < call_otm < 2.0:
        print(f"    ✓ OTM call has low value")
        tests_passed += 1
    else:
        print(f"    ✗ OTM call price not reasonable")
        tests_failed += 1
        failures.append(f"OTM call price {call_otm:.4f} not in (0, 2)")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"OTM call test error: {e}")

# Test 6: Monotonicity - higher strike = lower call price
print("\n[6] Testing monotonicity (strike vs call price)...")
try:
    strikes = [90, 95, 100, 105, 110]
    prices = [black_scholes(S, K_test, T, r, sigma, 'call') for K_test in strikes]
    
    print(f"    Strikes: {strikes}")
    print(f"    Prices:  {[f'{p:.4f}' for p in prices]}")
    
    # Check if prices are strictly decreasing
    is_decreasing = all(prices[i] > prices[i+1] for i in range(len(prices)-1))
    
    if is_decreasing:
        print(f"    ✓ Call prices decrease with strike (monotonic)")
        tests_passed += 1
    else:
        print(f"    ✗ Call prices not monotonic")
        tests_failed += 1
        failures.append("Call prices not decreasing with strike")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"Monotonicity test error: {e}")

# Test 7: Edge case - zero time to expiry
print("\n[7] Testing edge case (T=0, expiry)...")
try:
    call_expired = black_scholes(105, 100, 0, r, sigma, 'call')
    put_expired = black_scholes(95, 100, 0, r, sigma, 'put')
    
    print(f"    Call (S=105, K=100, T=0): ${call_expired:.4f} (expected: $5.00)")
    print(f"    Put (S=95, K=100, T=0): ${put_expired:.4f} (expected: $5.00)")
    
    # At expiry, should be intrinsic value
    if abs(call_expired - 5.0) < 0.01 and abs(put_expired - 5.0) < 0.01:
        print(f"    ✓ Expired options = intrinsic value")
        tests_passed += 1
    else:
        print(f"    ✗ Expired option values incorrect")
        tests_failed += 1
        failures.append(f"Expired values: call={call_expired:.4f}, put={put_expired:.4f}")
        
except Exception as e:
    print(f"    ✗ Error: {e}")
    tests_failed += 1
    failures.append(f"Edge case test error: {e}")

# Summary
print("\n" + "="*60)
print("TEST 3 RESULTS")
print("="*60)
print(f"Passed: {tests_passed}/7")
print(f"Failed: {tests_failed}/7")

if failures:
    print("\nFailures:")
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}")

if tests_failed > 0:
    print("\nTest 3: FAILED")
    sys.exit(1)
else:
    print("\nTest 3: PASSED - Black-Scholes baseline working correctly!")
    sys.exit(0)
