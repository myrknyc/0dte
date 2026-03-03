"""
Tests for Heston Characteristic Function Pricing
Tests: CF vs BS convergence, put-call parity, intrinsic bounds
"""

import numpy as np
import sys


def test_heston_cf_vs_bs():
    """Heston CF should converge to BS when sigma_v → 0 and rho → 0."""
    print("\n" + "=" * 60)
    print("TEST: Heston CF vs Black-Scholes Convergence")
    print("=" * 60)

    from calibration.heston_cf import heston_cf_price
    from pricing.black_scholes import black_scholes

    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    sigma = 0.20
    v0 = sigma ** 2

    # Nearly BS: sigma_v very small, rho=0
    cf_call = heston_cf_price(S, K, T, r, kappa=2.0, theta_v=v0,
                              sigma_v=0.01, rho=0.0, v0=v0, option_type='call')
    bs_call = black_scholes(S, K, T, r, sigma, 'call')

    diff = abs(cf_call - bs_call)
    print(f"  CF call:  ${cf_call:.4f}")
    print(f"  BS call:  ${bs_call:.4f}")
    print(f"  Diff:     ${diff:.4f}")

    ok = diff < 0.10  # within $0.10
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: diff < $0.10")
    return ok


def test_put_call_parity():
    """C - P = S - K·e^(-rT)."""
    print("\n" + "=" * 60)
    print("TEST: Put-Call Parity")
    print("=" * 60)

    from calibration.heston_cf import heston_cf_price

    S, K, T, r = 100.0, 102.0, 0.25, 0.05
    kappa, theta_v, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    call = heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0, 'call')
    put = heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0, 'put')

    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    diff = abs(lhs - rhs)

    print(f"  C - P = {lhs:.4f}")
    print(f"  S - Ke^(-rT) = {rhs:.4f}")
    print(f"  Diff: {diff:.4f}")

    ok = diff < 0.05
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: parity within $0.05")
    return ok


def test_intrinsic_bounds():
    """CF price should not be below intrinsic value."""
    print("\n" + "=" * 60)
    print("TEST: Intrinsic Value Bounds")
    print("=" * 60)

    from calibration.heston_cf import heston_cf_price

    S, T, r = 100.0, 0.1, 0.05
    kappa, theta_v, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04
    
    all_ok = True
    for K in [90, 95, 100, 105, 110]:
        call = heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0, 'call')
        put = heston_cf_price(S, K, T, r, kappa, theta_v, sigma_v, rho, v0, 'put')
        # European lower bound is discounted intrinsic
        disc = np.exp(-r * T)
        intrinsic_c = max(S - K * disc, 0)
        intrinsic_p = max(K * disc - S, 0) if K > S else 0
        # For puts, the simpler bound is max(K*e^{-rT} - S, 0)
        intrinsic_p = max(K - S, 0) * disc

        ok_c = call >= intrinsic_c - 0.01
        ok_p = put >= intrinsic_p - 0.01
        all_ok = all_ok and ok_c and ok_p
        
        print(f"  K={K}: call=${call:.4f} >= ${intrinsic_c:.2f} {'✓' if ok_c else '✗'}  "
              f"put=${put:.4f} >= ${intrinsic_p:.2f} {'✓' if ok_p else '✗'}")

    print(f"\n  {'✓ PASS' if all_ok else '✗ FAIL'}")
    return all_ok


def test_chain_pricing():
    """Price a chain of strikes at once."""
    print("\n" + "=" * 60)
    print("TEST: Chain Pricing")
    print("=" * 60)

    from calibration.heston_cf import heston_cf_price_chain

    S, T, r = 100, 0.25, 0.05
    strikes = np.array([95, 97.5, 100, 102.5, 105])
    kappa, theta_v, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    prices = heston_cf_price_chain(S, strikes, T, r, kappa, theta_v, sigma_v, rho, v0)

    all_ok = True
    for K, p in zip(strikes, prices):
        ok = p >= 0
        all_ok = all_ok and ok
        print(f"  K={K}: ${p:.4f} {'✓' if ok else '✗'}")

    # Monotonicity: call prices decrease with strike
    mono = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
    all_ok = all_ok and mono
    print(f"  Monotonicity: {'✓' if mono else '✗'}")

    print(f"\n  {'✓ PASS' if all_ok else '✗ FAIL'}")
    return all_ok


def test_expiry():
    """At T=0, CF should return intrinsic."""
    print("\n" + "=" * 60)
    print("TEST: At-Expiry Pricing")
    print("=" * 60)

    from calibration.heston_cf import heston_cf_price

    S, r = 100.0, 0.05
    kappa, theta_v, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    all_ok = True
    for K in [95, 100, 105]:
        call = heston_cf_price(S, K, 0.0, r, kappa, theta_v, sigma_v, rho, v0, 'call')
        put = heston_cf_price(S, K, 0.0, r, kappa, theta_v, sigma_v, rho, v0, 'put')
        exp_c = max(S - K, 0)
        exp_p = max(K - S, 0)
        ok_c = abs(call - exp_c) < 0.01
        ok_p = abs(put - exp_p) < 0.01
        all_ok = all_ok and ok_c and ok_p
        print(f"  K={K}: call={call:.4f}=={exp_c:.2f}{'✓' if ok_c else '✗'}  "
              f"put={put:.4f}=={exp_p:.2f}{'✓' if ok_p else '✗'}")

    print(f"\n  {'✓ PASS' if all_ok else '✗ FAIL'}")
    return all_ok


def run_all():
    results = []
    results.append(("CF vs BS", test_heston_cf_vs_bs()))
    results.append(("Put-Call Parity", test_put_call_parity()))
    results.append(("Intrinsic Bounds", test_intrinsic_bounds()))
    results.append(("Chain Pricing", test_chain_pricing()))
    results.append(("At-Expiry", test_expiry()))

    print("\n" + "=" * 60)
    print("HESTON CF TEST SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name:<25} {'PASS ✓' if ok else 'FAIL ✗'}")
    total = sum(ok for _, ok in results)
    print(f"\n  {total}/{len(results)} passed")
    return total == len(results)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
