"""
Test 2: Configuration
=====================
Goal: Verify config parameters are valid
"""

import sys
from datetime import time


def run_all():
    print("="*60)
    print("TEST 2: CONFIGURATION VERIFICATION")
    print("="*60)

    try:
        import config
        print("OK imported config module")
    except Exception as e:
        print(f"FAIL to import config: {e}")
        return False

    tests_passed = 0
    tests_failed = 0
    failures = []

    # Test 1: Check DEFAULT_PARAMS exists and has expected keys
    print("\n[1] Checking DEFAULT_PARAMS...")
    try:
        assert hasattr(config, 'DEFAULT_PARAMS'), "DEFAULT_PARAMS not found"
        params = config.DEFAULT_PARAMS

        expected_keys = [
            'kappa', 'theta_v', 'sigma_v', 'rho', 'v0',
            'lambda_jump', 'mu_jump', 'sigma_jump',
            'theta_mr', 'mu_0',
            'r',
            'measure'
        ]

        print(f"    Expected keys: {len(expected_keys)}")
        missing_keys = [key for key in expected_keys if key not in params]

        if missing_keys:
            print(f"    FAIL Missing keys: {missing_keys}")
            tests_failed += 1
            failures.append(f"Missing keys in DEFAULT_PARAMS: {missing_keys}")
        else:
            print(f"    OK All {len(expected_keys)} expected keys present")
            tests_passed += 1

    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 2: Verify 'measure' parameter
    print("\n[2] Checking 'measure' parameter...")
    try:
        measure = config.DEFAULT_PARAMS.get('measure')
        assert measure is not None, "'measure' parameter is None"
        assert measure in ['risk_neutral', 'real_world'], f"Invalid measure value: {measure}"
        print(f"    OK measure = '{measure}' (valid)")
        tests_passed += 1
    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 3: Verify 'mu_0' parameter
    print("\n[3] Checking 'mu_0' parameter...")
    try:
        assert 'mu_0' in config.DEFAULT_PARAMS, "'mu_0' key not found"
        mu_0 = config.DEFAULT_PARAMS['mu_0']
        print(f"    OK mu_0 = {mu_0} (can be None)")
        tests_passed += 1
    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 4: Check HESTON_PARAMS
    print("\n[4] Checking HESTON_PARAMS...")
    try:
        assert hasattr(config, 'HESTON_PARAMS'), "HESTON_PARAMS not found"
        heston_keys = ['kappa', 'theta_v', 'sigma_v', 'rho', 'v0']
        missing = [k for k in heston_keys if k not in config.HESTON_PARAMS]

        if missing:
            raise AssertionError(f"Missing Heston keys: {missing}")

        print(f"    OK All Heston parameters present")
        print(f"       kappa={config.HESTON_PARAMS['kappa']}, theta_v={config.HESTON_PARAMS['theta_v']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 5: Check JUMP_PARAMS
    print("\n[5] Checking JUMP_PARAMS...")
    try:
        assert hasattr(config, 'JUMP_PARAMS'), "JUMP_PARAMS not found"
        jump_keys = ['lambda_jump', 'mu_jump', 'sigma_jump']
        missing = [k for k in jump_keys if k not in config.JUMP_PARAMS]

        if missing:
            raise AssertionError(f"Missing Jump keys: {missing}")

        print(f"    OK All Jump parameters present")
        print(f"       lambda={config.JUMP_PARAMS['lambda_jump']}, mu_jump={config.JUMP_PARAMS['mu_jump']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 6: Check MEAN_REVERSION_PARAMS
    print("\n[6] Checking MEAN_REVERSION_PARAMS...")
    try:
        assert hasattr(config, 'MEAN_REVERSION_PARAMS'), "MEAN_REVERSION_PARAMS not found"
        mr_keys = ['theta_mr', 'mu_0']
        missing = [k for k in mr_keys if k not in config.MEAN_REVERSION_PARAMS]

        if missing:
            raise AssertionError(f"Missing MR keys: {missing}")

        print(f"    OK All Mean Reversion parameters present")
        print(f"       theta_mr={config.MEAN_REVERSION_PARAMS['theta_mr']}, mu_0={config.MEAN_REVERSION_PARAMS['mu_0']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Test 7: Test get_time_to_expiry() function
    print("\n[7] Testing get_time_to_expiry() function...")
    try:
        assert hasattr(config, 'get_time_to_expiry'), "get_time_to_expiry function not found"

        T = config.get_time_to_expiry()
        assert isinstance(T, (int, float)), "T should be numeric"
        assert T >= 0, "T should be non-negative"
        print(f"    OK get_time_to_expiry() = {T:.6f} years")

        T_custom = config.get_time_to_expiry(time(16, 0))
        assert isinstance(T_custom, (int, float)), "T_custom should be numeric"
        print(f"    OK get_time_to_expiry(4PM) = {T_custom:.6f} years")

        tests_passed += 1
    except Exception as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(f"get_time_to_expiry error: {e}")

    # Test 8: Check other important constants
    print("\n[8] Checking constants...")
    try:
        constants = [
            'RISK_FREE_RATE',
            'TRADING_DAYS_PER_YEAR',
            'N_PATHS_DEFAULT',
            'N_STEPS_DEFAULT',
            'MIN_VARIANCE',
            'MAX_PRICE',
            'MIN_PRICE'
        ]

        missing_constants = [c for c in constants if not hasattr(config, c)]

        if missing_constants:
            print(f"    Warning: Missing constants: {missing_constants}")
        else:
            print(f"    OK All important constants present")

        if hasattr(config, 'RISK_FREE_RATE'):
            print(f"       RISK_FREE_RATE = {config.RISK_FREE_RATE}")
        if hasattr(config, 'N_PATHS_DEFAULT'):
            print(f"       N_PATHS_DEFAULT = {config.N_PATHS_DEFAULT}")

        tests_passed += 1
    except Exception as e:
        print(f"    FAIL {e}")
        tests_failed += 1
        failures.append(str(e))

    # Summary
    print("\n" + "="*60)
    print("TEST 2 RESULTS")
    print("="*60)
    print(f"Passed: {tests_passed}/8")
    print(f"Failed: {tests_failed}/8")

    if failures:
        print("\nFailures:")
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")

    if tests_failed > 0:
        print("\nTest 2: FAILED")
        return False
    else:
        print("\nTest 2: PASSED - Configuration is valid!")
        return True


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
