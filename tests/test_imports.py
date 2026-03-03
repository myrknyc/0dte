
import sys
import os
import traceback

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all():
    print("="*60)
    print("TEST 1: IMPORT VERIFICATION")
    print("="*60)

    packages_to_test = [
        ('config', lambda: __import__('config')),
        ('pricing.black_scholes', lambda: __import__('pricing.black_scholes', fromlist=['black_scholes'])),
        ('data.data_loader', lambda: __import__('data.data_loader', fromlist=['DataLoader'])),
        ('data.streaming_data_feed', lambda: __import__('data.streaming_data_feed', fromlist=['StreamingDataFeed'])),
        ('signals.signal_logger', lambda: __import__('signals.signal_logger', fromlist=['SignalLogger'])),
        ('trading.trading_system', lambda: __import__('trading.trading_system', fromlist=['TradingSystem'])),
        ('trading.risk_manager', lambda: __import__('trading.risk_manager', fromlist=['FlexibleRiskManager'])),
        ('core.time_grid', lambda: __import__('core.time_grid', fromlist=['generate_time_grid'])),
        ('core.random_numbers', lambda: __import__('core.random_numbers', fromlist=['generate_correlated_normals'])),
        ('models.heston', lambda: __import__('models.heston', fromlist=['simulate_variance_paths'])),
        ('models.jump_diffusion', lambda: __import__('models.jump_diffusion', fromlist=['generate_jumps_vectorized'])),
        ('models.mean_reversion', lambda: __import__('models.mean_reversion', fromlist=['compute_mu_t'])),
        ('models.combined_model', lambda: __import__('models.combined_model', fromlist=['simulate_combined_paths_fast'])),
        ('pricing.european', lambda: __import__('pricing.european', fromlist=['price_european_option'])),
        ('pricing.greeks', lambda: __import__('pricing.greeks', fromlist=['compute_all_greeks'])),
        ('pricing.american', lambda: __import__('pricing.american', fromlist=['price_american_lsm'])),
        ('calibration.heston_calibrator', lambda: __import__('calibration.heston_calibrator', fromlist=['calibrate_to_realized_vol'])),
        ('calibration.jump_calibrator', lambda: __import__('calibration.jump_calibrator', fromlist=['calibrate_from_returns'])),
        ('calibration.mean_reversion_calibrator', lambda: __import__('calibration.mean_reversion_calibrator', fromlist=['calibrate_from_intraday'])),
        ('variance_reduction.control_variates', lambda: __import__('variance_reduction.control_variates', fromlist=['apply_control_variate'])),
        ('variance_reduction.antithetic', lambda: __import__('variance_reduction.antithetic', fromlist=['verify_antithetic_correlation'])),
        ('variance_reduction.moment_matching', lambda: __import__('variance_reduction.moment_matching', fromlist=['verify_moment_matching'])),
    ]

    failed_imports = []
    passed_imports = []

    for name, import_func in packages_to_test:
        try:
            import_func()
            print(f"OK  {name}")
            passed_imports.append(name)
        except Exception as e:
            print(f"FAIL {name}")
            print(f"     Error: {e}")
            failed_imports.append((name, str(e)))
            traceback.print_exc()

    print("\n" + "="*60)
    print("IMPORT TEST RESULTS")
    print("="*60)
    print(f"Passed: {len(passed_imports)}/{len(packages_to_test)}")
    print(f"Failed: {len(failed_imports)}/{len(packages_to_test)}")

    if failed_imports:
        print("\nFailed Imports:")
        for name, error in failed_imports:
            print(f"  - {name}: {error}")
        print("\nTest 1: FAILED")
        return False
    else:
        print("\nTest 1: PASSED - All imports successful!")
        return True


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
