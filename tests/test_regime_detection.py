"""
Phase 2 tests — Regime detection, thresholds, and jump blending.

Tests:
  1. Regime classification buckets match canonical values
  2. Threshold overlay applies correct values per regime
  3. Jump blending formula and weight clamping
  4. Canonical buckets imported by backtest_metrics
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRegimeClassification:
    """Test classify() matches canonical bucket boundaries."""

    def test_calm(self):
        from calibration.regime_detector import classify
        assert classify(0.0) == 'calm'
        assert classify(0.2) == 'calm'
        assert classify(0.49) == 'calm'

    def test_normal(self):
        from calibration.regime_detector import classify
        assert classify(0.5) == 'normal'
        assert classify(0.75) == 'normal'
        assert classify(0.99) == 'normal'

    def test_volatile(self):
        from calibration.regime_detector import classify
        assert classify(1.0) == 'volatile'
        assert classify(1.5) == 'volatile'
        assert classify(1.99) == 'volatile'

    def test_extreme(self):
        from calibration.regime_detector import classify
        assert classify(2.0) == 'extreme'
        assert classify(5.0) == 'extreme'
        assert classify(10.0) == 'extreme'

    def test_negative_uses_abs(self):
        from calibration.regime_detector import classify
        assert classify(-0.3) == 'calm'
        assert classify(-1.5) == 'volatile'

    def test_none_returns_unknown(self):
        from calibration.regime_detector import classify
        assert classify(None) == 'unknown'

    def test_nan_returns_unknown(self):
        from calibration.regime_detector import classify
        assert classify(float('nan')) == 'unknown'


class TestAdjustedThresholds:
    """Test regime-conditioned filter overlays."""

    def test_calm_thresholds(self):
        from calibration.regime_detector import get_adjusted_thresholds
        t = get_adjusted_thresholds('calm')
        assert t['min_edge'] == 0.015
        assert t['min_confidence'] == 0.45
        assert t['max_cvar_loss'] == -1.50
        assert t['min_eu'] == 0.00

    def test_extreme_thresholds(self):
        from calibration.regime_detector import get_adjusted_thresholds
        t = get_adjusted_thresholds('extreme')
        assert t['min_edge'] == 0.050
        assert t['min_confidence'] == 0.70
        assert t['max_cvar_loss'] == -3.00
        assert t['min_eu'] == 0.05

    def test_unknown_falls_back_to_normal(self):
        from calibration.regime_detector import get_adjusted_thresholds
        t = get_adjusted_thresholds('unknown')
        t_normal = get_adjusted_thresholds('normal')
        assert t == t_normal

    def test_thresholds_monotonic(self):
        """min_edge should increase with volatility."""
        from calibration.regime_detector import get_adjusted_thresholds
        edges = [get_adjusted_thresholds(r)['min_edge']
                 for r in ['calm', 'normal', 'volatile', 'extreme']]
        assert edges == sorted(edges)


class TestJumpBlending:
    """Test regime-adaptive jump parameter blending."""

    def test_blend_weight_boundaries(self):
        """Weight is clamped to [0.2, 0.9]."""
        from calibration.regime_detector import regime_blend_weight
        # Strong calibration: many returns, stable variance
        w = regime_blend_weight(n_returns=100, variance_stability=1.0)
        assert 0.2 <= w <= 0.9

        # Weak calibration: few returns, unstable variance
        w = regime_blend_weight(n_returns=2, variance_stability=3.0)
        assert w == 0.2  # floor

    def test_more_data_higher_weight(self):
        from calibration.regime_detector import regime_blend_weight
        w_few = regime_blend_weight(n_returns=5, variance_stability=1.0)
        w_many = regime_blend_weight(n_returns=100, variance_stability=1.0)
        assert w_many > w_few

    def test_stable_variance_higher_weight(self):
        from calibration.regime_detector import regime_blend_weight
        w_unstable = regime_blend_weight(n_returns=50, variance_stability=3.0)
        w_stable = regime_blend_weight(n_returns=50, variance_stability=1.0)
        assert w_stable > w_unstable

    def test_blend_between_calibrated_and_prior(self):
        """Blended λ must be between calibrated and prior values."""
        from calibration.regime_detector import blend_jump_params
        calibrated = {'lambda_jump': 15.0, 'mu_jump': -0.04, 'sigma_jump': 0.06}
        result = blend_jump_params(
            calibrated, regime='calm',
            n_returns=50, variance_stability=1.0
        )
        # calm prior λ = 2.0, calibrated = 15.0
        # blended should be between 2.0 and 15.0
        assert 2.0 <= result['lambda_jump'] <= 15.0

    def test_blend_with_zero_returns(self):
        """Zero returns → w at floor, mostly prior."""
        from calibration.regime_detector import blend_jump_params
        calibrated = {'lambda_jump': 100.0, 'mu_jump': -0.10, 'sigma_jump': 0.20}
        result = blend_jump_params(
            calibrated, regime='normal',
            n_returns=0, variance_stability=1.0
        )
        # With n_returns=0, w=0.2 (floor), so result ≈ 0.2*100 + 0.8*5 = 24
        # It should be much closer to the prior than to 100
        assert result['lambda_jump'] < 30.0

    def test_jump_prior_keys(self):
        """Jump priors have expected keys."""
        from calibration.regime_detector import get_jump_prior
        for regime in ['calm', 'normal', 'volatile', 'extreme']:
            prior = get_jump_prior(regime)
            assert 'lambda_jump' in prior
            assert 'mu_jump' in prior
            assert 'sigma_jump' in prior


class TestCanonicalBucketConsistency:
    """Verify backtest_metrics uses the same buckets as regime_detector."""

    def test_same_buckets(self):
        from calibration.regime_detector import REGIME_BUCKETS as canonical
        from trading.backtest_metrics import REGIME_BUCKETS as analytics
        assert canonical == list(analytics)


class TestPayoffDistributionStats:
    """Test that european.py returns payoff distribution stats."""

    def test_pricer_returns_stats(self):
        import numpy as np
        from pricing.european import price_european_option

        np.random.seed(42)
        S0 = 590.0
        n_paths = 1000
        n_steps = 10
        S_paths = np.full((n_paths, n_steps + 1), S0)
        # Simple random walk
        for i in range(1, n_steps + 1):
            S_paths[:, i] = S_paths[:, i-1] * np.exp(
                np.random.normal(-0.0001, 0.005, n_paths)
            )

        result = price_european_option(
            S_paths=S_paths, K=590.0, T=1/252, r=0.05,
            option_type='call'
        )

        assert 'payoff_mean_pos' in result
        assert 'payoff_mean_zero' in result
        assert 'payoff_frac_pos' in result
        assert result['payoff_frac_pos'] >= 0.0
        assert result['payoff_frac_pos'] <= 1.0
        assert result['payoff_mean_pos'] >= 0.0
        assert result['payoff_mean_zero'] >= 0.0


class TestRegimeExitParams:
    """Test regime-conditioned TP/SL exit parameters (H4 extension)."""

    def test_calm_tighter_exits(self):
        from calibration.regime_detector import get_exit_params
        p = get_exit_params('calm')
        assert p['tp_pct'] == 0.15
        assert p['sl_pct'] == 0.12

    def test_extreme_wider_exits(self):
        from calibration.regime_detector import get_exit_params
        p = get_exit_params('extreme')
        assert p['tp_pct'] == 0.40
        assert p['sl_pct'] == 0.35

    def test_unknown_uses_defaults(self):
        from calibration.regime_detector import get_exit_params
        p = get_exit_params('unknown', default_tp=0.22, default_sl=0.18)
        assert p['tp_pct'] == 0.22
        assert p['sl_pct'] == 0.18

    def test_tp_sl_monotonic_with_volatility(self):
        """TP and SL should widen as regime volatility increases."""
        from calibration.regime_detector import get_exit_params
        regimes = ['calm', 'normal', 'volatile', 'extreme']
        tps = [get_exit_params(r)['tp_pct'] for r in regimes]
        sls = [get_exit_params(r)['sl_pct'] for r in regimes]
        assert tps == sorted(tps), f"TP not monotonic: {tps}"
        assert sls == sorted(sls), f"SL not monotonic: {sls}"

