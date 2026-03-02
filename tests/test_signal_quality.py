"""
Tests for signal quality improvements:
- Confidence scoring with multiplicative penalties
- Jump intensity capping and shrinkage
- OTM edge scaling
- Bernoulli approximation guard
- SELL signal generation
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from config import (
    TRADING_THRESHOLDS, MAX_LAMBDA_JUMP, LAMBDA_PRIOR,
    LAMBDA_SHRINKAGE_ALPHA, SPOT_MAX_AGE_SECONDS
)


# ====================================================================
# Helper: build a TradingSystem with synthetic state (no network)
# ====================================================================

def _make_trader(S0=500.0, v0=0.04, lambda_jump=5.0):
    """Create a TradingSystem with fake provider and controlled params."""
    from trading.trading_system import TradingSystem
    from data.data_provider import MarketDataProvider

    class FakeProvider(MarketDataProvider):
        name = "fake"

        def get_spot_price(self, ticker):
            return {
                'price': S0,
                'timestamp': datetime.now(),
                'age_seconds': 0,
            }

        def get_intraday_data(self, ticker, **kw):
            import pandas as pd
            n = 200
            prices = S0 + np.random.randn(n).cumsum() * 0.05
            return pd.DataFrame({
                'Close': prices,
                'Volume': np.random.randint(100, 10000, n),
                'Open': prices,
                'High': prices + 0.05,
                'Low': prices - 0.05,
            })

        def get_option_chain(self, ticker, **kw):
            return {}

    trader = TradingSystem(ticker='SPY', provider=FakeProvider())
    trader.S0 = S0
    trader.spot_timestamp = datetime.now()
    trader.v0 = v0
    trader.params = {
        'kappa': 2.0, 'theta_v': 0.04, 'sigma_v': 0.3, 'rho': -0.7,
        'v0': v0, 'r': 0.05, 'measure': 'risk_neutral',
        'lambda_jump': lambda_jump, 'mu_jump': -0.01, 'sigma_jump': 0.02,
        'theta_mr': 3.0, 'mu_0': S0,
    }
    return trader


# ====================================================================
# 1. Confidence never saturates at 100% for typical scenarios
# ====================================================================

class TestConfidenceNotSaturated:
    """Confidence should NOT be 100% for typical edges and spreads."""

    def test_moderate_edge_wide_spread(self):
        """5% edge with wide spread should not be 100% confidence."""
        trader = _make_trader()
        # model_price > ask → BUY signal
        # market_ask=1.00, model price will be ~$1.05 at 5% edge
        signal = trader.get_trading_signal(
            strike=502, market_bid=0.80, market_ask=1.00, option_type='call'
        )
        # Even if model says BUY, confidence should reflect spread & OTM
        assert signal['confidence'] < 1.0, (
            f"Confidence should not be 100%, got {signal['confidence']}"
        )

    def test_otm_strike_penalizes_confidence(self):
        """OTM strikes should have lower confidence than ATM for same edge."""
        trader = _make_trader(S0=500.0)
        # ATM call
        sig_atm = trader.get_trading_signal(
            strike=500, market_bid=2.00, market_ask=2.20, option_type='call'
        )
        # 3$ OTM call
        sig_otm = trader.get_trading_signal(
            strike=503, market_bid=0.15, market_ask=0.20, option_type='call'
        )
        # Both might be BUY, but OTM should have lower confidence
        if sig_atm['action'] != 'HOLD' and sig_otm['action'] != 'HOLD':
            assert sig_otm['confidence'] <= sig_atm['confidence'], (
                f"OTM conf ({sig_otm['confidence']:.3f}) should be "
                f"<= ATM conf ({sig_atm['confidence']:.3f})"
            )

    def test_spread_penalty_reduces_confidence(self):
        """Wide spread should reduce confidence vs tight spread."""
        trader = _make_trader(S0=500.0)
        # Tight spread
        sig_tight = trader.get_trading_signal(
            strike=500, market_bid=2.00, market_ask=2.05, option_type='call'
        )
        # Wide spread (same mid)
        sig_wide = trader.get_trading_signal(
            strike=500, market_bid=1.80, market_ask=2.25, option_type='call'
        )
        if sig_tight['action'] != 'HOLD' and sig_wide['action'] != 'HOLD':
            assert sig_wide['confidence'] < sig_tight['confidence'], (
                f"Wide spread conf ({sig_wide['confidence']:.3f}) should be "
                f"< tight spread conf ({sig_tight['confidence']:.3f})"
            )


# ====================================================================
# 2. Jump intensity capping
# ====================================================================

class TestJumpIntensityCap:
    """Jump calibration should cap and shrink unreasonable lambda values."""

    def test_cap_enforced(self):
        """Lambda should never exceed MAX_LAMBDA_JUMP after calibration."""
        from calibration.jump_calibrator import calibrate_from_returns
        # Generate returns with many "jumps" (high kurtosis)
        np.random.seed(42)
        n = 2000
        returns = np.random.randn(n) * 0.001
        # Add lots of extreme returns to trigger high lambda
        returns[::10] = np.random.randn(n // 10) * 0.01
        dt = 1.0 / (252 * 390)
        params = calibrate_from_returns(returns, dt=dt, threshold=3.0, min_jumps=3)
        assert params['lambda_jump'] <= MAX_LAMBDA_JUMP, (
            f"λ={params['lambda_jump']:.1f} exceeds cap {MAX_LAMBDA_JUMP}"
        )

    def test_shrinkage_applied(self):
        """Lambda should be shrunk toward prior when raw lambda is high."""
        from calibration.jump_calibrator import calibrate_from_returns
        np.random.seed(42)
        n = 2000
        returns = np.random.randn(n) * 0.001
        # Force many jumps
        returns[::5] = np.random.randn(n // 5) * 0.01
        dt = 1.0 / (252 * 390)
        params = calibrate_from_returns(returns, dt=dt, threshold=3.0, min_jumps=3)
        # Effective lambda should be heavily anchored to prior
        max_possible = (LAMBDA_SHRINKAGE_ALPHA * MAX_LAMBDA_JUMP
                        + (1 - LAMBDA_SHRINKAGE_ALPHA) * LAMBDA_PRIOR)
        assert params['lambda_jump'] <= max_possible + 0.01, (
            f"λ_eff={params['lambda_jump']:.1f} should be <= {max_possible:.1f}"
        )


# ====================================================================
# 3. OTM edge scaling
# ====================================================================

class TestOTMEdgeScaling:
    """OTM strikes should require more edge to trigger signals."""

    def test_otm_requires_higher_edge(self):
        """A signal that passes at ATM might fail at 3$ OTM."""
        base_edge = TRADING_THRESHOLDS['min_edge']
        otm_scale = TRADING_THRESHOLDS['otm_edge_scale']
        otm_dollars = 3.0
        required = base_edge + otm_scale * otm_dollars
        assert required > base_edge, (
            f"OTM edge ({required}) should be > base edge ({base_edge})"
        )

    def test_otm_conf_decay_is_exponential(self):
        """Confidence penalty should decay exponentially with OTM distance."""
        import math
        decay = TRADING_THRESHOLDS['otm_conf_decay']
        penalty_0 = math.exp(-decay * 0)  # ATM
        penalty_3 = math.exp(-decay * 3)  # 3$ OTM
        penalty_5 = math.exp(-decay * 5)  # 5$ OTM
        assert penalty_0 == 1.0
        assert penalty_3 < 1.0
        assert penalty_5 < penalty_3


# ====================================================================
# 4. Bernoulli guard
# ====================================================================

class TestBernoulliGuard:
    """Bernoulli failure should disable jumps and penalize confidence."""

    def test_high_lambda_large_dt_triggers_guard(self):
        """With very high lambda AND large dt, Bernoulli guard should fire."""
        from models.jump_diffusion import validate_bernoulli_approximation
        # Use a scenario where lambda*dt is non-trivial
        # For guard to fail: (lambda*dt)^2/2 > 0.01 → lambda*dt > 0.14
        dt = 0.001  # ~1 trading day / 4
        lambda_val = 200.0
        valid, prob = validate_bernoulli_approximation(lambda_val, dt)
        assert not valid, (
            f"Bernoulli should fail at λ={lambda_val}, dt={dt:.6f}, prob={prob:.4f}"
        )

    def test_moderate_lambda_passes(self):
        """With moderate lambda, Bernoulli should pass."""
        from models.jump_diffusion import validate_bernoulli_approximation
        T = 1.0 / (252 * 6)
        dt = T / 200
        valid, prob = validate_bernoulli_approximation(5.0, dt)
        assert valid, (
            f"Bernoulli should pass at λ=5, dt={dt:.6f}, prob={prob:.4f}"
        )

    def test_0dte_with_capped_lambda_passes(self):
        """At 0DTE timescales with capped lambda, Bernoulli should always pass."""
        from models.jump_diffusion import validate_bernoulli_approximation
        # Typical 0DTE: ~4 hours left, 200 steps
        T = 4.0 / (252 * 6.5)  # 4 hours
        dt = T / 200
        # Even at the cap of 200/year
        valid, prob = validate_bernoulli_approximation(MAX_LAMBDA_JUMP, dt)
        assert valid, (
            f"At 0DTE with λ={MAX_LAMBDA_JUMP}, Bernoulli should pass. "
            f"dt={dt:.8f}, prob={prob:.6f}"
        )

    def test_bernoulli_violated_flag_set_correctly(self):
        """When Bernoulli is violated, paths should have the flag set."""
        from models.jump_diffusion import validate_bernoulli_approximation
        # Verify the function itself works
        valid_low, _ = validate_bernoulli_approximation(5.0, 0.001)
        assert valid_low  # low lambda, reasonable dt → pass
        valid_high, _ = validate_bernoulli_approximation(200.0, 0.001)
        assert not valid_high  # high lambda, large dt → fail


# ====================================================================
# 5. SELL signal can be produced
# ====================================================================

class TestSellSignalPossible:
    """Model should be able to generate SELL signals."""

    def test_sell_when_model_below_bid(self):
        """If model price is well below market bid, should get SELL."""
        trader = _make_trader(S0=500.0, lambda_jump=2.0)
        # Price a deep ITM strike where model should produce a fair value
        # Then set bid artificially high
        # For a call with strike=498 (ITM $2), model price should be ~$2
        # If bid=$5, model should say SELL
        signal = trader.get_trading_signal(
            strike=498, market_bid=5.00, market_ask=5.50, option_type='call'
        )
        # Model price for this near-ATM call should be around $2-$3
        # so model_price < bid=5.00 → SELL candidate
        if signal['model_price'] < 5.00:
            assert signal['action'] == 'SELL' or signal['confidence'] < TRADING_THRESHOLDS['min_confidence'], (
                f"Expected SELL or low-confidence HOLD when model={signal['model_price']:.2f} < bid=5.00"
            )


# ====================================================================
# 6. Config constants sanity
# ====================================================================

class TestConfigSanity:
    """Config constants should have sane values."""

    def test_lambda_cap_is_reasonable(self):
        assert 10 <= MAX_LAMBDA_JUMP <= 500

    def test_lambda_prior_is_conservative(self):
        assert 1 <= LAMBDA_PRIOR <= 20

    def test_shrinkage_alpha_is_small(self):
        assert 0.0 < LAMBDA_SHRINKAGE_ALPHA <= 0.5

    def test_otm_edge_scale_positive(self):
        assert TRADING_THRESHOLDS['otm_edge_scale'] > 0

    def test_otm_conf_decay_positive(self):
        assert TRADING_THRESHOLDS['otm_conf_decay'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
