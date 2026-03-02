"""
Tests for P0 spot freshness fixes.

Covers:
1. get_spot_price() returns dict with price, timestamp, age_seconds
2. TradingSystem.get_trading_signal() returns HOLD when spot is stale
3. Path cache invalidated when S0 moves > threshold
4. Path cache reused when S0 stays within threshold
5. Signal logger accepts and stores spot_timestamp + spot_age_seconds
"""

import sys
import os
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SPOT_MAX_AGE_SECONDS, SPOT_CHANGE_THRESHOLD


# ====================================================================
# 1. Data Provider: get_spot_price returns dict
# ====================================================================
class TestSpotReturnType:
    """get_spot_price() should return a dict with required keys."""

    def test_yfinance_returns_dict(self):
        """YFinanceProvider.get_spot_price returns dict with expected keys."""
        from data.data_provider import YFinanceProvider

        # Mock yfinance to avoid network dependency
        with patch('data.data_provider.YFinanceProvider.__init__', return_value=None):
            provider = YFinanceProvider.__new__(YFinanceProvider)
            provider._yf = MagicMock()

            mock_ticker = MagicMock()
            mock_ticker.fast_info = {'lastPrice': 590.50}
            provider._yf.Ticker.return_value = mock_ticker

            result = provider.get_spot_price('SPY')

        assert isinstance(result, dict)
        assert 'price' in result
        assert 'timestamp' in result
        assert 'age_seconds' in result
        assert result['price'] == 590.50
        assert isinstance(result['timestamp'], datetime)
        assert result['age_seconds'] >= 0


# ====================================================================
# 2. Freshness Gate: stale spot → HOLD
# ====================================================================
class TestFreshnessGate:
    """Signals should be suppressed when spot data is stale."""

    def _make_trader(self):
        """Create a TradingSystem with mock provider (no network)."""
        from trading.trading_system import TradingSystem
        from data.data_provider import MarketDataProvider

        class FakeProvider(MarketDataProvider):
            name = "fake"
            def get_spot_price(self, ticker):
                return {
                    'price': 590.0,
                    'timestamp': datetime.now(),
                    'age_seconds': 0.0,
                }
            def get_intraday_data(self, ticker, **kw):
                import pandas as pd
                idx = pd.date_range('2025-01-01 09:30', periods=100, freq='1min')
                data = pd.DataFrame({
                    'Open': 590.0,
                    'High': 591.0,
                    'Low': 589.0,
                    'Close': np.linspace(589, 591, 100),
                    'Volume': 1000,
                }, index=idx)
                return data
            def get_option_chain(self, ticker, **kw):
                return {'calls': None, 'puts': None, 'expiry_date': '2025-01-01'}

        trader = TradingSystem('SPY', provider=FakeProvider())
        # Set up minimal state without full calibration
        trader.S0 = 590.0
        trader.v0 = 0.04
        trader.params.update({
            'kappa': 2.0, 'theta_v': 0.04, 'sigma_v': 0.3,
            'rho': -0.7, 'r': 0.05, 'lambda_jump': 1.0,
            'mu_jump': -0.01, 'sigma_jump': 0.02,
            'measure': 'risk_neutral',
        })
        return trader

    def test_fresh_spot_allows_signal(self):
        """A fresh timestamp should allow normal signal generation."""
        trader = self._make_trader()
        trader.spot_timestamp = datetime.now()  # fresh

        signal = trader.get_trading_signal(590, 1.50, 1.60, 'call')
        # Signal should NOT be stale-hold
        assert 'stale' not in signal['reason'].lower() or signal['action'] != 'HOLD'

    def test_stale_spot_returns_hold(self):
        """A stale timestamp should suppress signals."""
        trader = self._make_trader()
        # Set timestamp far in the past
        trader.spot_timestamp = datetime.now() - timedelta(seconds=SPOT_MAX_AGE_SECONDS + 60)

        signal = trader.get_trading_signal(590, 1.50, 1.60, 'call')
        assert signal['action'] == 'HOLD'
        assert 'stale' in signal['reason'].lower()
        assert signal['spot_age_seconds'] > SPOT_MAX_AGE_SECONDS

    def test_none_timestamp_treated_as_stale(self):
        """Missing timestamp (None) should be treated as stale."""
        trader = self._make_trader()
        trader.spot_timestamp = None

        signal = trader.get_trading_signal(590, 1.50, 1.60, 'call')
        assert signal['action'] == 'HOLD'
        assert 'stale' in signal['reason'].lower()


# ====================================================================
# 3. Cache Invalidation on Spot Change
# ====================================================================
class TestCacheInvalidation:
    """Path cache should be invalidated when S0 moves significantly."""

    def _make_trader(self):
        from trading.trading_system import TradingSystem
        from data.data_provider import MarketDataProvider

        class FakeProvider(MarketDataProvider):
            name = "fake"
            def get_spot_price(self, ticker):
                return {
                    'price': 590.0,
                    'timestamp': datetime.now(),
                    'age_seconds': 0.0,
                }
            def get_intraday_data(self, ticker, **kw):
                import pandas as pd
                idx = pd.date_range('2025-01-01 09:30', periods=100, freq='1min')
                return pd.DataFrame({
                    'Open': 590.0, 'High': 591.0, 'Low': 589.0,
                    'Close': np.linspace(589, 591, 100), 'Volume': 1000,
                }, index=idx)
            def get_option_chain(self, ticker, **kw):
                return {'calls': None, 'puts': None, 'expiry_date': '2025-01-01'}

        trader = TradingSystem('SPY', provider=FakeProvider())
        trader.S0 = 590.0
        trader.spot_timestamp = datetime.now()
        trader.v0 = 0.04
        trader.params.update({
            'kappa': 2.0, 'theta_v': 0.04, 'sigma_v': 0.3,
            'rho': -0.7, 'r': 0.05, 'lambda_jump': 1.0,
            'mu_jump': -0.01, 'sigma_jump': 0.02,
            'measure': 'risk_neutral',
        })
        return trader

    def test_cache_reused_when_spot_stable(self):
        """Cache should be reused when spot hasn't moved."""
        trader = self._make_trader()
        T = 0.001  # small T

        paths1 = trader._generate_paths(T)
        paths2 = trader._generate_paths(T)

        # Same object reference means cache was reused
        assert paths1 is paths2

    def test_cache_invalidated_on_big_spot_move(self):
        """Cache should be invalidated when spot moves > threshold."""
        trader = self._make_trader()
        T = 0.001

        paths1 = trader._generate_paths(T)

        # Move spot by more than SPOT_CHANGE_THRESHOLD (0.1%)
        # 590 * 0.001 = 0.59, so move at least 0.60
        trader.S0 = 591.0  # ~0.17% move, above threshold

        paths2 = trader._generate_paths(T)

        # Different object reference means cache was regenerated
        assert paths1 is not paths2

    def test_cache_kept_on_small_spot_move(self):
        """Cache should be kept when spot moves < threshold."""
        trader = self._make_trader()
        T = 0.001

        paths1 = trader._generate_paths(T)

        # Move spot by less than SPOT_CHANGE_THRESHOLD (0.1%)
        # 590 * 0.001 = 0.59, so move by 0.01
        trader.S0 = 590.01  # ~0.002% move, below threshold

        paths2 = trader._generate_paths(T)

        # Same object reference means cache was reused
        assert paths1 is paths2


# ====================================================================
# 4. update_spot() convenience method
# ====================================================================
class TestUpdateSpot:
    """update_spot() should fetch, update, and conditionally invalidate."""

    def test_update_spot_updates_fields(self):
        from trading.trading_system import TradingSystem
        from data.data_provider import MarketDataProvider

        class FakeProvider(MarketDataProvider):
            name = "fake"
            def get_spot_price(self, ticker):
                return {
                    'price': 592.0,
                    'timestamp': datetime.now(),
                    'age_seconds': 0.1,
                }
            def get_intraday_data(self, ticker, **kw):
                import pandas as pd
                return pd.DataFrame()
            def get_option_chain(self, ticker, **kw):
                return {}

        trader = TradingSystem('SPY', provider=FakeProvider())
        trader.S0 = 590.0
        trader.spot_timestamp = datetime.now() - timedelta(seconds=30)

        result = trader.update_spot()

        assert trader.S0 == 592.0
        assert trader.spot_timestamp is not None
        assert result['price'] == 592.0


# ====================================================================
# 5. Signal Logger: new columns
# ====================================================================
class TestSignalLoggerTimestamps:
    """Signal logger should accept and store spot_timestamp + spot_age_seconds."""

    def test_log_signal_with_timestamps(self, tmp_path):
        from signals.signal_logger import SignalLogger

        db_path = str(tmp_path / "test_signals.db")
        logger = SignalLogger(db_path=db_path)

        ts = datetime.now()
        row_id = logger.log_signal(
            ticker='SPY', strike=590, option_type='call',
            action='BUY', edge=0.05, confidence=0.8,
            model_price=2.50, market_bid=2.30, market_ask=2.40,
            market_mid=2.35, spread=0.10, std_error=0.02,
            spot_price=590.0, time_to_expiry=0.001, iv=0.20,
            reason='test', source='test',
            spot_timestamp=ts, spot_age_seconds=1.5,
        )

        # Retrieve and verify
        signals = logger.get_signals()
        assert len(signals) == 1
        row = signals[0]
        assert row['spot_timestamp'] is not None
        assert row['spot_age_seconds'] == 1.5
        logger.close()

    def test_log_signal_without_timestamps(self, tmp_path):
        """Backward compatibility: omitting new fields should still work."""
        from signals.signal_logger import SignalLogger

        db_path = str(tmp_path / "test_signals_compat.db")
        logger = SignalLogger(db_path=db_path)

        row_id = logger.log_signal(
            ticker='SPY', strike=590, option_type='call',
            action='HOLD', edge=0.0, confidence=0.0,
            model_price=2.50, market_bid=2.30, market_ask=2.40,
            market_mid=2.35, spread=0.10, std_error=0.02,
            spot_price=590.0, time_to_expiry=0.001, iv=0.20,
            reason='test', source='test',
        )

        signals = logger.get_signals()
        assert len(signals) == 1
        row = signals[0]
        assert row['spot_timestamp'] is None
        assert row['spot_age_seconds'] is None
        logger.close()


# ====================================================================
# 6. Config constants present
# ====================================================================
class TestConfigConstants:
    """New config constants should be importable and have sane values."""

    def test_spot_max_age_exists(self):
        assert SPOT_MAX_AGE_SECONDS > 0

    def test_spot_change_threshold_exists(self):
        assert 0 < SPOT_CHANGE_THRESHOLD < 0.1  # between 0% and 10%
