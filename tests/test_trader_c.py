"""
Track C (buy-only) tests — puts, SELL rejection, tuple-key quotes.

Tests:
  1. Track C rejects SELL signals
  2. Track C accepts BUY calls
  3. Track C accepts BUY puts
  4. Track C respects its own position limits
  5. _get_quote works with tuple keys
  6. _get_quote falls back to strike-only keys
  7. _record_quotes handles tuple keys
  8. Put signal EU direction (BUY put: G when put payoff > ask)
"""

import os
import copy
import tempfile
import pytest
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.paper_journal import PaperJournal, to_utc_str
from trading.paper_trader import PaperTrader
from trading.paper_config import PAPER_TRADING

_ET = ZoneInfo('America/New_York')
_UTC = ZoneInfo('UTC')


# ── Fixtures ──

@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def journal(tmp_db):
    j = PaperJournal(db_path=tmp_db)
    yield j
    j.close()


@pytest.fixture
def config():
    cfg = copy.deepcopy(PAPER_TRADING)
    cfg['decision_times'] = ['10:00']
    cfg['max_trades_per_decision'] = 2
    cfg['max_open_positions'] = 5
    cfg['slippage_mode'] = 'pct_of_spread'
    cfg['slippage_pct'] = 0.10
    cfg['commission_per_contract'] = 0.65
    cfg['exchange_fees_per_contract'] = 0.05
    cfg['buy_only'] = {
        'enabled': True,
        'enter_on': 'decision_time',
        'action_filter': 'BUY',
        'option_types': ['call', 'put'],
        'max_open_positions': 3,
        'max_trades_per_decision': 1,
        'use_eu_scoring': True,
        'use_regime_thresholds': False,
        'selection_policy': 'eu_ranked',
        'filters': {
            'min_confidence': 0.50,
            'min_edge': 0.02,
            'max_spread_pct': 0.40,
            'min_option_mid': 0.05,
            'max_spot_age_seconds': 10,
            'skip_bernoulli_violated': True,
            'max_otm_dollars': 5.0,
            'min_eu': 0.0,
        },
    }
    return cfg


def _make_signal(strike=590.0, action='BUY', confidence=0.80, edge=0.05,
                 bid=2.00, ask=2.20, option_type='call', spot_age=3,
                 otm_dollars=1.0, **kwargs):
    """Create a mock signal dict with payoff stats for EU scoring."""
    sig = {
        'strike': strike,
        'action': action,
        'confidence': confidence,
        'edge': edge,
        'required_edge': 0.02,
        'market_bid': bid,
        'market_ask': ask,
        'market_mid': (bid + ask) / 2.0,
        'model_price': (bid + ask) / 2.0 * (1 + edge),
        'std_error': 0.01,
        'market_iv': 0.25,
        'spot_age_seconds': spot_age,
        'otm_dollars': otm_dollars,
        'bernoulli_violated': False,
        'ticker': 'SPY',
        'option_type': option_type,
        'spread': ask - bid,
        'payoff_mean_pos': ask * 1.3,   # 30% above ask → positive EU
        'payoff_mean_zero': 0.0,
        'payoff_frac_pos': 0.6,
        'cvar_95': -0.50,
    }
    sig.update(kwargs)
    return sig


# ═══════════════════════════════════════════════════════════════
#  Track C: Buy-Only Behavior
# ═══════════════════════════════════════════════════════════════

class TestTrackCBuyOnly:

    def test_sell_rejected(self, journal, config):
        """Track C rejects SELL signals."""
        trader = PaperTrader(journal=journal, config=config)

        signals = [_make_signal(action='SELL', option_type='call')]
        qm = {(590.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2}}
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 0

    def test_buy_call_accepted(self, journal, config):
        """Track C accepts BUY call signals."""
        trader = PaperTrader(journal=journal, config=config)

        signals = [_make_signal(action='BUY', option_type='call')]
        qm = {(590.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2}}
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 1

    def test_buy_put_accepted(self, journal, config):
        """Track C accepts BUY put signals."""
        trader = PaperTrader(journal=journal, config=config)

        signals = [_make_signal(action='BUY', option_type='put')]
        qm = {(590.0, 'put'): {'bid': 1.5, 'ask': 1.7, 'spot': 590.0, 'spot_age': 2}}
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 1
        # Verify it's a put
        trade = list(trader._open_trades_c.values())[0]
        assert trade['option_type'] == 'put'

    def test_hold_rejected(self, journal, config):
        """Track C rejects HOLD signals."""
        trader = PaperTrader(journal=journal, config=config)

        signals = [_make_signal(action='HOLD')]
        qm = {(590.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2}}
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 0

    def test_position_limit(self, journal, config):
        """Track C enforces its own max_open_positions."""
        config['buy_only']['max_open_positions'] = 2
        config['buy_only']['max_trades_per_decision'] = 5
        trader = PaperTrader(journal=journal, config=config)

        signals = [
            _make_signal(strike=590.0, action='BUY', option_type='call'),
            _make_signal(strike=591.0, action='BUY', option_type='call'),
            _make_signal(strike=592.0, action='BUY', option_type='call'),
        ]
        qm = {
            (590.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2},
            (591.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2},
            (592.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590.0, 'spot_age': 2},
        }
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 2  # not 3

    def test_option_type_filter(self, journal, config):
        """Track C respects option_types whitelist."""
        config['buy_only']['option_types'] = ['call']  # puts disabled
        trader = PaperTrader(journal=journal, config=config)

        signals = [_make_signal(action='BUY', option_type='put')]
        qm = {(590.0, 'put'): {'bid': 1.5, 'ask': 1.7, 'spot': 590.0, 'spot_age': 2}}
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)

        trader._make_decision_c(signals, qm, 590.0, ts, to_utc_str(ts))
        assert len(trader._open_trades_c) == 0


# ═══════════════════════════════════════════════════════════════
#  Quote Map: Tuple-Key Support
# ═══════════════════════════════════════════════════════════════

class TestQuoteMapTupleKeys:

    def test_get_quote_tuple_key(self, journal, config):
        """_get_quote finds quote by (strike, option_type) tuple."""
        trader = PaperTrader(journal=journal, config=config)
        qm = {
            (590.0, 'call'): {'bid': 2.0, 'ask': 2.2},
            (590.0, 'put'): {'bid': 1.5, 'ask': 1.7},
        }
        call_q = trader._get_quote(qm, 590.0, 'call')
        put_q = trader._get_quote(qm, 590.0, 'put')
        assert call_q['bid'] == 2.0
        assert put_q['bid'] == 1.5

    def test_get_quote_fallback_to_strike_only(self, journal, config):
        """_get_quote falls back to strike-only key for backward compat."""
        trader = PaperTrader(journal=journal, config=config)
        qm = {590.0: {'bid': 2.0, 'ask': 2.2}}  # old-style
        q = trader._get_quote(qm, 590.0, 'call')
        assert q['bid'] == 2.0

    def test_get_quote_returns_none(self, journal, config):
        """_get_quote returns None for missing quotes."""
        trader = PaperTrader(journal=journal, config=config)
        qm = {(590.0, 'call'): {'bid': 2.0, 'ask': 2.2}}
        q = trader._get_quote(qm, 591.0, 'call')
        assert q is None

    def test_record_quotes_tuple_keys(self, journal, config):
        """_record_quotes handles (strike, opt_type) keys."""
        trader = PaperTrader(journal=journal, config=config)
        qm = {
            (590.0, 'call'): {'bid': 2.0, 'ask': 2.2, 'spot': 590, 'spot_age': 1},
            (590.0, 'put'): {'bid': 1.5, 'ask': 1.7, 'spot': 590, 'spot_age': 1},
        }
        ts = to_utc_str(datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET))
        # Should not raise
        trader._record_quotes(qm, 590.0, ts)


# ═══════════════════════════════════════════════════════════════
#  EU Direction: Put BUY
# ═══════════════════════════════════════════════════════════════

class TestPutEUDirection:

    def test_buy_put_eu_computes(self, journal, config):
        """BUY put signal has meaningful EU value."""
        trader = PaperTrader(journal=journal, config=config)
        sig = _make_signal(
            action='BUY', option_type='put',
            confidence=0.7,
            payoff_mean_pos=1.80,  # put payoff when ITM
            payoff_mean_zero=0.0,
            bid=1.00, ask=1.20,
        )
        eu = trader._compute_eu(sig)
        assert isinstance(eu, float)
        # G = 1.80 - 1.20 = 0.60, L = 1.20 - 0.0 = 1.20
        # EU = 0.7 * 0.60 - 0.3 * 1.20 - C ≈ 0.42 - 0.36 - C ≈ small positive
        # The exact value depends on cost, but it should compute without error

    def test_put_eu_vs_call_eu_different(self, journal, config):
        """Same stats but different option types should give same EU (both are BUY)."""
        trader = PaperTrader(journal=journal, config=config)
        call_sig = _make_signal(action='BUY', option_type='call',
                                payoff_mean_pos=2.0, bid=1.50, ask=1.60)
        put_sig = _make_signal(action='BUY', option_type='put',
                               payoff_mean_pos=2.0, bid=1.50, ask=1.60)
        # EU formula doesn't depend on option_type, only on action + payoff stats
        eu_call = trader._compute_eu(call_sig)
        eu_put = trader._compute_eu(put_sig)
        assert eu_call == eu_put  # same inputs → same EU


# ═══════════════════════════════════════════════════════════════
#  Track C Independent from Track A
# ═══════════════════════════════════════════════════════════════

class TestTrackCIndependence:

    def test_track_c_independent_from_track_a(self, journal, config):
        """Track C open positions don't count against Track A limits."""
        config['max_open_positions'] = 1  # Track A limit: 1
        config['buy_only']['max_open_positions'] = 2
        trader = PaperTrader(journal=journal, config=config)

        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(ts)

        # Enter 2 Track C trades
        sig1 = _make_signal(strike=590.0, action='BUY')
        sig2 = _make_signal(strike=591.0, action='BUY')
        trader._enter_trade(sig1, 590.0, ts_utc, 'buy_only')
        trader._enter_trade(sig2, 590.0, ts_utc, 'buy_only')
        assert len(trader._open_trades_c) == 2

        # Track A should still have its own slots
        sig3 = _make_signal(strike=592.0, action='BUY')
        trader._enter_trade(sig3, 590.0, ts_utc, 'decision_time')
        assert len(trader._open_trades_a) == 1

    def test_track_c_close_removes_from_c(self, journal, config):
        """Closing Track C trade removes from _open_trades_c, not _a or _b."""
        trader = PaperTrader(journal=journal, config=config)
        ts = datetime(2026, 3, 3, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(ts)

        sig = _make_signal(strike=590.0, action='BUY')
        trader._enter_trade(sig, 590.0, ts_utc, 'buy_only')
        assert len(trader._open_trades_c) == 1

        # Close via exit check with a quote
        qm = {(590.0, 'call'): {'bid': 2.3, 'ask': 2.5, 'spot': 591.0, 'spot_age': 2}}
        exit_ts = ts + timedelta(minutes=35)  # past exit_time_minutes
        trader._check_exits(qm, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_c) == 0
        assert len(trader._open_trades_a) == 0  # untouched
