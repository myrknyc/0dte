"""
Paper Trading — test suite.

28 tests covering fills, PnL, exits, eligibility, selection, dedup,
decision-time logic, Track B behavior, and EOD reporting.
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

from trading.fill_model import simulate_fill, compute_fees, compute_pnl
from trading.paper_journal import PaperJournal, to_utc_str
from trading.paper_trader import PaperTrader
from trading.paper_config import PAPER_TRADING
from trading.eod_reporter import EODReporter

_ET = ZoneInfo('America/New_York')
_UTC = ZoneInfo('UTC')


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary database and return its path."""
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
    """Test config with modified settings for faster tests."""
    import copy
    cfg = copy.deepcopy(PAPER_TRADING)
    cfg['decision_times'] = ['10:00']
    cfg['max_trades_per_decision'] = 2
    cfg['max_open_positions'] = 5
    cfg['exit_time_minutes'] = 5
    cfg['tp_pct'] = 0.20
    cfg['sl_pct'] = 0.15
    cfg['slippage_mode'] = 'pct_of_spread'
    cfg['slippage_pct'] = 0.10
    cfg['commission_per_contract'] = 0.65
    cfg['exchange_fees_per_contract'] = 0.05
    cfg['cooldown_minutes'] = 5
    cfg['cooldown_scope'] = 'same_direction'
    cfg['min_strike_spacing'] = 1.0
    cfg['all_signals'] = {
        'enabled': True,
        'enter_on': 'every_scan',
        'apply_filters': {
            'use_spot_age': True,
            'use_spread_pct': False,
            'use_min_option_mid': False,
        },
        'dedup_policy': 'one_open_per_strike_action',
        'cooldown_minutes': 0,
        'exit_mode': 'fixed_horizon',
        'exit_horizon_minutes': 5,
        'eod_exit_time': '15:55',
    }
    return cfg


def _make_signal(strike=590.0, action='BUY', confidence=0.80, edge=0.05,
                 bid=2.00, ask=2.20, spot_age=3, otm_dollars=1.0,
                 bernoulli_violated=False, **kwargs):
    """Create a mock signal dict."""
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
        'bernoulli_violated': bernoulli_violated,
        'ticker': 'SPY',
        'option_type': 'call',
        'spread': ask - bid,
        # Payoff distribution stats for EU scoring
        'payoff_mean_pos': ask * 1.3,   # 30% above ask → profitable
        'payoff_mean_zero': 0.0,
        'payoff_frac_pos': 0.6,
    }
    sig.update(kwargs)
    return sig


def _make_quote_map(*strikes_with_quotes):
    """Build a quote_map. Each arg: (strike, bid, ask, spot=590, spot_age=2)."""
    qm = {}
    for item in strikes_with_quotes:
        strike, bid, ask = item[0], item[1], item[2]
        spot = item[3] if len(item) > 3 else 590.0
        age = item[4] if len(item) > 4 else 2
        qm[strike] = {'bid': bid, 'ask': ask, 'spot': spot, 'spot_age': age}
    return qm


# ═══════════════════════════════════════════════════════════════
#  1–6: Fill Model
# ═══════════════════════════════════════════════════════════════

class TestFillModel:

    def test_fill_mid(self):
        """Mid fill = (bid+ask)/2 for both BUY and SELL."""
        r = simulate_fill('BUY', 2.00, 2.20)
        assert r['mid'] == pytest.approx(2.10, abs=0.001)
        r = simulate_fill('SELL', 2.00, 2.20)
        assert r['mid'] == pytest.approx(2.10, abs=0.001)

    def test_fill_touch_buy(self):
        """BUY touch = ask."""
        r = simulate_fill('BUY', 2.00, 2.20)
        assert r['touch'] == pytest.approx(2.20, abs=0.001)

    def test_fill_touch_sell(self):
        """SELL touch = bid."""
        r = simulate_fill('SELL', 2.00, 2.20)
        assert r['touch'] == pytest.approx(2.00, abs=0.001)

    def test_fill_slippage_buy(self):
        """BUY slippage = ask + (spread × pct)."""
        cfg = {'slippage_mode': 'pct_of_spread', 'slippage_pct': 0.10,
               'commission_per_contract': 0.65, 'exchange_fees_per_contract': 0.05}
        r = simulate_fill('BUY', 2.00, 2.20, config=cfg)
        # spread=0.20, slip=0.02, slippage fill = 2.20 + 0.02 = 2.22
        assert r['slippage'] == pytest.approx(2.22, abs=0.001)

    def test_fill_slippage_sell(self):
        """SELL slippage = bid - (spread × pct)."""
        cfg = {'slippage_mode': 'pct_of_spread', 'slippage_pct': 0.10,
               'commission_per_contract': 0.65, 'exchange_fees_per_contract': 0.05}
        r = simulate_fill('SELL', 2.00, 2.20, config=cfg)
        # spread=0.20, slip=0.02, slippage fill = 2.00 - 0.02 = 1.98
        assert r['slippage'] == pytest.approx(1.98, abs=0.001)

    def test_commission_calc(self):
        """Commissions = (per_contract + exchange_fees) × qty."""
        cfg = {'commission_per_contract': 0.65, 'exchange_fees_per_contract': 0.05}
        assert compute_fees(1, cfg) == pytest.approx(0.70, abs=0.01)
        assert compute_fees(5, cfg) == pytest.approx(3.50, abs=0.01)


# ═══════════════════════════════════════════════════════════════
#  7–10: PnL Calculation
# ═══════════════════════════════════════════════════════════════

class TestPnLCalculation:

    def test_pnl_buy_win(self):
        """BUY wins when exit > entry."""
        r = compute_pnl('BUY', entry_fill=2.10, exit_fill=2.50,
                        quantity=1, entry_fees=0.70, exit_fees=0.70)
        assert r['gross_pnl'] == pytest.approx(40.0, abs=0.01)  # (2.50-2.10)*100
        assert r['net_pnl'] == pytest.approx(38.60, abs=0.01)   # 40 - 1.40
        assert r['return_pct'] > 0

    def test_pnl_buy_loss(self):
        """BUY loses when exit < entry."""
        r = compute_pnl('BUY', entry_fill=2.10, exit_fill=1.80,
                        quantity=1, entry_fees=0.70, exit_fees=0.70)
        assert r['gross_pnl'] == pytest.approx(-30.0, abs=0.01)
        assert r['net_pnl'] < 0

    def test_pnl_sell_win(self):
        """SELL wins when exit < entry."""
        r = compute_pnl('SELL', entry_fill=2.00, exit_fill=1.70,
                        quantity=1, entry_fees=0.70, exit_fees=0.70)
        assert r['gross_pnl'] == pytest.approx(30.0, abs=0.01)  # (2.00-1.70)*100
        assert r['net_pnl'] > 0

    def test_pnl_sell_loss(self):
        """SELL loses when exit > entry."""
        r = compute_pnl('SELL', entry_fill=2.00, exit_fill=2.30,
                        quantity=1, entry_fees=0.70, exit_fees=0.70)
        assert r['gross_pnl'] == pytest.approx(-30.0, abs=0.01)
        assert r['net_pnl'] < 0


# ═══════════════════════════════════════════════════════════════
#  11–14: Exit Logic
# ═══════════════════════════════════════════════════════════════

class TestExitLogic:

    def test_tp_exit_on_option_quote(self, journal, config):
        """TP fires on option mid price, not SPY spot."""
        config['exit_mode'] = 'tp_sl'
        config['tp_pct'] = 0.20
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(entry_ts)

        # Enter trade
        trader._enter_trade(sig, 590.0, ts_utc, 'decision_time')
        assert len(trader._open_trades_a) == 1

        # Price moves up 25% → TP should fire
        # Entry mid = 2.10, need 2.10 * 1.20 = 2.52
        exit_ts = entry_ts + timedelta(minutes=2)
        quote_map = _make_quote_map((590.0, 2.50, 2.60))
        trader._check_exits(quote_map, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_a) == 0

    def test_sl_exit_on_option_quote(self, journal, config):
        """SL fires on option mid price."""
        config['exit_mode'] = 'tp_sl'
        config['sl_pct'] = 0.15
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(entry_ts)

        trader._enter_trade(sig, 590.0, ts_utc, 'decision_time')
        assert len(trader._open_trades_a) == 1

        # Price drops 20% → SL should fire
        # Entry mid = 2.10, need < 2.10 * (1-0.15) = 1.785
        exit_ts = entry_ts + timedelta(minutes=2)
        quote_map = _make_quote_map((590.0, 1.60, 1.80))
        trader._check_exits(quote_map, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_a) == 0

    def test_eod_exit(self, journal, config):
        """Force close at EOD time."""
        config['exit_mode'] = 'hybrid'
        config['eod_exit_time'] = '15:55'
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20)
        entry_ts = datetime(2026, 2, 25, 14, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(entry_ts)
        trader._enter_trade(sig, 590.0, ts_utc, 'decision_time')

        # At 15:55 → should force close
        eod_ts = datetime(2026, 2, 25, 15, 55, 0, tzinfo=_ET)
        quote_map = _make_quote_map((590.0, 2.05, 2.15))
        trader._check_exits(quote_map, eod_ts, to_utc_str(eod_ts))
        assert len(trader._open_trades_a) == 0

    def test_time_exit(self, journal, config):
        """Close after exit_time_minutes."""
        config['exit_mode'] = 'time'
        config['exit_time_minutes'] = 5
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(entry_ts)
        trader._enter_trade(sig, 590.0, ts_utc, 'decision_time')

        # At 10:06 → should close
        exit_ts = entry_ts + timedelta(minutes=6)
        quote_map = _make_quote_map((590.0, 2.05, 2.15))
        trader._check_exits(quote_map, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_a) == 0


# ═══════════════════════════════════════════════════════════════
#  15–16: Eligibility & Selection
# ═══════════════════════════════════════════════════════════════

class TestEligibility:

    def test_eligibility_filters(self, journal, config):
        """Each filter rejects with the correct reason."""
        trader = PaperTrader(journal=journal, config=config)

        # HOLD
        assert trader._check_eligibility_a(_make_signal(action='HOLD')) == 'hold_signal'
        # Low confidence
        assert trader._check_eligibility_a(_make_signal(confidence=0.3)) == 'low_confidence'
        # Low edge
        assert trader._check_eligibility_a(_make_signal(edge=0.01)) == 'low_edge'
        # Wide spread
        sig = _make_signal(bid=1.00, ask=2.00)  # 100% spread
        assert trader._check_eligibility_a(sig) == 'spread_too_wide'
        # Too cheap
        assert trader._check_eligibility_a(_make_signal(bid=0.039, ask=0.041)) == 'option_too_cheap'
        # Stale spot
        assert trader._check_eligibility_a(_make_signal(spot_age=30)) == 'stale_spot'
        # Bernoulli
        assert trader._check_eligibility_a(_make_signal(bernoulli_violated=True)) == 'bernoulli_violated'
        # OTM too far
        assert trader._check_eligibility_a(_make_signal(otm_dollars=10.0)) == 'too_far_otm'
        # Good signal → None
        assert trader._check_eligibility_a(_make_signal()) is None

    def test_selection_risk_adjusted(self, journal, config):
        """Risk-adjusted ranking: edge × confidence / spread."""
        config['selection_policy'] = 'risk_adjusted'
        config['spread_floor'] = 0.01
        trader = PaperTrader(journal=journal, config=config)

        sigs = [
            _make_signal(strike=590, edge=0.10, confidence=0.80, bid=2.00, ask=2.20),  # 0.10*0.80/0.20=0.40
            _make_signal(strike=591, edge=0.05, confidence=0.90, bid=2.00, ask=2.10),  # 0.05*0.90/0.10=0.45
            _make_signal(strike=592, edge=0.20, confidence=0.60, bid=2.00, ask=2.50),  # 0.20*0.60/0.50=0.24
        ]
        ranked = trader._rank_signals(sigs)
        # 591 should be first (highest score 0.45)
        assert ranked[0]['strike'] == 591


# ═══════════════════════════════════════════════════════════════
#  17–21: Decision Logic, Cooldown, Positions
# ═══════════════════════════════════════════════════════════════

class TestDecisionLogic:

    def test_decision_fires_once(self, journal, config):
        """Same decision time won't double-fire."""
        config['decision_times'] = ['10:00']
        trader = PaperTrader(journal=journal, config=config)

        ts1 = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        ts2 = datetime(2026, 2, 25, 10, 0, 20, tzinfo=_ET)

        assert trader._is_decision_time(ts1) is True
        assert trader._is_decision_time(ts2) is False  # already fired today

    def test_cooldown_same_direction(self, journal, config):
        """Same strike + same direction is blocked during cooldown."""
        config['cooldown_minutes'] = 5
        config['cooldown_scope'] = 'same_direction'
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY')
        ts = datetime(2026, 2, 25, 10, 5, 0, tzinfo=_ET)

        # Record a close 2 minutes ago
        close_time = datetime(2026, 2, 25, 15, 3, 0, tzinfo=_UTC)
        trader._trade_close_times['decision_time:590.0:BUY'] = close_time

        # 2 minutes later → still in cooldown
        ts_check = datetime(2026, 2, 25, 10, 5, 0, tzinfo=_ET)
        # Map to something within 5 minutes of the close
        trader._trade_close_times['decision_time:590.0:BUY'] = \
            ts_check.astimezone(_UTC) - timedelta(minutes=2)

        assert trader._in_cooldown(sig, 'decision_time', ts_check) is True

    def test_cooldown_allows_flip(self, journal, config):
        """Same strike OPPOSITE direction is allowed when scope=same_direction."""
        config['cooldown_minutes'] = 5
        config['cooldown_scope'] = 'same_direction'
        trader = PaperTrader(journal=journal, config=config)

        # Close a BUY trade 2 min ago
        ts = datetime(2026, 2, 25, 10, 5, 0, tzinfo=_ET)
        trader._trade_close_times['decision_time:590.0:BUY'] = \
            ts.astimezone(_UTC) - timedelta(minutes=2)

        # SELL on same strike should be allowed
        sig_sell = _make_signal(strike=590.0, action='SELL')
        assert trader._in_cooldown(sig_sell, 'decision_time', ts) is False

    def test_max_positions(self, journal, config):
        """Position limit is enforced."""
        config['max_open_positions'] = 2
        config['max_trades_per_decision'] = 5
        trader = PaperTrader(journal=journal, config=config)

        ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        ts_utc = to_utc_str(ts)

        # Enter 2 trades
        for i in range(2):
            sig = _make_signal(strike=590.0 + i, action='BUY')
            trader._enter_trade(sig, 590.0, ts_utc, 'decision_time')

        assert len(trader._open_trades_a) == 2

        # Make decision with more signals — should not enter any more
        signals = [_make_signal(strike=595.0 + i, action='BUY') for i in range(3)]
        quote_map = _make_quote_map((595.0, 2.0, 2.2), (596.0, 2.0, 2.2), (597.0, 2.0, 2.2))
        trader._make_decision_a(signals, quote_map, 590.0, ts, ts_utc)

        # Still only 2 — max reached
        assert len(trader._open_trades_a) == 2

    def test_min_strike_spacing(self, journal, config):
        """Adjacent strikes are deduplicated by min_strike_spacing."""
        config['min_strike_spacing'] = 2.0
        trader = PaperTrader(journal=journal, config=config)

        sigs = [
            _make_signal(strike=590.0, edge=0.10, confidence=0.80),
            _make_signal(strike=591.0, edge=0.15, confidence=0.90),  # too close to 590
            _make_signal(strike=593.0, edge=0.08, confidence=0.70),  # ok
        ]
        diversified = trader._diversify(sigs)
        strikes = [s['strike'] for s in diversified]
        # 591 highest score, then 590 too close, 593 ok
        assert 591.0 in strikes
        assert 593.0 in strikes
        assert 590.0 not in strikes


# ═══════════════════════════════════════════════════════════════
#  22–23: Quote Tape & EOD Stats
# ═══════════════════════════════════════════════════════════════

class TestQuoteTapeAndStats:

    def test_quote_tape_forward_returns(self, journal, config):
        """Forward returns can be computed from recorded quote tape."""
        config['eval_horizons_minutes'] = [5]
        config['max_quote_gap_minutes_for_forward_eval'] = 10
        run_id = journal.start_run(config)

        entry_ts_utc = '2026-02-25T15:00:00+00:00'

        # Open a BUY trade at mid=2.10
        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20)
        fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
        trade_id = journal.open_trade(run_id, 'decision_time', sig, fill,
                                      590.0, entry_ts_utc)

        # Record quotes at entry +5 min
        journal.record_quotes(run_id, '2026-02-25T15:05:00+00:00', [
            {'strike': 590.0, 'bid': 2.30, 'ask': 2.50, 'mid': 2.40,
             'spread': 0.20, 'spot': 591.0, 'spot_age': 2}
        ])

        # Close trade
        exit_fill = {'mid': 2.40, 'touch': 2.30, 'slippage': 2.28}
        pnl = {'gross_pnl': 30.0, 'net_pnl': 28.60, 'return_pct': 0.14}
        journal.close_trade(trade_id, 2.30, 2.50, exit_fill, 591.0,
                            '2026-02-25T15:05:00+00:00', 'fixed_horizon',
                            pnl, pnl, pnl, 5.0)

        reporter = EODReporter(journal, config)
        acc = reporter.compute_forward_accuracy(run_id, track='decision_time')

        assert 5 in acc
        assert acc[5]['total'] == 1
        assert acc[5]['correct'] == 1  # BUY and price went up
        assert acc[5]['accuracy'] == 1.0

    def test_eod_summary_stats(self, journal, config):
        """Aggregate stats match manual calculation."""
        run_id = journal.start_run(config)
        ts = '2026-02-25T15:00:00+00:00'

        # 2 winning BUY trades, 1 losing
        for i, (strike, exit_mid) in enumerate([(590, 2.40), (591, 2.30), (592, 1.80)]):
            sig = _make_signal(strike=float(strike), action='BUY')
            fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
            tid = journal.open_trade(run_id, 'decision_time', sig, fill,
                                     590.0, ts, entry_fees=0.70)
            exit_fill = {'mid': exit_mid, 'touch': exit_mid - 0.05, 'slippage': exit_mid - 0.07}
            entry_amt = 2.10
            gross = (exit_mid - entry_amt) * 100
            net = gross - 1.40
            pnl = {'gross_pnl': gross, 'net_pnl': net, 'return_pct': net / (entry_amt * 100)}
            journal.close_trade(tid, exit_mid - 0.05, exit_mid + 0.05,
                                exit_fill, 590.0, ts, 'time_exit',
                                pnl, pnl, pnl, 5.0, exit_fees=0.70)

        stats = journal.trade_summary(run_id, track='decision_time')
        assert stats['total'] == 3
        assert stats['wins'] == 2
        assert stats['losses'] == 1
        assert stats['win_rate'] == pytest.approx(2 / 3, abs=0.01)


# ═══════════════════════════════════════════════════════════════
#  24–28: Track B
# ═══════════════════════════════════════════════════════════════

class TestTrackB:

    def test_track_b_ignores_thresholds(self, journal, config):
        """Track B enters trades regardless of confidence/edge thresholds."""
        trader = PaperTrader(journal=journal, config=config)

        # Signal with low confidence and low edge
        sig = _make_signal(strike=590.0, action='BUY',
                           confidence=0.10, edge=0.001, spot_age=2)
        ts = datetime(2026, 2, 25, 10, 30, 0, tzinfo=_ET)
        ts_utc = to_utc_str(ts)
        quote_map = _make_quote_map((590.0, 2.00, 2.20))

        trader._enter_all_signals_b([sig], quote_map, 590.0, ts, ts_utc)
        assert len(trader._open_trades_b) == 1

    def test_track_b_dedup(self, journal, config):
        """No repeated (strike, action) while already open."""
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', spot_age=2)
        ts1 = datetime(2026, 2, 25, 10, 30, 0, tzinfo=_ET)
        ts2 = datetime(2026, 2, 25, 10, 31, 0, tzinfo=_ET)

        qm = _make_quote_map((590.0, 2.00, 2.20))
        trader._enter_all_signals_b([sig], qm, 590.0, ts1, to_utc_str(ts1))
        assert len(trader._open_trades_b) == 1

        # Try again — should be deduped
        trader._enter_all_signals_b([sig], qm, 590.0, ts2, to_utc_str(ts2))
        assert len(trader._open_trades_b) == 1  # still 1

    def test_track_b_fixed_horizon_exit(self, journal, config):
        """Track B closes at configured fixed horizon."""
        config['all_signals']['exit_horizon_minutes'] = 5
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', spot_age=2)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        trader._enter_trade(sig, 590.0, to_utc_str(entry_ts), 'all_signals')
        assert len(trader._open_trades_b) == 1

        # 6 minutes later → should close
        exit_ts = entry_ts + timedelta(minutes=6)
        qm = _make_quote_map((590.0, 2.05, 2.15))
        trader._check_exits(qm, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_b) == 0

    def test_eod_report_per_track(self, journal, config):
        """EOD reporter produces stats per track."""
        run_id = journal.start_run(config)
        ts = '2026-02-25T15:00:00+00:00'

        # 1 Track A trade
        sig = _make_signal(strike=590.0, action='BUY')
        fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
        tid = journal.open_trade(run_id, 'decision_time', sig, fill,
                                 590.0, ts, entry_fees=0.70)
        pnl = {'gross_pnl': 30.0, 'net_pnl': 28.60, 'return_pct': 0.136}
        exit_fill = {'mid': 2.40, 'touch': 2.30, 'slippage': 2.28}
        journal.close_trade(tid, 2.30, 2.50, exit_fill, 591.0, ts,
                            'time_exit', pnl, pnl, pnl, 5.0, exit_fees=0.70)

        # 1 Track B trade
        tid2 = journal.open_trade(run_id, 'all_signals', sig, fill,
                                  590.0, ts, entry_fees=0.70)
        journal.close_trade(tid2, 2.30, 2.50, exit_fill, 591.0, ts,
                            'fixed_horizon', pnl, pnl, pnl, 5.0, exit_fees=0.70)

        stats_a = journal.trade_summary(run_id, track='decision_time')
        stats_b = journal.trade_summary(run_id, track='all_signals')

        assert stats_a['total'] == 1
        assert stats_b['total'] == 1

    def test_forward_eval_per_track(self, journal, config):
        """Forward accuracy computed per track."""
        config['eval_horizons_minutes'] = [5]
        config['max_quote_gap_minutes_for_forward_eval'] = 10
        run_id = journal.start_run(config)

        ts_entry = '2026-02-25T15:00:00+00:00'
        ts_exit = '2026-02-25T15:05:00+00:00'

        # Track A: BUY goes up → correct
        sig = _make_signal(strike=590.0, action='BUY')
        fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
        tid_a = journal.open_trade(run_id, 'decision_time', sig, fill,
                                   590.0, ts_entry, entry_fees=0.70)

        # Track B: SELL also goes up → incorrect for SELL
        sig_sell = _make_signal(strike=591.0, action='SELL')
        tid_b = journal.open_trade(run_id, 'all_signals', sig_sell, fill,
                                   590.0, ts_entry, entry_fees=0.70)

        # Quotes at +5 min: both go up
        journal.record_quotes(run_id, ts_exit, [
            {'strike': 590.0, 'bid': 2.30, 'ask': 2.50, 'mid': 2.40,
             'spread': 0.20, 'spot': 591.0, 'spot_age': 2},
            {'strike': 591.0, 'bid': 2.30, 'ask': 2.50, 'mid': 2.40,
             'spread': 0.20, 'spot': 591.0, 'spot_age': 2},
        ])

        # Close both
        pnl = {'gross_pnl': 30.0, 'net_pnl': 28.60, 'return_pct': 0.14}
        exit_fill = {'mid': 2.40, 'touch': 2.30, 'slippage': 2.28}
        journal.close_trade(tid_a, 2.30, 2.50, exit_fill, 591.0, ts_exit,
                            'time_exit', pnl, pnl, pnl, 5.0, exit_fees=0.70)
        journal.close_trade(tid_b, 2.30, 2.50, exit_fill, 591.0, ts_exit,
                            'fixed_horizon', pnl, pnl, pnl, 5.0, exit_fees=0.70)

        reporter = EODReporter(journal, config)
        acc_a = reporter.compute_forward_accuracy(run_id, track='decision_time')
        acc_b = reporter.compute_forward_accuracy(run_id, track='all_signals')

        assert acc_a[5]['accuracy'] == 1.0   # BUY went up → correct
        assert acc_b[5]['accuracy'] == 0.0   # SELL went up → incorrect


# ═══════════════════════════════════════════════════════════════
#  Schema test
# ═══════════════════════════════════════════════════════════════

class TestSchema:

    def test_journal_schema(self, tmp_db):
        """Tables are created correctly on fresh DB."""
        j = PaperJournal(db_path=tmp_db)
        tables = [r[0] for r in j.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        for t in ['backtest_runs', 'paper_trades', 'paper_trade_events',
                   'decision_snapshots', 'option_quotes']:
            assert t in tables, f"Missing table: {t}"

        ver = j.conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver == 2
        j.close()


# ═══════════════════════════════════════════════════════════════
#  Track C Decision Gate (P0 fix)
# ═══════════════════════════════════════════════════════════════

class TestTrackCDecisionGate:

    def test_track_c_fires_at_same_decision_time_as_a(self, journal, config):
        """Track C should enter trades at the same decision time as Track A."""
        import copy
        cfg = copy.deepcopy(config)
        cfg['decision_times'] = ['10:00']
        cfg['max_trades_per_decision'] = 1  # Track A takes only 1
        cfg['buy_only'] = {
            'enabled': True,
            'enter_on': 'decision_time',
            'action_filter': 'BUY',
            'option_types': ['call', 'put'],
            'max_open_positions': 3,
            'max_trades_per_decision': 1,
            'use_eu_scoring': False,
            'use_regime_thresholds': False,
            'selection_policy': 'risk_adjusted',
            'dedup_policy': 'one_open_per_strike_type',
            'cooldown_minutes': 0,
            'cross_track_dedup': True,
            'filters': cfg.get('filters', {}),
            'exit_mode': 'hybrid',
            'tp_pct': 0.25,
            'sl_pct': 0.20,
            'exit_time_minutes': 30,
            'eod_exit_time': '15:55',
        }
        trader = PaperTrader(journal=journal, config=cfg)

        # Three BUY signals at DIFFERENT strikes
        # Track A takes top-1 (590), Track C should pick from 591 or 592
        sig1 = _make_signal(strike=590.0, action='BUY', edge=0.10, confidence=0.90)
        sig2 = _make_signal(strike=591.0, action='BUY', edge=0.08, confidence=0.85)
        sig3 = _make_signal(strike=592.0, action='BUY', edge=0.06, confidence=0.80)

        ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        qm = _make_quote_map((590.0, 2.00, 2.20), (591.0, 1.90, 2.10),
                             (592.0, 1.80, 2.00))

        trader.on_scan([sig1, sig2, sig3], qm, 590.0, ts)

        # Both tracks should have entered
        assert len(trader._open_trades_a) > 0, "Track A should have entered trades"
        assert len(trader._open_trades_c) > 0, "Track C should have entered trades (gate fix)"


# ═══════════════════════════════════════════════════════════════
#  Track B Absolute Stop (P2 improvement)
# ═══════════════════════════════════════════════════════════════

class TestTrackBAbsoluteStop:

    def test_absolute_stop_triggers(self, journal, config):
        """Track B closes when unrealized loss exceeds max_loss_dollars."""
        config['all_signals']['max_loss_dollars'] = 60.0
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20, spot_age=2)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        trader._enter_trade(sig, 590.0, to_utc_str(entry_ts), 'all_signals')
        assert len(trader._open_trades_b) == 1

        # Entry mid = 2.10; loss of $60 per contract → mid needs to drop
        # to 2.10 - 0.60 = 1.50  (0.60 × 100 = $60)
        exit_ts = entry_ts + timedelta(minutes=2)
        qm = _make_quote_map((590.0, 1.40, 1.60))  # mid = 1.50 → loss = $60
        trader._check_exits(qm, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_b) == 0, "Absolute stop should have closed the trade"

    def test_absolute_stop_does_not_fire_below_threshold(self, journal, config):
        """Trade stays open when loss is below max_loss_dollars."""
        config['all_signals']['max_loss_dollars'] = 60.0
        config['all_signals']['exit_horizon_minutes'] = 30  # long horizon
        trader = PaperTrader(journal=journal, config=config)

        sig = _make_signal(strike=590.0, action='BUY', bid=2.00, ask=2.20, spot_age=2)
        entry_ts = datetime(2026, 2, 25, 10, 0, 0, tzinfo=_ET)
        trader._enter_trade(sig, 590.0, to_utc_str(entry_ts), 'all_signals')

        # Entry mid = 2.10; loss of $40 → below threshold
        exit_ts = entry_ts + timedelta(minutes=2)
        qm = _make_quote_map((590.0, 1.60, 1.80))  # mid = 1.70 → loss = $40
        trader._check_exits(qm, exit_ts, to_utc_str(exit_ts))
        assert len(trader._open_trades_b) == 1, "Trade should stay open (loss below cap)"


# ═══════════════════════════════════════════════════════════════
#  EOD Reporter Decision Time Display (ET fix)
# ═══════════════════════════════════════════════════════════════

class TestDecisionTimeDisplay:

    def test_decision_time_shows_et_not_utc(self, journal, config, capsys):
        """Decision time breakdown should display ET, not UTC."""
        run_id = journal.start_run(config)
        ts_utc = '2026-02-25T15:00:00+00:00'  # 15:00 UTC = 10:00 ET

        sig = _make_signal(strike=590.0, action='BUY')
        fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
        tid = journal.open_trade(run_id, 'decision_time', sig, fill,
                                 590.0, ts_utc, entry_fees=0.70,
                                 decision_timestamp_utc=ts_utc)
        pnl = {'gross_pnl': 30.0, 'net_pnl': 28.60, 'return_pct': 0.14}
        exit_fill = {'mid': 2.40, 'touch': 2.30, 'slippage': 2.28}
        journal.close_trade(tid, 2.30, 2.50, exit_fill, 591.0, ts_utc,
                            'time_exit', pnl, pnl, pnl, 5.0, exit_fees=0.70)

        reporter = EODReporter(journal, config)
        trades = journal.get_all_trades(run_id, track='decision_time')
        reporter._print_by_decision_time(trades)

        output = capsys.readouterr().out
        assert '10:00' in output, f"Should show 10:00 ET, got: {output}"
        assert '15:00' not in output, f"Should NOT show 15:00 UTC, got: {output}"

