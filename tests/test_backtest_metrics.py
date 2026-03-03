"""
Backtest Metrics — test suite.

10 tests covering per-day metrics, drawdown (trade-by-trade and daily),
tail-loss frequency, regime stability, and edge cases.
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trading.paper_journal import PaperJournal, to_utc_str
from trading.backtest_metrics import BacktestMetrics

_ET = ZoneInfo('America/New_York')
_UTC = ZoneInfo('UTC')


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

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


def _make_signal(strike=590.0, action='BUY', **kwargs):
    """Minimal signal dict for test trades."""
    sig = {
        'strike': strike,
        'action': action,
        'confidence': 0.80,
        'edge': 0.05,
        'required_edge': 0.02,
        'market_bid': 2.00,
        'market_ask': 2.20,
        'market_mid': 2.10,
        'model_price': 2.20,
        'std_error': 0.01,
        'market_iv': 0.25,
        'spot_age_seconds': 3,
        'otm_dollars': 1.0,
        'bernoulli_violated': False,
        'ticker': 'SPY',
        'option_type': 'call',
    }
    sig.update(kwargs)
    return sig


def _insert_closed_trade(journal, run_id, track, entry_ts_utc, pnl_touch,
                          strike=590.0, action='BUY'):
    """Insert a closed trade with known PnL for testing metrics."""
    sig = _make_signal(strike=strike, action=action)
    fill = {'mid': 2.10, 'touch': 2.20, 'slippage': 2.22, 'spread': 0.20}
    tid = journal.open_trade(run_id, track, sig, fill, 590.0, entry_ts_utc,
                              entry_fees=0.70)

    # Compute fill prices from desired PnL
    # pnl_touch = (exit - entry) * 100 * qty - fees for BUY
    gross = pnl_touch + 1.40  # add back entry + exit fees
    exit_mid = 2.10 + gross / 100  # reverse engineer

    exit_fill = {'mid': exit_mid, 'touch': exit_mid, 'slippage': exit_mid}
    pnl_dict = {'gross_pnl': gross, 'net_pnl': pnl_touch,
                'return_pct': pnl_touch / 210}

    journal.close_trade(tid, exit_mid - 0.05, exit_mid + 0.05,
                        exit_fill, 590.0, entry_ts_utc,
                        'time_exit', pnl_dict, pnl_dict, pnl_dict,
                        5.0, exit_fees=0.70)
    return tid


# ═══════════════════════════════════════════════════════════════
#  1–3: Per-Day Metrics
# ═══════════════════════════════════════════════════════════════

class TestPerDayMetrics:

    def test_median_daily_avg_pnl(self, journal):
        """Median Per-Day Avg PnL computed correctly across days."""
        run_id = journal.start_run({})

        # Day 1 (ET): 2 trades, PnL +10, +20 → avg = +15
        # Day 2 (ET): 1 trade, PnL +5 → avg = +5
        # Day 3 (ET): 2 trades, PnL -10, +30 → avg = +10
        # Median of [15, 5, 10] = 10

        # All timestamps in ET afternoon (so UTC date = same ET date)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-23T19:00:00+00:00', 10.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-23T19:30:00+00:00', 20.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-24T19:00:00+00:00', 5.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T19:00:00+00:00', -10.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T19:30:00+00:00', 30.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        result = m.per_day_metrics()

        assert result['median_daily_avg_pnl'] == 10.0
        assert len(result['daily_series']) == 3

    def test_median_daily_total_pnl(self, journal):
        """Median Daily Total PnL captures different operational reality."""
        run_id = journal.start_run({})

        # Day 1: 1 trade, PnL +40 → total = +40
        # Day 2: 4 trades, PnL +5 each → total = +20
        # Median of [40, 20] = 30
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-23T19:00:00+00:00', 40.0)

        for i in range(4):
            ts = f'2026-02-24T{15+i}:00:00+00:00'
            _insert_closed_trade(journal, run_id, 'decision_time', ts, 5.0,
                                  strike=590.0 + i)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        result = m.per_day_metrics()

        assert result['median_daily_total_pnl'] == 30.0

    def test_empty_trades(self, journal):
        """No trades returns zero metrics."""
        run_id = journal.start_run({})
        m = BacktestMetrics(journal, run_id)
        result = m.per_day_metrics()
        assert result['median_daily_avg_pnl'] == 0.0
        assert result['daily_series'] == []


# ═══════════════════════════════════════════════════════════════
#  4–5: Drawdown
# ═══════════════════════════════════════════════════════════════

class TestDrawdown:

    def test_trade_by_trade_drawdown(self, journal):
        """Trade-by-trade drawdown captures intraday swings."""
        run_id = journal.start_run({})

        # Trades in sequence: +20, -50, +10
        # Equity:  20 → -30 → -20
        # Peak:    20 →  20 →  20
        # DD:       0 →  50 →  40
        # Max DD = $50

        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:00:00+00:00', 20.0,
                              strike=590.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:10:00+00:00', -50.0,
                              strike=591.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:20:00+00:00', 10.0,
                              strike=592.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        dd = m.drawdown()

        assert dd['trade_by_trade']['max_dd_dollars'] == 50.0
        assert len(dd['trade_by_trade']['dd_series']) == 3

    def test_all_winners_no_drawdown(self, journal):
        """All winning trades have zero drawdown."""
        run_id = journal.start_run({})

        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:00:00+00:00', 10.0,
                              strike=590.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:10:00+00:00', 15.0,
                              strike=591.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        dd = m.drawdown()

        assert dd['trade_by_trade']['max_dd_dollars'] == 0.0


# ═══════════════════════════════════════════════════════════════
#  6–7: Tail-Loss Frequency
# ═══════════════════════════════════════════════════════════════

class TestTailLoss:

    def test_tail_identification(self, journal):
        """Bad and extreme tails identified using abs-based thresholds."""
        run_id = journal.start_run({})

        # Losses: -10, -10, -10, -10, -50
        # avg_loss_abs = (10+10+10+10+50)/5 = 18
        # bad threshold = 2 × 18 = 36  → -50 qualifies
        # extreme threshold = 3 × 18 = 54 → nothing qualifies

        for i in range(4):
            _insert_closed_trade(journal, run_id, 'decision_time',
                                  f'2026-02-25T{15+i}:00:00+00:00', -10.0,
                                  strike=590.0 + i)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T19:30:00+00:00', -50.0,
                              strike=595.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        tl = m.tail_loss()

        assert tl['avg_loss_abs'] == 18.0
        assert tl['bad_count'] == 1
        assert tl['extreme_count'] == 0
        assert tl['total_losses'] == 5

    def test_no_losses_no_tails(self, journal):
        """All winners → no tail losses."""
        run_id = journal.start_run({})

        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:00:00+00:00', 20.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:30:00+00:00', 30.0,
                              strike=591.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        tl = m.tail_loss()

        assert tl['avg_loss_abs'] == 0.0
        assert tl['bad_count'] == 0
        assert tl['total_losses'] == 0

    def test_tail_contribution_to_net_pnl(self, journal):
        """Tail % of net PnL reflects how much tails dominate."""
        run_id = journal.start_run({})

        # 3 wins of +10
        for i in range(3):
            _insert_closed_trade(journal, run_id, 'decision_time',
                                  f'2026-02-25T{15+i}:00:00+00:00', 10.0,
                                  strike=590.0 + i)
        # 2 losses: -5, -25
        # avg_loss_abs = (5+25)/2 = 15
        # bad threshold = 2×15 = 30 → nothing qualifies (25 < 30)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T19:00:00+00:00', -5.0,
                              strike=594.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T19:30:00+00:00', -25.0,
                              strike=595.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        tl = m.tail_loss()

        # Net PnL = 30 - 30 = 0 → tail % of net PnL = 0
        assert tl['bad_count'] == 0  # 25 < 30 so no bad tails


# ═══════════════════════════════════════════════════════════════
#  8–10: Regime Stability
# ═══════════════════════════════════════════════════════════════

class TestRegimeStability:

    def test_regime_bucketing(self, journal):
        """Trades correctly bucketed by intraday_move_pct at entry."""
        run_id = journal.start_run({})

        # Record snapshots with different intraday moves
        journal.record_snapshot(run_id, 'decision_time',
                                '2026-02-25T15:00:00+00:00',
                                590.0, 5, 3, 2, 2,
                                intraday_move_pct=0.3)  # calm
        journal.record_snapshot(run_id, 'decision_time',
                                '2026-02-25T16:00:00+00:00',
                                590.0, 5, 3, 2, 2,
                                intraday_move_pct=1.5)  # volatile

        # Trade entered near calm snapshot → should bucket as calm
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:01:00+00:00', 10.0,
                              strike=590.0)
        # Trade entered near volatile snapshot → should bucket as volatile
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T16:01:00+00:00', -5.0,
                              strike=591.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        rs = m.regime_stability()

        assert 'calm' in rs
        assert 'volatile' in rs
        assert rs['calm']['n_trades'] == 1
        assert rs['volatile']['n_trades'] == 1

    def test_small_sample_warning(self, journal):
        """Buckets with <10 trades are flagged."""
        run_id = journal.start_run({})

        journal.record_snapshot(run_id, 'decision_time',
                                '2026-02-25T15:00:00+00:00',
                                590.0, 5, 3, 2, 2,
                                intraday_move_pct=0.3)

        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:01:00+00:00', 10.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        rs = m.regime_stability()

        assert rs['calm']['small_sample'] is True

    def test_pf_all_winners(self, journal):
        """PF = inf when no losses in a regime bucket."""
        run_id = journal.start_run({})

        journal.record_snapshot(run_id, 'decision_time',
                                '2026-02-25T15:00:00+00:00',
                                590.0, 5, 3, 2, 2,
                                intraday_move_pct=0.3)

        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:01:00+00:00', 10.0,
                              strike=590.0)
        _insert_closed_trade(journal, run_id, 'decision_time',
                              '2026-02-25T15:02:00+00:00', 20.0,
                              strike=591.0)

        m = BacktestMetrics(journal, run_id, track='decision_time')
        rs = m.regime_stability()

        assert rs['calm']['profit_factor'] == float('inf')
        assert rs['calm']['win_rate'] == 100.0
