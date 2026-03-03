"""
Backtest Metrics — advanced analytics for paper trading runs.

Computes:
  1. Median Per-Day Metrics (avg PnL/trade + total PnL)
  2. Drawdown (trade-by-trade primary, daily secondary)
  3. Tail-Loss Frequency (2× and 3× avg loss thresholds)
  4. PF / Accuracy Stability by Regime (calm/normal/volatile/extreme)

All daily grouping uses ET session date, not UTC.
All computations are read-only — no DB schema changes.
"""

from datetime import datetime
from statistics import median
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')

# ── Regime buckets — import from canonical source ────────────
from calibration.regime_detector import REGIME_BUCKETS

MIN_SAMPLE_WARNING = 10   # flag buckets with fewer trades


class BacktestMetrics:
    """Read-only analytics computed from PaperJournal data."""

    def __init__(self, journal, run_id: str, track: str = None):
        self.journal = journal
        self.run_id = run_id
        self.track = track

        # Load closed trades once
        self._trades = journal.get_closed_trades(
            run_id, track=track
        )
        # Sort by entry timestamp (chronological)
        self._trades.sort(
            key=lambda t: t.get('entry_timestamp_utc') or ''
        )

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _et_date(utc_timestamp_str: str) -> str:
        """Convert a UTC ISO timestamp to ET session date string (YYYY-MM-DD)."""
        try:
            dt = datetime.fromisoformat(utc_timestamp_str)
            return dt.astimezone(_ET).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return 'unknown'

    def _pnl(self, trade: dict) -> float:
        """Primary PnL value (touch tier)."""
        return trade.get('net_pnl_touch') or 0.0

    def _group_by_et_date(self) -> Dict[str, List[dict]]:
        """Group trades by ET session date."""
        by_date: Dict[str, List[dict]] = {}
        for t in self._trades:
            d = self._et_date(t.get('entry_timestamp_utc', ''))
            by_date.setdefault(d, []).append(t)
        return by_date

    # ─────────────────────────────────────────────────────────
    #  1. Median Per-Day Metrics
    # ─────────────────────────────────────────────────────────

    def per_day_metrics(self) -> Dict[str, Any]:
        """
        Returns:
            median_daily_avg_pnl:  median of (daily sum / daily n_trades)
            median_daily_total_pnl: median of daily total PnL
            daily_series: [{date, n_trades, total_pnl, avg_pnl}, ...]
        """
        by_date = self._group_by_et_date()
        if not by_date:
            return {
                'median_daily_avg_pnl': 0.0,
                'median_daily_total_pnl': 0.0,
                'daily_series': [],
            }

        daily_series = []
        daily_avgs = []
        daily_totals = []

        for d in sorted(by_date):
            trades = by_date[d]
            total = sum(self._pnl(t) for t in trades)
            avg = total / len(trades) if trades else 0.0
            daily_series.append({
                'date': d,
                'n_trades': len(trades),
                'total_pnl': round(total, 2),
                'avg_pnl': round(avg, 2),
            })
            daily_avgs.append(avg)
            daily_totals.append(total)

        return {
            'median_daily_avg_pnl': round(median(daily_avgs), 2),
            'median_daily_total_pnl': round(median(daily_totals), 2),
            'daily_series': daily_series,
        }

    # ─────────────────────────────────────────────────────────
    #  2. Drawdown
    # ─────────────────────────────────────────────────────────

    def drawdown(self) -> Dict[str, Any]:
        """
        Primary: trade-by-trade equity curve drawdown.
        Secondary: daily equity curve drawdown.

        Returns dict with:
            trade_by_trade: {max_dd_dollars, max_dd_pct, dd_series}
            daily:          {max_dd_dollars, max_dd_pct, dd_series}
        """
        return {
            'trade_by_trade': self._drawdown_from_pnls(
                [self._pnl(t) for t in self._trades],
                labels=[t.get('entry_timestamp_utc', '')[:19]
                        for t in self._trades],
            ),
            'daily': self._drawdown_from_daily(),
        }

    def _drawdown_from_pnls(self, pnls: List[float],
                            labels: List[str] = None) -> Dict[str, Any]:
        """Compute drawdown from an ordered list of PnL values."""
        if not pnls:
            return {
                'max_dd_dollars': 0.0,
                'max_dd_pct': 0.0,
                'dd_series': [],
            }

        equity = 0.0
        peak = 0.0
        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        dd_series = []

        for i, pnl in enumerate(pnls):
            equity += pnl
            if equity > peak:
                peak = equity

            dd_dollars = peak - equity
            dd_pct = (dd_dollars / peak * 100) if peak > 0 else 0.0

            if dd_dollars > max_dd_dollars:
                max_dd_dollars = dd_dollars
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            dd_series.append({
                'label': labels[i] if labels and i < len(labels) else str(i),
                'equity': round(equity, 2),
                'peak': round(peak, 2),
                'drawdown_dollars': round(dd_dollars, 2),
                'drawdown_pct': round(dd_pct, 2),
            })

        return {
            'max_dd_dollars': round(max_dd_dollars, 2),
            'max_dd_pct': round(max_dd_pct, 2),
            'dd_series': dd_series,
        }

    def _drawdown_from_daily(self) -> Dict[str, Any]:
        """Secondary: daily equity drawdown."""
        by_date = self._group_by_et_date()
        if not by_date:
            return {'max_dd_dollars': 0.0, 'max_dd_pct': 0.0, 'dd_series': []}

        dates = sorted(by_date)
        daily_pnls = [sum(self._pnl(t) for t in by_date[d]) for d in dates]
        return self._drawdown_from_pnls(daily_pnls, labels=dates)

    # ─────────────────────────────────────────────────────────
    #  3. Tail-Loss Frequency
    # ─────────────────────────────────────────────────────────

    def tail_loss(self, bad_mult: float = 2.0,
                  extreme_mult: float = 3.0) -> Dict[str, Any]:
        """
        Identify tail losses exceeding multiples of avg absolute loss.

        Args:
            bad_mult:     multiplier for "bad" tail (default 2×)
            extreme_mult: multiplier for "extreme" tail (default 3×)

        Returns dict with counts, percentages, magnitudes, and
        tail contribution to both total losses and net PnL.
        """
        losses = [self._pnl(t) for t in self._trades if self._pnl(t) < 0]
        total_trades = len(self._trades)
        total_net_pnl = sum(self._pnl(t) for t in self._trades)
        total_loss_sum = sum(losses)  # negative number

        if not losses:
            return {
                'avg_loss_abs': 0.0,
                'bad_threshold': 0.0,
                'extreme_threshold': 0.0,
                'bad_count': 0,
                'bad_pct': 0.0,
                'extreme_count': 0,
                'extreme_pct': 0.0,
                'avg_bad_tail_magnitude': 0.0,
                'avg_extreme_tail_magnitude': 0.0,
                'tail_pct_of_total_losses': 0.0,
                'tail_pct_of_net_pnl': 0.0,
                'total_trades': total_trades,
                'total_losses': 0,
            }

        avg_loss_abs = sum(abs(l) for l in losses) / len(losses)
        bad_thresh = bad_mult * avg_loss_abs
        extreme_thresh = extreme_mult * avg_loss_abs

        bad_tails = [l for l in losses if abs(l) > bad_thresh]
        extreme_tails = [l for l in losses if abs(l) > extreme_thresh]

        bad_tail_sum = sum(bad_tails)  # negative
        avg_bad_mag = (sum(abs(l) for l in bad_tails) / len(bad_tails)
                       if bad_tails else 0.0)
        avg_extreme_mag = (sum(abs(l) for l in extreme_tails) / len(extreme_tails)
                           if extreme_tails else 0.0)

        # Contribution metrics
        tail_pct_of_losses = (
            (bad_tail_sum / total_loss_sum * 100)
            if total_loss_sum < 0 else 0.0
        )
        tail_pct_of_net = (
            (bad_tail_sum / total_net_pnl * 100)
            if total_net_pnl != 0 else 0.0
        )

        return {
            'avg_loss_abs': round(avg_loss_abs, 2),
            'bad_threshold': round(bad_thresh, 2),
            'extreme_threshold': round(extreme_thresh, 2),
            'bad_count': len(bad_tails),
            'bad_pct': round(len(bad_tails) / total_trades * 100, 1)
                       if total_trades else 0.0,
            'extreme_count': len(extreme_tails),
            'extreme_pct': round(len(extreme_tails) / total_trades * 100, 1)
                           if total_trades else 0.0,
            'avg_bad_tail_magnitude': round(avg_bad_mag, 2),
            'avg_extreme_tail_magnitude': round(avg_extreme_mag, 2),
            'tail_pct_of_total_losses': round(tail_pct_of_losses, 1),
            'tail_pct_of_net_pnl': round(tail_pct_of_net, 1),
            'total_trades': total_trades,
            'total_losses': len(losses),
        }

    # ─────────────────────────────────────────────────────────
    #  4. PF / Accuracy by Regime
    # ─────────────────────────────────────────────────────────

    def regime_stability(self) -> Dict[str, Any]:
        """
        Compute win_rate, profit_factor, expectancy per regime bucket.

        Regime is determined by intraday_move_pct from the decision_snapshot
        closest to (and ≤) each trade's entry time, for the same track.

        Returns dict keyed by regime name with stats + warning flags.
        """
        # Load snapshots for this run/track
        snap_q = """
            SELECT timestamp_utc, intraday_move_pct, track
            FROM decision_snapshots
            WHERE run_id = ?
        """
        params = [self.run_id]
        if self.track:
            snap_q += " AND track = ?"
            params.append(self.track)
        snap_q += " ORDER BY timestamp_utc"

        snaps = [dict(r) for r in
                 self.journal.conn.execute(snap_q, params).fetchall()]

        # Build lookup: for each trade, find nearest prior snapshot
        def _find_regime(trade: dict) -> str:
            entry_ts = trade.get('entry_timestamp_utc', '')
            trade_track = trade.get('track', '')
            best = None
            for s in snaps:
                s_ts = s.get('timestamp_utc', '')
                s_track = s.get('track', '')
                if s_ts <= entry_ts and (not self.track or s_track == trade_track):
                    best = s
            if best is None or best.get('intraday_move_pct') is None:
                return 'unknown'
            move = abs(best['intraday_move_pct'])
            for name, lo, hi in REGIME_BUCKETS:
                if lo <= move < hi:
                    return name
            return 'unknown'

        # Bucket trades by regime
        buckets: Dict[str, List[dict]] = {name: [] for name, _, _ in REGIME_BUCKETS}
        buckets['unknown'] = []

        for t in self._trades:
            regime = _find_regime(t)
            buckets[regime].append(t)

        # Compute stats per bucket
        results = {}
        for regime, trades in buckets.items():
            n = len(trades)
            if n == 0:
                continue

            wins = [t for t in trades if self._pnl(t) > 0]
            losses_list = [t for t in trades if self._pnl(t) <= 0]

            total_pnl = sum(self._pnl(t) for t in trades)
            gross_wins = sum(self._pnl(t) for t in wins)
            gross_losses = abs(sum(self._pnl(t) for t in losses_list))

            if gross_losses > 0:
                pf = gross_wins / gross_losses
            elif gross_wins > 0:
                pf = float('inf')
            else:
                pf = 0.0

            results[regime] = {
                'n_trades': n,
                'wins': len(wins),
                'losses': len(losses_list),
                'win_rate': round(len(wins) / n * 100, 1),
                'profit_factor': round(pf, 2) if pf != float('inf') else float('inf'),
                'expectancy': round(total_pnl / n, 2),
                'total_pnl': round(total_pnl, 2),
                'small_sample': n < MIN_SAMPLE_WARNING,
            }

        return results

    # ─────────────────────────────────────────────────────────
    #  All-in-one
    # ─────────────────────────────────────────────────────────

    def compute_all(self) -> Dict[str, Any]:
        """Compute all four metrics and return as a single dict."""
        return {
            'per_day': self.per_day_metrics(),
            'drawdown': self.drawdown(),
            'tail_loss': self.tail_loss(),
            'regime': self.regime_stability(),
        }
