import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from trading.paper_journal import PaperJournal
from trading.paper_config import PAPER_TRADING


class EODReporter:
    """Generate end-of-day reports from PaperJournal data."""

    def __init__(self, journal: PaperJournal, config: dict = None):
        self.journal = journal
        self.cfg = config or PAPER_TRADING

    # ================================================================== #
    #  Console summary                                                     #
    # ================================================================== #

    def print_summary(self, run_id: str, date_str: str = None):
        """Print per-track EOD summary to console."""
        tracks = self.cfg.get('tracks', ['decision_time', 'all_signals'])

        print(f"\n{'='*65}")
        print(f"📊 PAPER TRADING — END OF DAY SUMMARY")
        if date_str:
            print(f"   Date: {date_str}")
        print(f"{'='*65}")

        for track in tracks:
            self._print_track_summary(run_id, track, date_str)

        # Survivorship funnel
        self._print_survivorship(run_id, date_str)
        print(f"{'='*65}\n")

    def _print_track_summary(self, run_id: str, track: str,
                             date_str: str = None):
        """Print summary for one track."""
        trades = self.journal.get_closed_trades(run_id, track=track,
                                                date_str=date_str)
        open_trades = self.journal.get_open_trades(run_id, track=track)

        print(f"\n{'─'*65}")
        label = 'TRACK A: decision_time' if track == 'decision_time' else 'TRACK B: all_signals'
        print(f"  {label}")
        print(f"{'─'*65}")

        if not trades and not open_trades:
            print(f"  No trades.")
            return

        total = len(trades)
        buys = sum(1 for t in trades if t['action'] == 'BUY')
        sells = sum(1 for t in trades if t['action'] == 'SELL')
        wins = sum(1 for t in trades if (t.get('net_pnl_touch') or 0) > 0)
        losses = total - wins

        # PnL (touch tier = primary)
        total_pnl_mid = sum(t.get('net_pnl_mid') or 0 for t in trades)
        total_pnl_touch = sum(t.get('net_pnl_touch') or 0 for t in trades)
        total_pnl_slip = sum(t.get('net_pnl_slippage') or 0 for t in trades)

        gross_wins = sum(t.get('net_pnl_touch') or 0 for t in trades
                         if (t.get('net_pnl_touch') or 0) > 0)
        gross_losses = abs(sum(t.get('net_pnl_touch') or 0 for t in trades
                               if (t.get('net_pnl_touch') or 0) <= 0))

        win_rate = wins / total * 100 if total else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        expectancy = total_pnl_touch / total if total else 0

        avg_hold = (sum(t.get('hold_minutes') or 0 for t in trades) / total
                    if total else 0)
        avg_edge = (sum(t.get('edge_entry') or 0 for t in trades) / total
                    if total else 0)
        avg_conf = (sum(t.get('confidence_entry') or 0 for t in trades) / total
                    if total else 0)

        print(f"  Trades      : {total} (BUY: {buys}, SELL: {sells})")
        print(f"  Win/Loss    : {wins}W / {losses}L ({win_rate:.1f}%)")
        print(f"  Still Open  : {len(open_trades)}")
        print(f"  Avg Hold    : {avg_hold:.1f} min")
        print(f"  Avg Edge    : {avg_edge*100:.2f}%")
        print(f"  Avg Conf    : {avg_conf*100:.1f}%")
        print()
        print(f"  PnL (mid)      : ${total_pnl_mid:+.2f}")
        print(f"  PnL (touch)    : ${total_pnl_touch:+.2f}")
        print(f"  PnL (slippage) : ${total_pnl_slip:+.2f}")
        print(f"  Profit Factor  : {profit_factor:.2f}")
        print(f"  Expectancy     : ${expectancy:+.2f}/trade")

        # ── Top/Bottom 5 ──
        if trades:
            sorted_trades = sorted(trades,
                                   key=lambda t: t.get('net_pnl_touch') or 0,
                                   reverse=True)
            print(f"\n  Top 5:")
            for t in sorted_trades[:5]:
                print(f"    {t['strike']}C {t['action']} "
                      f"PnL=${t.get('net_pnl_touch', 0):+.2f} "
                      f"({t.get('exit_reason', '?')})")
            if len(sorted_trades) > 5:
                print(f"  Bottom 5:")
                for t in sorted_trades[-5:]:
                    print(f"    {t['strike']}C {t['action']} "
                          f"PnL=${t.get('net_pnl_touch', 0):+.2f} "
                          f"({t.get('exit_reason', '?')})")

        # ── Performance by decision time ──
        if track == 'decision_time' and trades:
            self._print_by_decision_time(trades)

        # ── Performance by moneyness ──
        if trades:
            self._print_by_moneyness(trades)

    def _print_by_decision_time(self, trades: List[dict]):
        """PnL breakdown by decision time."""
        buckets: Dict[str, list] = {}
        for t in trades:
            ts = t.get('decision_timestamp_utc', '')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    hhmm = dt.strftime('%H:%M')
                except (ValueError, TypeError):
                    hhmm = '??:??'
            else:
                hhmm = '??:??'
            buckets.setdefault(hhmm, []).append(t)

        if buckets:
            print(f"\n  By Decision Time:")
            for hhmm in sorted(buckets):
                bt = buckets[hhmm]
                pnl = sum(t.get('net_pnl_touch') or 0 for t in bt)
                n = len(bt)
                w = sum(1 for t in bt if (t.get('net_pnl_touch') or 0) > 0)
                print(f"    {hhmm}: {n} trades, {w}W, PnL=${pnl:+.2f}")

    def _print_by_moneyness(self, trades: List[dict]):
        """PnL breakdown by ATM/OTM distance."""
        buckets = {'ATM (≤$1)': [], 'Near OTM ($1-3)': [], 'Far OTM (>$3)': []}
        for t in trades:
            otm = abs(t.get('otm_dollars_entry') or 0)
            if otm <= 1:
                buckets['ATM (≤$1)'].append(t)
            elif otm <= 3:
                buckets['Near OTM ($1-3)'].append(t)
            else:
                buckets['Far OTM (>$3)'].append(t)

        print(f"\n  By Moneyness:")
        for label, bt in buckets.items():
            if bt:
                pnl = sum(t.get('net_pnl_touch') or 0 for t in bt)
                n = len(bt)
                w = sum(1 for t in bt if (t.get('net_pnl_touch') or 0) > 0)
                print(f"    {label}: {n} trades, {w}W, PnL=${pnl:+.2f}")

    def _print_survivorship(self, run_id: str, date_str: str = None):
        """Print survivorship funnel from decision snapshots."""
        q = "SELECT * FROM decision_snapshots WHERE run_id = ?"
        params = [run_id]
        if date_str:
            q += " AND timestamp_utc LIKE ?"
            params.append(f"{date_str}%")

        rows = self.journal.conn.execute(q, params).fetchall()
        if not rows:
            return

        print(f"\n{'─'*65}")
        print(f"  📋 SURVIVORSHIP FUNNEL")
        print(f"{'─'*65}")

        for track in ['decision_time', 'all_signals']:
            track_rows = [dict(r) for r in rows if r['track'] == track]
            if not track_rows:
                continue

            total_sig = sum(r.get('n_signals') or 0 for r in track_rows)
            total_elig = sum(r.get('n_eligible') or 0 for r in track_rows)
            total_sel = sum(r.get('n_selected') or 0 for r in track_rows)
            total_ent = sum(r.get('n_entered') or 0 for r in track_rows)

            label = 'Track A' if track == 'decision_time' else 'Track B'
            print(f"\n  {label}: {len(track_rows)} decision points")
            print(f"    Signals seen  : {total_sig}")
            print(f"    Eligible      : {total_elig}")
            print(f"    Selected      : {total_sel}")
            print(f"    Entered       : {total_ent}")

            # Aggregate rejection reasons
            import json as _json
            all_rejections: Dict[str, int] = {}
            for r in track_rows:
                rej = r.get('rejection_breakdown_json')
                if rej:
                    try:
                        rej_dict = _json.loads(rej)
                        for reason, count in rej_dict.items():
                            all_rejections[reason] = all_rejections.get(reason, 0) + count
                    except (ValueError, TypeError):
                        pass

            if all_rejections:
                print(f"    Rejections:")
                for reason, count in sorted(all_rejections.items(),
                                            key=lambda x: -x[1]):
                    print(f"      {reason}: {count}")

    # ================================================================== #
    #  Forward evaluation                                                  #
    # ================================================================== #

    def compute_forward_accuracy(self, run_id: str, track: str = None,
                                 date_str: str = None) -> Dict:
        """Compute forward return accuracy at configured horizons.

        Uses the quote tape to evaluate whether each trade's direction
        was correct at 1/3/5/10/15/30/60 minute horizons.

        Returns dict of {horizon_min: {total, correct, accuracy}}.
        """
        horizons = self.cfg.get('eval_horizons_minutes', [1, 5, 15, 30, 60])
        max_gap = self.cfg.get('max_quote_gap_minutes_for_forward_eval', 2)

        trades = self.journal.get_all_trades(run_id, track=track,
                                             date_str=date_str)
        if not trades:
            return {}

        results = {h: {'total': 0, 'correct': 0} for h in horizons}

        for trade in trades:
            strike = trade['strike']
            action = trade['action']
            entry_mid = trade.get('entry_mid', 0)
            entry_ts = trade.get('entry_timestamp_utc')

            if not entry_ts or not entry_mid or entry_mid <= 0:
                continue

            quotes = self.journal.get_quotes_for_strike(
                run_id, strike, since_utc=entry_ts
            )
            if not quotes:
                continue

            entry_dt = datetime.fromisoformat(entry_ts)

            for horizon in horizons:
                # Find quote closest to entry + horizon minutes
                target_dt = entry_dt + __import__('datetime').timedelta(minutes=horizon)
                best_quote = None
                best_diff = float('inf')

                for q in quotes:
                    try:
                        q_dt = datetime.fromisoformat(q['timestamp_utc'])
                        diff = abs((q_dt - target_dt).total_seconds())
                        if diff < best_diff:
                            best_diff = diff
                            best_quote = q
                    except (ValueError, TypeError):
                        continue

                # Check if quote is within acceptable gap
                if best_quote and best_diff <= max_gap * 60:
                    future_mid = best_quote.get('mid', 0)
                    if future_mid > 0:
                        results[horizon]['total'] += 1
                        if action == 'BUY' and future_mid > entry_mid:
                            results[horizon]['correct'] += 1
                        elif action == 'SELL' and future_mid < entry_mid:
                            results[horizon]['correct'] += 1

        # Compute accuracy
        for h in horizons:
            r = results[h]
            r['accuracy'] = r['correct'] / r['total'] if r['total'] > 0 else 0.0

        return results

    def print_forward_accuracy(self, run_id: str, date_str: str = None):
        """Print forward accuracy for each track."""
        tracks = self.cfg.get('tracks', ['decision_time', 'all_signals'])

        print(f"\n{'─'*65}")
        print(f"  🎯 FORWARD ACCURACY (from quote tape)")
        print(f"{'─'*65}")

        for track in tracks:
            label = 'Track A' if track == 'decision_time' else 'Track B'
            results = self.compute_forward_accuracy(run_id, track=track,
                                                    date_str=date_str)
            if not results:
                print(f"\n  {label}: No data")
                continue

            print(f"\n  {label}:")
            print(f"    {'Horizon':>10}  {'Trades':>6}  {'Correct':>7}  {'Accuracy':>8}")
            for h in sorted(results):
                r = results[h]
                acc = f"{r['accuracy']*100:.1f}%" if r['total'] > 0 else 'N/A'
                print(f"    {h:>7} min  {r['total']:>6}  {r['correct']:>7}  {acc:>8}")

    # ================================================================== #
    #  CSV export                                                          #
    # ================================================================== #

    def export_trades_csv(self, run_id: str, filepath: str,
                          date_str: str = None):
        """Export all trades (both tracks) to CSV."""
        trades = self.journal.get_all_trades(run_id, date_str=date_str)
        if not trades:
            print("No trades to export.")
            return

        keys = trades[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for t in trades:
                writer.writerow(t)

        print(f"✓ Exported {len(trades)} trades → {filepath}")

    def export_snapshots_csv(self, run_id: str, filepath: str):
        """Export decision snapshots to CSV."""
        rows = self.journal.conn.execute(
            "SELECT * FROM decision_snapshots WHERE run_id = ? ORDER BY timestamp_utc",
            (run_id,)
        ).fetchall()

        if not rows:
            print("No snapshots to export.")
            return

        rows = [dict(r) for r in rows]
        keys = rows[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"✓ Exported {len(rows)} snapshots → {filepath}")

    # ================================================================== #
    #  Full EOD report                                                     #
    # ================================================================== #

    def full_report(self, run_id: str, date_str: str = None,
                    export_dir: str = None):
        """Print full EOD report and optionally export CSVs."""
        self.print_summary(run_id, date_str)
        self.print_forward_accuracy(run_id, date_str)

        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            d = date_str or datetime.now().strftime('%Y-%m-%d')
            self.export_trades_csv(
                run_id, os.path.join(export_dir, f'paper_trades_{d}.csv'),
                date_str=date_str
            )
            self.export_snapshots_csv(
                run_id, os.path.join(export_dir, f'decision_snapshots_{d}.csv')
            )
