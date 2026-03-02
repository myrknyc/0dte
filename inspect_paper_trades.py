"""
Paper Trade Inspector — view all trades, verify fills vs real quotes.

Usage:
    python inspect_paper_trades.py                 # latest run, all trades
    python inspect_paper_trades.py --track A       # Track A only
    python inspect_paper_trades.py --track B       # Track B only
    python inspect_paper_trades.py --export        # also export CSVs
    python inspect_paper_trades.py --run 0         # first run (0-indexed)
    python inspect_paper_trades.py --eod           # print EOD summary
"""

import sqlite3
import csv
import sys
import os
import argparse
from datetime import datetime

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DB_PATH = os.path.join(os.path.dirname(__file__), 'paper_trades.db')


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_runs(conn):
    runs = conn.execute(
        "SELECT run_id, start_time_utc, end_time_utc, strategy_version "
        "FROM backtest_runs ORDER BY start_time_utc"
    ).fetchall()
    print(f"\n{'='*70}")
    print(f"  BACKTEST RUNS ({len(runs)} total)")
    print(f"{'='*70}")
    for i, r in enumerate(runs):
        end = r['end_time_utc'] or '(still open)'
        print(f"  [{i}] {r['run_id'][:16]}...  {r['start_time_utc'][:19]}  ->  {end[:19]}")
    return runs


def show_trades(conn, run_id, track_filter=None):
    q = "SELECT * FROM paper_trades WHERE run_id = ?"
    params = [run_id]
    if track_filter:
        q += " AND track = ?"
        params.append(track_filter)
    q += " ORDER BY entry_timestamp_utc"

    trades = conn.execute(q, params).fetchall()
    trades = [dict(t) for t in trades]

    for track in ['decision_time', 'all_signals']:
        if track_filter and track != track_filter:
            continue

        track_trades = [t for t in trades if t['track'] == track]
        label = 'TRACK A (decision_time)' if track == 'decision_time' else 'TRACK B (all_signals)'

        print(f"\n{'-'*70}")
        print(f"  {label} -- {len(track_trades)} trades")
        print(f"{'-'*70}")

        if not track_trades:
            print("  (none)")
            continue

        print(f"  {'Time (UTC)':>19}  {'Strike':>6}  {'Act':>4}  {'Status':>8}  "
              f"{'Entry Mid':>9}  {'Exit Mid':>8}  {'PnL(touch)':>10}  {'Exit Reason':>12}")
        print(f"  {'-'*19}  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*12}")

        total_pnl = 0
        wins = 0
        for t in track_trades:
            ts = (t.get('entry_timestamp_utc') or '')[:19]
            strike = t.get('strike', 0)
            action = t.get('action', '?')
            status = t.get('status', '?')
            entry_mid = t.get('entry_mid', 0) or 0
            exit_mid = t.get('exit_mid') or 0
            pnl = t.get('net_pnl_touch') or 0
            reason = t.get('exit_reason') or ''
            total_pnl += pnl
            if pnl > 0:
                wins += 1

            pnl_str = f"${pnl:+.2f}" if status == 'CLOSED' else '--'
            exit_str = f"${exit_mid:.2f}" if exit_mid else '--'

            print(f"  {ts:>19}  {strike:>6.0f}  {action:>4}  {status:>8}  "
                  f"${entry_mid:>8.4f}  {exit_str:>8}  {pnl_str:>10}  {reason:>12}")

        closed = [t for t in track_trades if t['status'] == 'CLOSED']
        if closed:
            wr = wins / len(closed) * 100 if closed else 0
            print(f"\n  Summary: {len(closed)} closed, {wins}W/{len(closed)-wins}L "
                  f"({wr:.0f}% WR), Net PnL: ${total_pnl:+.2f}")


def show_quote_tape(conn, run_id, strike=None, limit=20):
    """Show recorded quotes for verification against real prices."""
    q = "SELECT * FROM option_quotes WHERE run_id = ?"
    params = [run_id]
    if strike:
        q += " AND strike = ?"
        params.append(strike)
    q += " ORDER BY timestamp_utc DESC LIMIT ?"
    params.append(limit)

    quotes = conn.execute(q, params).fetchall()
    quotes = [dict(q) for q in quotes]

    print(f"\n{'-'*70}")
    print(f"  QUOTE TAPE (last {limit} entries{f' for strike {strike}' if strike else ''})")
    print(f"{'-'*70}")
    print(f"  {'Time (UTC)':>19}  {'Strike':>6}  {'Bid':>6}  {'Ask':>6}  {'Mid':>6}  {'Spread':>6}  {'Spot':>7}")
    print(f"  {'-'*19}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")

    for q in reversed(quotes):
        ts = (q.get('timestamp_utc') or '')[:19]
        print(f"  {ts:>19}  {q['strike']:>6.0f}  "
              f"${q['bid']:>5.2f}  ${q['ask']:>5.2f}  ${q['mid']:>5.2f}  "
              f"${q['spread']:>5.2f}  ${q['spot']:>6.2f}")


def show_trade_detail(conn, trade_id):
    """Full detail for one trade, including 3-tier fills."""
    t = conn.execute("SELECT * FROM paper_trades WHERE trade_id = ?",
                     (trade_id,)).fetchone()
    if not t:
        print(f"Trade {trade_id} not found.")
        return
    t = dict(t)

    print(f"\n{'='*70}")
    print(f"  TRADE DETAIL: {t['trade_id'][:16]}...")
    print(f"{'='*70}")
    print(f"  Track     : {t['track']}")
    print(f"  Strike    : {t['strike']}")
    print(f"  Action    : {t['action']}")
    print(f"  Status    : {t['status']}")
    print(f"  Entry Time: {t['entry_timestamp_utc']}")
    print(f"  Exit Time : {t.get('exit_timestamp_utc', '--')}")
    print(f"  Exit Reason: {t.get('exit_reason', '--')}")
    print(f"  Hold (min): {t.get('hold_minutes', '--')}")
    print()
    print(f"  Entry Fills:")
    print(f"    Mid      : ${t['entry_mid']:.4f}")
    print(f"    Touch    : ${t['entry_touch']:.4f}")
    print(f"    Slippage : ${t['entry_slippage']:.4f}")
    print(f"    Spread   : ${t.get('entry_spread', 0):.4f}")
    print(f"    Fees     : ${t.get('entry_fees', 0):.4f}")
    print()
    if t['status'] == 'CLOSED':
        print(f"  Exit Fills:")
        print(f"    Mid      : ${t.get('exit_mid', 0):.4f}")
        print(f"    Touch    : ${t.get('exit_touch', 0):.4f}")
        print(f"    Slippage : ${t.get('exit_slippage', 0):.4f}")
        print(f"    Fees     : ${t.get('exit_fees', 0):.4f}")
        print()
        print(f"  PnL (3-tier):")
        print(f"    Mid      : ${t.get('net_pnl_mid', 0):+.2f}")
        print(f"    Touch    : ${t.get('net_pnl_touch', 0):+.2f}")
        print(f"    Slippage : ${t.get('net_pnl_slippage', 0):+.2f}")
        print()
        print(f"  MAE/MFE:")
        print(f"    MAE (max drawdown): ${t.get('mae', 0):.4f}")
        print(f"    MFE (max runup)   : ${t.get('mfe', 0):.4f}")

    # Show matching quotes for verification
    print(f"\n  Quote history for strike {t['strike']}:")
    quotes = conn.execute(
        "SELECT * FROM option_quotes WHERE run_id=? AND strike=? "
        "ORDER BY timestamp_utc",
        (t['run_id'], t['strike'])
    ).fetchall()
    for q in quotes:
        ts = q['timestamp_utc'][:19]
        print(f"    {ts}  bid=${q['bid']:.2f}  ask=${q['ask']:.2f}  "
              f"mid=${q['mid']:.2f}  spot=${q['spot']:.2f}")


def export_csv(conn, run_id):
    """Export trades to CSV for external analysis."""
    trades = conn.execute(
        "SELECT * FROM paper_trades WHERE run_id = ? ORDER BY entry_timestamp_utc",
        (run_id,)
    ).fetchall()
    if not trades:
        print("No trades to export.")
        return

    fname = f"paper_trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades = [dict(t) for t in trades]
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
    print(f"[OK] Exported {len(trades)} trades -> {fname}")


def main():
    parser = argparse.ArgumentParser(description='Inspect paper trades')
    parser.add_argument('--track', choices=['A', 'B', 'a', 'b'], help='Filter by track')
    parser.add_argument('--run', type=int, default=-1, help='Run index (default: latest)')
    parser.add_argument('--export', action='store_true', help='Export trades to CSV')
    parser.add_argument('--quotes', type=float, help='Show quote tape for a strike')
    parser.add_argument('--trade', type=str, help='Show full detail for a trade_id')
    parser.add_argument('--eod', action='store_true', help='Print EOD summary')
    args = parser.parse_args()

    conn = connect()
    runs = list_runs(conn)

    if not runs:
        print("\nNo runs found. Run continuous_monitor.py first.")
        return

    run = runs[args.run]
    run_id = run['run_id']
    print(f"\n  → Using run [{args.run}]: {run_id[:16]}...")

    track_filter = None
    if args.track:
        track_filter = 'decision_time' if args.track.upper() == 'A' else 'all_signals'

    if args.trade:
        show_trade_detail(conn, args.trade)
    elif args.quotes:
        show_quote_tape(conn, run_id, strike=args.quotes)
    elif args.eod:
        from trading.eod_reporter import EODReporter
        from trading.paper_journal import PaperJournal
        journal = PaperJournal(db_path=DB_PATH)
        reporter = EODReporter(journal)
        reporter.full_report(run_id)
    else:
        show_trades(conn, run_id, track_filter)
        if not track_filter or track_filter == 'decision_time':
            show_quote_tape(conn, run_id, limit=10)

    if args.export:
        export_csv(conn, run_id)

    conn.close()


if __name__ == '__main__':
    main()
