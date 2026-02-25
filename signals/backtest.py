"""
Backtest Analysis — reads signal_log.db and computes performance metrics.

Usage:
    python -m signals.backtest                 # Analyze all signals
    python -m signals.backtest 2026-02-24      # Analyze a specific date
    python -m signals.backtest --last 7        # Analyze last 7 days
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'signal_log.db')


def _connect(db_path: str = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signal log not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def signal_summary(db_path: str = None, date_from: str = None,
                   date_to: str = None) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all logged signals.

    Returns dict with: total, buys, sells, holds, avg_edge, avg_confidence,
    edge_by_hour, strike_distribution, spread_stats.
    """
    conn = _connect(db_path)

    # Build WHERE clause
    conditions = []
    params = []
    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to + " 23:59:59")
    where = " WHERE " + " AND ".join(conditions) if conditions else ""

    # --- Core counts ---
    row = conn.execute(
        f"SELECT COUNT(*) as total,"
        f" SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) as buys,"
        f" SUM(CASE WHEN action='SELL' THEN 1 ELSE 0 END) as sells,"
        f" SUM(CASE WHEN action='HOLD' THEN 1 ELSE 0 END) as holds"
        f" FROM signals{where}", params
    ).fetchone()

    total = row['total']
    if total == 0:
        print("No signals found in the specified range.")
        conn.close()
        return {"total": 0}

    buys = row['buys'] or 0
    sells = row['sells'] or 0
    holds = row['holds'] or 0

    # --- Avg edge & confidence (actionable signals only) ---
    act_cond = conditions + ["action != 'HOLD'"]
    act_where = " WHERE " + " AND ".join(act_cond) if act_cond else ""
    avg_row = conn.execute(
        f"SELECT AVG(edge) as avg_edge, AVG(confidence) as avg_conf,"
        f" MIN(edge) as min_edge, MAX(edge) as max_edge,"
        f" AVG(spread) as avg_spread"
        f" FROM signals{act_where}", params
    ).fetchone()

    avg_edge = avg_row['avg_edge'] or 0.0
    avg_conf = avg_row['avg_conf'] or 0.0
    min_edge = avg_row['min_edge'] or 0.0
    max_edge = avg_row['max_edge'] or 0.0
    avg_spread = avg_row['avg_spread'] or 0.0

    # --- Edge by hour-of-day ---
    edge_by_hour = {}
    rows = conn.execute(
        f"SELECT CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) as hour,"
        f" AVG(edge) as avg_edge, COUNT(*) as cnt"
        f" FROM signals{act_where}"
        f" GROUP BY hour ORDER BY hour", params
    ).fetchall()
    for r in rows:
        edge_by_hour[r['hour']] = {'avg_edge': r['avg_edge'], 'count': r['cnt']}

    # --- Model accuracy vs. spread ---
    spread_rows = conn.execute(
        f"SELECT"
        f" CASE WHEN spread < 0.05 THEN 'tight (<$0.05)'"
        f"      WHEN spread < 0.10 THEN 'medium ($0.05-0.10)'"
        f"      ELSE 'wide (>$0.10)' END as bucket,"
        f" COUNT(*) as cnt, AVG(edge) as avg_edge, AVG(confidence) as avg_conf"
        f" FROM signals{act_where}"
        f" GROUP BY bucket", params
    ).fetchall()
    spread_stats = {r['bucket']: {'count': r['cnt'], 'avg_edge': r['avg_edge'],
                                   'avg_conf': r['avg_conf']} for r in spread_rows}

    # --- Outcome stats (if any) ---
    outcome_row = conn.execute(
        "SELECT COUNT(*) as cnt,"
        " SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins,"
        " AVG(pnl) as avg_pnl, SUM(pnl) as total_pnl,"
        " AVG(realized_edge) as avg_real_edge"
        " FROM outcomes"
    ).fetchone()
    outcomes_count = outcome_row['cnt'] or 0
    wins = outcome_row['wins'] or 0
    losses = outcomes_count - wins
    win_rate = (wins / outcomes_count * 100) if outcomes_count > 0 else None

    # --- Sessions ---
    session_count = conn.execute(
        f"SELECT COUNT(DISTINCT session_id) FROM signals{where}", params
    ).fetchone()[0]

    # --- Date range ---
    date_range = conn.execute(
        f"SELECT MIN(timestamp), MAX(timestamp) FROM signals{where}", params
    ).fetchone()

    conn.close()

    stats = {
        'total': total,
        'buys': buys,
        'sells': sells,
        'holds': holds,
        'sessions': session_count,
        'date_range': (date_range[0], date_range[1]),
        'avg_edge': avg_edge,
        'avg_confidence': avg_conf,
        'min_edge': min_edge,
        'max_edge': max_edge,
        'avg_spread': avg_spread,
        'edge_by_hour': edge_by_hour,
        'spread_stats': spread_stats,
        'outcomes_count': outcomes_count,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_pnl': outcome_row['avg_pnl'],
        'total_pnl': outcome_row['total_pnl'],
        'avg_realized_edge': outcome_row['avg_real_edge'],
    }

    return stats


def print_report(stats: Dict[str, Any]) -> None:
    """Pretty-print the backtest report."""
    if stats.get('total', 0) == 0:
        return

    print("=" * 60)
    print("📊 BACKTEST ANALYSIS REPORT")
    print("=" * 60)

    dr = stats.get('date_range', (None, None))
    if dr[0]:
        print(f"  Period     : {dr[0][:10]} → {dr[1][:10]}")
    print(f"  Sessions   : {stats['sessions']}")
    print(f"  Total sigs : {stats['total']:,}")

    # --- Signal breakdown ---
    print(f"\n  Signal Breakdown:")
    print(f"    BUY  : {stats['buys']:4d}  ({stats['buys']/stats['total']*100:.1f}%)")
    print(f"    SELL : {stats['sells']:4d}  ({stats['sells']/stats['total']*100:.1f}%)")
    print(f"    HOLD : {stats['holds']:4d}  ({stats['holds']/stats['total']*100:.1f}%)")

    # --- Edge stats ---
    print(f"\n  Edge Statistics (actionable signals):")
    print(f"    Avg edge       : {stats['avg_edge']*100:.2f}%")
    print(f"    Min / Max edge : {stats['min_edge']*100:.2f}% / {stats['max_edge']*100:.2f}%")
    print(f"    Avg confidence : {stats['avg_confidence']*100:.1f}%")
    print(f"    Avg spread     : ${stats['avg_spread']:.4f}")

    # --- Edge by hour ---
    if stats.get('edge_by_hour'):
        print(f"\n  Edge by Hour-of-Day:")
        for hour in sorted(stats['edge_by_hour']):
            info = stats['edge_by_hour'][hour]
            bar = "█" * max(1, int(info['avg_edge'] * 500))
            print(f"    {hour:02d}:00  {info['avg_edge']*100:+.2f}%  "
                  f"(n={info['count']:3d})  {bar}")

    # --- Spread buckets ---
    if stats.get('spread_stats'):
        print(f"\n  Edge by Spread Bucket:")
        for bucket, info in stats['spread_stats'].items():
            print(f"    {bucket:25s}  edge={info['avg_edge']*100:.2f}%  "
                  f"conf={info['avg_conf']*100:.0f}%  (n={info['count']})")

    # --- Outcomes ---
    if stats.get('outcomes_count', 0) > 0:
        print(f"\n  Realized Outcomes:")
        print(f"    Trades    : {stats['outcomes_count']}")
        print(f"    Wins      : {stats['wins']}")
        print(f"    Losses    : {stats['losses']}")
        print(f"    Win rate  : {stats['win_rate']:.1f}%")
        if stats['avg_pnl'] is not None:
            print(f"    Avg PnL   : ${stats['avg_pnl']:.2f}")
            print(f"    Total PnL : ${stats['total_pnl']:.2f}")
        if stats['avg_realized_edge'] is not None:
            print(f"    Avg realized edge: {stats['avg_realized_edge']*100:.2f}%")
    else:
        print(f"\n  ℹ No outcomes recorded yet.")
        print(f"    Use SignalLogger.record_outcome() after trades close.")

    print("=" * 60)


def main():
    """CLI entry point."""
    date_from = None
    date_to = None

    args = sys.argv[1:]

    if len(args) == 1 and not args[0].startswith('--'):
        # Single date
        date_from = args[0]
        date_to = args[0]
    elif len(args) == 2 and args[0] == '--last':
        days = int(args[1])
        date_from = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    elif len(args) == 2:
        date_from = args[0]
        date_to = args[1]

    stats = signal_summary(date_from=date_from, date_to=date_to)
    print_report(stats)


if __name__ == '__main__':
    main()
