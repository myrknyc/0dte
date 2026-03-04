import sqlite3
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 2
_ET = ZoneInfo('America/New_York')
_UTC = ZoneInfo('UTC')


def now_utc() -> str:
    """Current time as UTC ISO-8601 string."""
    return datetime.now(_UTC).isoformat()


def now_et() -> datetime:
    """Current time as timezone-aware ET datetime."""
    return datetime.now(_ET)


def to_utc_str(dt) -> str:
    """Convert any datetime to UTC ISO string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt.astimezone(_UTC).isoformat()


class PaperJournal:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'paper_trades.db'
            )
        self.db_path = os.path.abspath(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self):
        ver = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if ver == 0:
            self._create_tables_v1()
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()
        elif ver < SCHEMA_VERSION:
            self._migrate(ver)

    def _create_tables_v1(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id          TEXT PRIMARY KEY,
                start_time_utc  TEXT NOT NULL,
                end_time_utc    TEXT,
                config_json     TEXT,
                config_hash     TEXT,
                strategy_version TEXT,
                git_hash        TEXT,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id                TEXT PRIMARY KEY,
                run_id                  TEXT NOT NULL,
                track                   TEXT NOT NULL,
                date                    TEXT NOT NULL,
                decision_timestamp_utc  TEXT,
                entry_timestamp_utc     TEXT NOT NULL,
                ticker                  TEXT NOT NULL DEFAULT 'SPY',
                strike                  REAL NOT NULL,
                option_type             TEXT NOT NULL DEFAULT 'call',
                action                  TEXT NOT NULL,
                quantity                INTEGER NOT NULL DEFAULT 1,
                entry_bid               REAL,
                entry_ask               REAL,
                entry_mid               REAL,
                entry_fill_mid          REAL,
                entry_fill_touch        REAL,
                entry_fill_slippage     REAL,
                entry_spread            REAL,
                spot_entry              REAL,
                spot_age_seconds        REAL,
                confidence_entry        REAL,
                edge_entry              REAL,
                required_edge           REAL,
                model_price             REAL,
                std_error               REAL,
                market_iv               REAL,
                lambda_jump             REAL,
                v0                      REAL,
                kappa                   REAL,
                theta_v                 REAL,
                sigma_v                 REAL,
                rho                     REAL,
                calibration_flags       TEXT,
                bernoulli_violated      INTEGER DEFAULT 0,
                otm_dollars             REAL DEFAULT 0,
                entry_fees              REAL DEFAULT 0,
                status                  TEXT NOT NULL DEFAULT 'OPEN',
                exit_timestamp_utc      TEXT,
                exit_bid                REAL,
                exit_ask                REAL,
                exit_mid                REAL,
                exit_fill_mid           REAL,
                exit_fill_touch         REAL,
                exit_fill_slippage      REAL,
                exit_spread             REAL,
                spot_exit               REAL,
                exit_reason             TEXT,
                hold_minutes            REAL,
                gross_pnl_mid           REAL,
                net_pnl_mid             REAL,
                return_pct_mid          REAL,
                gross_pnl_touch         REAL,
                net_pnl_touch           REAL,
                return_pct_touch        REAL,
                gross_pnl_slippage      REAL,
                net_pnl_slippage        REAL,
                return_pct_slippage     REAL,
                exit_fees               REAL DEFAULT 0,
                mae                     REAL,
                mfe                     REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS paper_trade_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id        TEXT,
                track           TEXT NOT NULL,
                timestamp_utc   TEXT NOT NULL,
                event_type      TEXT NOT NULL,
                details_json    TEXT
            );

            CREATE TABLE IF NOT EXISTS decision_snapshots (
                snap_id                 TEXT PRIMARY KEY,
                run_id                  TEXT NOT NULL,
                track                   TEXT NOT NULL,
                timestamp_utc           TEXT NOT NULL,
                spot_price              REAL,
                intraday_move_pct       REAL,
                regime                  TEXT,
                n_signals               INTEGER,
                n_eligible              INTEGER,
                n_selected              INTEGER,
                n_entered               INTEGER,
                rejection_breakdown_json TEXT,
                signals_json            TEXT,
                context_json            TEXT,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS option_quotes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                timestamp_utc   TEXT NOT NULL,
                strike          REAL NOT NULL,
                option_type     TEXT NOT NULL DEFAULT 'call',
                bid             REAL,
                ask             REAL,
                mid             REAL,
                spread          REAL,
                spot            REAL,
                spot_age        REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_quotes_run_strike_ts
                ON option_quotes(run_id, strike, timestamp_utc);

            CREATE INDEX IF NOT EXISTS idx_trades_run_status
                ON paper_trades(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_trades_run_track
                ON paper_trades(run_id, track);
        """)

    def _migrate(self, from_version: int):
        """Incremental migrations.  Add new blocks as schema evolves."""
        if from_version < 2:
            # Phase 2.5: add regime label to snapshots
            try:
                self.conn.execute(
                    "ALTER TABLE decision_snapshots ADD COLUMN regime TEXT"
                )
            except Exception:
                pass  # column may already exist
            from_version = 2
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def start_run(self, config: dict, strategy_version: str = '',
                  git_hash: str = '', notes: str = '') -> str:
        """Create a new backtest run and return its run_id."""
        run_id = str(uuid.uuid4())
        config_json = json.dumps(config, default=str)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        self.conn.execute("""
            INSERT INTO backtest_runs
                (run_id, start_time_utc, config_json, config_hash,
                 strategy_version, git_hash, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, now_utc(), config_json, config_hash,
              strategy_version, git_hash, notes))
        self.conn.commit()
        return run_id

    def end_run(self, run_id: str):
        """Mark a run as finished."""
        self.conn.execute(
            "UPDATE backtest_runs SET end_time_utc = ? WHERE run_id = ?",
            (now_utc(), run_id))
        self.conn.commit()

   
    def open_trade(self, run_id: str, track: str, signal: dict,
                   entry_fill: dict, spot: float, timestamp_utc: str,
                   quantity: int = 1, entry_fees: float = 0.0,
                   decision_timestamp_utc: str = None) -> str:
        """Insert a new OPEN paper trade, return trade_id."""
        trade_id = str(uuid.uuid4())
        # Use ET session date (not UTC date) so a single trading day isn't
        # split by the midnight-UTC boundary
        utc_dt = datetime.fromisoformat(timestamp_utc)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=_UTC)
        today = utc_dt.astimezone(_ET).strftime('%Y-%m-%d')

        self.conn.execute("""
            INSERT INTO paper_trades
                (trade_id, run_id, track, date,
                 decision_timestamp_utc, entry_timestamp_utc,
                 ticker, strike, option_type, action, quantity,
                 entry_bid, entry_ask, entry_mid,
                 entry_fill_mid, entry_fill_touch, entry_fill_slippage,
                 entry_spread, spot_entry, spot_age_seconds,
                 confidence_entry, edge_entry, required_edge,
                 model_price, std_error, market_iv,
                 lambda_jump, v0, kappa, theta_v, sigma_v, rho,
                 calibration_flags, bernoulli_violated,
                 otm_dollars, entry_fees, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade_id, run_id, track, today,
            decision_timestamp_utc, timestamp_utc,
            signal.get('ticker', 'SPY'),
            signal.get('strike'),
            signal.get('option_type', 'call'),
            signal.get('action'),
            quantity,
            signal.get('market_bid'),
            signal.get('market_ask'),
            signal.get('market_mid'),
            entry_fill.get('mid'),
            entry_fill.get('touch'),
            entry_fill.get('slippage'),
            entry_fill.get('spread'),
            spot,
            signal.get('spot_age_seconds'),
            signal.get('confidence'),
            signal.get('edge'),
            signal.get('required_edge'),
            signal.get('model_price'),
            signal.get('std_error'),
            signal.get('market_iv'),
            signal.get('lambda_jump'),
            signal.get('v0'),
            signal.get('kappa'),
            signal.get('theta_v'),
            signal.get('sigma_v'),
            signal.get('rho'),
            json.dumps(signal.get('calibration_flags')) if signal.get('calibration_flags') else None,
            1 if signal.get('bernoulli_violated') else 0,
            signal.get('otm_dollars', 0.0),
            entry_fees,
            'OPEN',
        ))
        self.conn.commit()

        self._log_event(trade_id, track, 'ENTERED', {
            'fill': entry_fill, 'spot': spot,
        })

        return trade_id

    def close_trade(self, trade_id: str, exit_bid: float, exit_ask: float,
                    exit_fill: dict, spot_exit: float, timestamp_utc: str,
                    exit_reason: str, pnl_mid: dict, pnl_touch: dict,
                    pnl_slippage: dict, hold_minutes: float,
                    exit_fees: float = 0.0,
                    mae: float = None, mfe: float = None):
        """Close an OPEN trade with full exit data and PnL."""
        exit_mid = (exit_bid + exit_ask) / 2.0
        spread_exit = exit_ask - exit_bid

        self.conn.execute("""
            UPDATE paper_trades SET
                exit_timestamp_utc = ?,
                exit_bid = ?, exit_ask = ?, exit_mid = ?,
                exit_fill_mid = ?, exit_fill_touch = ?,
                exit_fill_slippage = ?,
                exit_spread = ?, spot_exit = ?,
                exit_reason = ?, hold_minutes = ?,
                gross_pnl_mid = ?, net_pnl_mid = ?,
                return_pct_mid = ?,
                gross_pnl_touch = ?, net_pnl_touch = ?,
                return_pct_touch = ?,
                gross_pnl_slippage = ?, net_pnl_slippage = ?,
                return_pct_slippage = ?,
                exit_fees = ?, mae = ?, mfe = ?,
                status = 'CLOSED'
            WHERE trade_id = ?
        """, (
            timestamp_utc,
            exit_bid, exit_ask, exit_mid,
            exit_fill.get('mid'), exit_fill.get('touch'),
            exit_fill.get('slippage'),
            spread_exit, spot_exit,
            exit_reason, hold_minutes,
            pnl_mid.get('gross_pnl'), pnl_mid.get('net_pnl'),
            pnl_mid.get('return_pct'),
            pnl_touch.get('gross_pnl'), pnl_touch.get('net_pnl'),
            pnl_touch.get('return_pct'),
            pnl_slippage.get('gross_pnl'), pnl_slippage.get('net_pnl'),
            pnl_slippage.get('return_pct'),
            exit_fees, mae, mfe,
            trade_id,
        ))
        self.conn.commit()

        track = self.conn.execute(
            "SELECT track FROM paper_trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()['track']
        self._log_event(trade_id, track, 'EXITED', {
            'reason': exit_reason, 'pnl_touch': pnl_touch,
            'hold_minutes': hold_minutes,
        })

    def get_open_trades(self, run_id: str, track: str = None) -> List[Dict]:
        """Get all OPEN trades, optionally filtered by track."""
        q = "SELECT * FROM paper_trades WHERE run_id = ? AND status = 'OPEN'"
        params = [run_id]
        if track:
            q += " AND track = ?"
            params.append(track)
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def get_closed_trades(self, run_id: str, track: str = None,
                          date_str: str = None) -> List[Dict]:
        """Get CLOSED trades with optional filters."""
        q = "SELECT * FROM paper_trades WHERE run_id = ? AND status = 'CLOSED'"
        params = [run_id]
        if track:
            q += " AND track = ?"
            params.append(track)
        if date_str:
            q += " AND date = ?"
            params.append(date_str)
        q += " ORDER BY entry_timestamp_utc"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def get_all_trades(self, run_id: str, track: str = None,
                       date_str: str = None) -> List[Dict]:
        """Get all trades (open + closed)."""
        q = "SELECT * FROM paper_trades WHERE run_id = ?"
        params = [run_id]
        if track:
            q += " AND track = ?"
            params.append(track)
        if date_str:
            q += " AND date = ?"
            params.append(date_str)
        q += " ORDER BY entry_timestamp_utc"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def has_open_trade(self, run_id: str, track: str, strike: float,
                       action: str) -> bool:
        """Check if an OPEN trade exists for (track, strike, action)."""
        row = self.conn.execute("""
            SELECT COUNT(*) FROM paper_trades
            WHERE run_id = ? AND track = ? AND strike = ? AND action = ?
                  AND status = 'OPEN'
        """, (run_id, track, strike, action)).fetchone()
        return row[0] > 0

    def _log_event(self, trade_id: str, track: str, event_type: str,
                   details: dict = None):
        self.conn.execute("""
            INSERT INTO paper_trade_events
                (trade_id, track, timestamp_utc, event_type, details_json)
            VALUES (?, ?, ?, ?, ?)
        """, (trade_id, track, now_utc(), event_type,
              json.dumps(details, default=str) if details else None))
        self.conn.commit()

    def log_skip(self, track: str, signal: dict, reason: str):
        """Log a SKIPPED event (signal rejected by filters)."""
        self.conn.execute("""
            INSERT INTO paper_trade_events
                (trade_id, track, timestamp_utc, event_type, details_json)
            VALUES (?, ?, ?, 'SKIPPED', ?)
        """, (None, track, now_utc(), json.dumps({
            'reason': reason,
            'strike': signal.get('strike'),
            'action': signal.get('action'),
            'confidence': signal.get('confidence'),
            'edge': signal.get('edge'),
        }, default=str)))
        self.conn.commit()

    def record_snapshot(self, run_id: str, track: str, timestamp_utc: str,
                        spot_price: float, n_signals: int, n_eligible: int,
                        n_selected: int, n_entered: int,
                        rejection_breakdown: dict = None,
                        signals_json: str = None,
                        context: dict = None,
                        intraday_move_pct: float = None,
                        regime: str = None):
        snap_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO decision_snapshots
                (snap_id, run_id, track, timestamp_utc, spot_price,
                 intraday_move_pct, regime,
                 n_signals, n_eligible, n_selected, n_entered,
                 rejection_breakdown_json, signals_json, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (snap_id, run_id, track, timestamp_utc, spot_price,
              intraday_move_pct, regime,
              n_signals, n_eligible, n_selected, n_entered,
              json.dumps(rejection_breakdown, default=str) if rejection_breakdown else None,
              signals_json,
              json.dumps(context, default=str) if context else None))
        self.conn.commit()
        return snap_id

  
    def record_quotes(self, run_id: str, timestamp_utc: str,
                      quotes: List[Dict]):
        if not quotes:
            return
        self.conn.executemany("""
            INSERT INTO option_quotes
                (run_id, timestamp_utc, strike, option_type,
                 bid, ask, mid, spread, spot, spot_age)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            run_id, timestamp_utc,
            q.get('strike'), q.get('option_type', 'call'),
            q.get('bid'), q.get('ask'), q.get('mid'),
            q.get('spread'), q.get('spot'), q.get('spot_age'),
        ) for q in quotes])
        self.conn.commit()

    def get_quotes_for_strike(self, run_id: str, strike: float,
                              since_utc: str = None) -> List[Dict]:
        """Get quote tape entries for a strike, optionally since a time."""
        q = "SELECT * FROM option_quotes WHERE run_id = ? AND strike = ?"
        params = [run_id, strike]
        if since_utc:
            q += " AND timestamp_utc >= ?"
            params.append(since_utc)
        q += " ORDER BY timestamp_utc"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def get_latest_quote(self, run_id: str, strike: float) -> Optional[Dict]:
        """Get the most recent quote for a strike."""
        row = self.conn.execute("""
            SELECT * FROM option_quotes
            WHERE run_id = ? AND strike = ?
            ORDER BY timestamp_utc DESC LIMIT 1
        """, (run_id, strike)).fetchone()
        return dict(row) if row else None

    def trade_summary(self, run_id: str, track: str = None) -> Dict:
        """Aggregate stats for closed trades."""
        q_base = "SELECT * FROM paper_trades WHERE run_id = ? AND status = 'CLOSED'"
        params = [run_id]
        if track:
            q_base += " AND track = ?"
            params.append(track)

        trades = [dict(r) for r in self.conn.execute(q_base, params).fetchall()]
        if not trades:
            return {'total': 0}

        wins = [t for t in trades if (t.get('net_pnl_touch') or 0) > 0]
        losses = [t for t in trades if (t.get('net_pnl_touch') or 0) <= 0]
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']

        total_pnl = sum(t.get('net_pnl_touch') or 0 for t in trades)
        gross_wins = sum(t.get('net_pnl_touch') or 0 for t in wins)
        gross_losses = abs(sum(t.get('net_pnl_touch') or 0 for t in losses))

        return {
            'total': len(trades),
            'buys': len(buys),
            'sells': len(sells),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl_touch': total_pnl,
            'profit_factor': gross_wins / gross_losses if gross_losses > 0 else float('inf'),
            'expectancy': total_pnl / len(trades) if trades else 0,
            'avg_hold_minutes': sum(t.get('hold_minutes') or 0 for t in trades) / len(trades),
            'avg_edge': sum(t.get('edge_entry') or 0 for t in trades) / len(trades),
            'avg_confidence': sum(t.get('confidence_entry') or 0 for t in trades) / len(trades),
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
