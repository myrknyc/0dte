"""
Signal Logger — SQLite-backed signal & outcome logging for 0DTE backtesting.

Schema versioned via PRAGMA user_version. Migrations applied automatically on init.
"""

import sqlite3
import csv
import uuid
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any


# Current schema version — bump this and add a migration when changing tables
SCHEMA_VERSION = 2


class SignalLogger:
    """Logs every trading signal with full model + market context.

    One SignalLogger instance = one session (UUID generated at construction).
    Designed for both one-shot (signal_generator.py) and long-running
    (continuous_monitor.py) use.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default: signal_log.db next to this file
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_log.db")

        self.db_path = db_path
        self.session_id = str(uuid.uuid4())
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # dict-like access on rows
        self.conn.execute("PRAGMA journal_mode=WAL")  # concurrent-read safe

        self._ensure_schema()

    # ------------------------------------------------------------------ #
    #  Schema creation & migration                                        #
    # ------------------------------------------------------------------ #

    def _ensure_schema(self):
        """Check user_version and create/migrate tables as needed."""
        current_version = self.conn.execute("PRAGMA user_version").fetchone()[0]

        if current_version == 0:
            # Fresh database — create everything
            self._create_tables_v1()
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()
        elif current_version < SCHEMA_VERSION:
            # Run incremental migrations
            self._migrate(current_version)
        # else: already at latest version

    def _create_tables_v1(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                ticker          TEXT    NOT NULL,
                strike          REAL    NOT NULL,
                option_type     TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                edge            REAL,
                confidence      REAL,
                model_price     REAL,
                market_bid      REAL,
                market_ask      REAL,
                market_mid      REAL,
                spread          REAL,
                std_error       REAL,
                spot_price      REAL,
                time_to_expiry  REAL,
                minutes_to_expiry INTEGER,
                iv              REAL,
                n_paths         INTEGER,
                vr_factor       REAL,
                reason          TEXT,
                source          TEXT,
                market_iv       REAL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_lookup
                ON signals (timestamp, ticker, action);

            CREATE INDEX IF NOT EXISTS idx_signals_session
                ON signals (session_id);

            CREATE TABLE IF NOT EXISTS outcomes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id       INTEGER NOT NULL REFERENCES signals(id),
                entry_price     REAL,
                exit_price      REAL,
                exit_time       TEXT,
                exit_reason     TEXT,
                exit_spot_price REAL,
                exit_iv         REAL,
                pnl             REAL,
                realized_edge   REAL,
                correct         INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_outcomes_signal
                ON outcomes (signal_id);
        """)

    def _migrate(self, from_version: int):
        """Run incremental migrations from from_version → SCHEMA_VERSION."""
        if from_version < 2:
            self.conn.execute("ALTER TABLE signals ADD COLUMN market_iv REAL")
            from_version = 2
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def log_signal(
        self,
        ticker: str,
        strike: float,
        option_type: str,
        action: str,
        edge: float,
        confidence: float,
        model_price: float,
        market_bid: float,
        market_ask: float,
        market_mid: float,
        spread: float,
        std_error: float,
        spot_price: float,
        time_to_expiry: float,
        iv: float,
        n_paths: int = None,
        vr_factor: float = None,
        reason: str = "",
        source: str = "",
        minutes_to_expiry: int = None,
        market_iv: float = None,
    ) -> int:
        """Insert a signal row and return its id."""
        # Auto-compute minutes if not provided
        if minutes_to_expiry is None and time_to_expiry is not None:
            # T is in trading years; 1 year = 252 days × 390 min
            minutes_to_expiry = int(round(time_to_expiry * 252 * 390))

        cur = self.conn.execute(
            """
            INSERT INTO signals (
                session_id, timestamp, ticker, strike, option_type,
                action, edge, confidence, model_price,
                market_bid, market_ask, market_mid, spread, std_error,
                spot_price, time_to_expiry, minutes_to_expiry, iv,
                n_paths, vr_factor, reason, source, market_iv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                datetime.now().isoformat(),
                ticker,
                strike,
                option_type,
                action,
                edge,
                confidence,
                model_price,
                market_bid,
                market_ask,
                market_mid,
                spread,
                std_error,
                spot_price,
                time_to_expiry,
                minutes_to_expiry,
                iv,
                n_paths,
                vr_factor,
                reason,
                source,
                market_iv,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_outcome(
        self,
        signal_id: int,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        correct: bool,
        exit_spot_price: float = None,
        exit_iv: float = None,
        realized_edge: float = None,
        exit_time: str = None,
    ) -> int:
        """Attach a trade outcome to a previously logged signal."""
        if exit_time is None:
            exit_time = datetime.now().isoformat()

        cur = self.conn.execute(
            """
            INSERT INTO outcomes (
                signal_id, entry_price, exit_price, exit_time,
                exit_reason, exit_spot_price, exit_iv,
                pnl, realized_edge, correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                entry_price,
                exit_price,
                exit_time,
                exit_reason,
                exit_spot_price,
                exit_iv,
                pnl,
                realized_edge,
                int(correct),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------ #
    #  Querying                                                           #
    # ------------------------------------------------------------------ #

    def get_signals(
        self,
        date_str: str = None,
        ticker: str = None,
        action: str = None,
        session_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query signals with optional filters and pagination."""
        clauses: List[str] = []
        params: List[Any] = []

        if date_str:
            clauses.append("timestamp LIKE ?")
            params.append(f"{date_str}%")
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT * FROM signals{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_outcomes(self, signal_id: int = None) -> List[Dict[str, Any]]:
        """Get outcomes, optionally filtered by signal_id."""
        if signal_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM outcomes WHERE signal_id = ?", (signal_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM outcomes").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def export_csv(
        self,
        filepath: str,
        date_from: str = None,
        date_to: str = None,
    ):
        """Export signals (left-joined with outcomes) to CSV."""
        clauses: List[str] = []
        params: List[Any] = []

        if date_from:
            clauses.append("s.timestamp >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("s.timestamp <= ?")
            params.append(date_to)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        query = f"""
            SELECT s.*, o.entry_price, o.exit_price, o.exit_time,
                   o.exit_reason, o.exit_spot_price, o.exit_iv,
                   o.pnl, o.realized_edge, o.correct
            FROM signals s
            LEFT JOIN outcomes o ON o.signal_id = s.id
            {where}
            ORDER BY s.timestamp
        """

        rows = self.conn.execute(query, params).fetchall()

        if not rows:
            print("No signals to export.")
            return

        keys = rows[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        print(f"✓ Exported {len(rows)} rows → {filepath}")

    # ------------------------------------------------------------------ #
    #  Summary / accuracy stats                                           #
    # ------------------------------------------------------------------ #

    def _where(self, conditions: List[str], base_params: List[Any]) -> tuple:
        """Build a WHERE clause from a list of conditions."""
        if not conditions:
            return "", []
        return " WHERE " + " AND ".join(conditions), list(base_params)

    def summary(self, date_str: str = None) -> Dict[str, Any]:
        """Print and return accuracy statistics.

        Returns:
            dict with keys: total, buys, sells, holds, outcomes_recorded,
                            wins, losses, win_rate, avg_edge, avg_confidence,
                            avg_realized_edge
        """
        # Build base filter
        base_conds: List[str] = []
        base_params: List[Any] = []
        if date_str:
            base_conds.append("timestamp LIKE ?")
            base_params.append(f"{date_str}%")

        where, params = self._where(base_conds, base_params)

        # Signal counts
        total = self.conn.execute(
            f"SELECT COUNT(*) FROM signals{where}", params
        ).fetchone()[0]

        buy_where, buy_params = self._where(base_conds + ["action='BUY'"], base_params)
        buys = self.conn.execute(
            f"SELECT COUNT(*) FROM signals{buy_where}", buy_params
        ).fetchone()[0]

        sell_where, sell_params = self._where(base_conds + ["action='SELL'"], base_params)
        sells = self.conn.execute(
            f"SELECT COUNT(*) FROM signals{sell_where}", sell_params
        ).fetchone()[0]

        holds = total - buys - sells

        # Averages on actionable signals
        act_where, act_params = self._where(base_conds + ["action != 'HOLD'"], base_params)
        avg_row = self.conn.execute(
            f"SELECT AVG(edge), AVG(confidence) FROM signals{act_where}",
            act_params,
        ).fetchone()
        avg_edge = avg_row[0] or 0.0
        avg_confidence = avg_row[1] or 0.0

        # Outcome stats
        if date_str:
            outcome_q = """
                SELECT COUNT(*), SUM(CASE WHEN o.correct=1 THEN 1 ELSE 0 END),
                       AVG(o.realized_edge)
                FROM outcomes o JOIN signals s ON o.signal_id = s.id
                WHERE s.timestamp LIKE ?
            """
            o_params = base_params
        else:
            outcome_q = """
                SELECT COUNT(*), SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END),
                       AVG(realized_edge)
                FROM outcomes
            """
            o_params = []

        o_row = self.conn.execute(outcome_q, o_params).fetchone()
        outcomes_recorded = o_row[0] or 0
        wins = o_row[1] or 0
        losses = outcomes_recorded - wins
        win_rate = (wins / outcomes_recorded * 100) if outcomes_recorded > 0 else 0.0
        avg_realized_edge = o_row[2] or 0.0

        stats = {
            "total": total,
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "avg_edge": avg_edge,
            "avg_confidence": avg_confidence,
            "outcomes_recorded": outcomes_recorded,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_realized_edge": avg_realized_edge,
        }

        # Pretty-print
        label = f" ({date_str})" if date_str else ""
        print(f"\n{'='*50}")
        print(f"📊 SIGNAL LOG SUMMARY{label}")
        print(f"{'='*50}")
        print(f"  Total signals : {total}")
        print(f"  BUY           : {buys}")
        print(f"  SELL          : {sells}")
        print(f"  HOLD          : {holds}")
        print(f"  Avg edge      : {avg_edge*100:.2f}%")
        print(f"  Avg confidence: {avg_confidence*100:.1f}%")
        if outcomes_recorded > 0:
            print(f"\n  Outcomes      : {outcomes_recorded}")
            print(f"  Wins / Losses : {wins} / {losses}")
            print(f"  Win rate      : {win_rate:.1f}%")
            print(f"  Avg real edge : {avg_realized_edge*100:.2f}%")
        else:
            print(f"\n  (No outcomes recorded yet)")
        print(f"{'='*50}\n")

        return stats

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
