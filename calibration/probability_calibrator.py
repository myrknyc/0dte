"""
Probability Calibrator — H2 scaffold.

Default mode: passthrough (calibrate(conf) → conf).
When activated with sufficient trade history, fits isotonic regression
to map raw confidence scores to calibrated win probabilities.

Activation requires:
  - USE_PROBABILITY_CALIBRATION = True in config.py
  - ≥ MIN_CALIBRATION_TRADES resolved trades in paper_trades.db
"""

import os
import sqlite3
from typing import Optional


# Minimum resolved trades before calibration activates
MIN_CALIBRATION_TRADES = 200


class ProbabilityCalibrator:
    """Map raw confidence scores to calibrated win probabilities.

    In passthrough mode (default), calibrate() returns the input unchanged.
    When fitted, uses isotonic regression for the mapping.
    """

    def __init__(self, db_path: str = None, auto_fit: bool = False):
        """
        Args:
            db_path: path to paper_trades.db (for fitting)
            auto_fit: if True and data is sufficient, fit on init
        """
        self._model = None
        self._n_trades = 0
        self._fitted = False

        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'paper_trades.db'
            )
        self._db_path = os.path.abspath(db_path)

        if auto_fit:
            self.try_fit()

    def calibrate(self, confidence: float) -> float:
        """Convert raw confidence to calibrated probability.

        In passthrough mode, returns confidence unchanged.
        When fitted, returns isotonic-calibrated p.

        Args:
            confidence: raw confidence score ∈ [0, 1]

        Returns:
            Calibrated probability ∈ [0, 1]
        """
        if not self._fitted or self._model is None:
            return max(0.0, min(1.0, confidence))

        try:
            import numpy as np
            p = self._model.predict(np.array([[confidence]]))[0]
            return float(max(0.0, min(1.0, p)))
        except Exception:
            return max(0.0, min(1.0, confidence))

    def is_ready(self) -> bool:
        """Check if sufficient trade data exists for fitting."""
        if not os.path.exists(self._db_path):
            return False
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT COUNT(*) FROM paper_trades "
                "WHERE status = 'CLOSED' AND track = 'decision_time'"
            ).fetchone()
            conn.close()
            self._n_trades = row[0] if row else 0
            return self._n_trades >= MIN_CALIBRATION_TRADES
        except Exception:
            return False

    def try_fit(self) -> bool:
        """Attempt to fit the calibration model from trade history.

        Returns True if fitting succeeded, False otherwise.
        """
        if not self.is_ready():
            return False

        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression

            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT confidence_entry, net_pnl_touch "
                "FROM paper_trades "
                "WHERE status = 'CLOSED' AND track = 'decision_time' "
                "  AND confidence_entry IS NOT NULL "
                "  AND net_pnl_touch IS NOT NULL"
            ).fetchall()
            conn.close()

            if len(rows) < MIN_CALIBRATION_TRADES:
                return False

            X = np.array([r[0] for r in rows])
            y = np.array([1 if r[1] > 0 else 0 for r in rows])

            model = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds='clip')
            model.fit(X, y)

            self._model = model
            self._fitted = True
            self._n_trades = len(rows)
            return True

        except ImportError:
            # sklearn not available — stay in passthrough
            return False
        except Exception:
            return False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_trades(self) -> int:
        return self._n_trades
