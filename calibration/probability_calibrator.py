"""
Probability Calibrator — H2 scaffold.

Default mode: passthrough (calibrate(conf) → conf).
When activated with sufficient trade history, fits either
Platt scaling (logistic) or isotonic regression to map raw
confidence scores to calibrated win probabilities.

Activation requires:
  - USE_PROBABILITY_CALIBRATION = True in config.py
  - ≥ MIN_CALIBRATION_TRADES resolved trades in paper_trades.db
"""

import math
import os
import sqlite3
from typing import Optional, Tuple


# Minimum resolved trades before calibration activates
MIN_CALIBRATION_TRADES = 200


class ProbabilityCalibrator:
    """Map raw confidence scores to calibrated win probabilities.

    In passthrough mode (default), calibrate() returns the input unchanged.
    When fitted, uses either Platt scaling or isotonic regression.
    """

    def __init__(self, db_path: str = None, auto_fit: bool = False,
                 method: str = 'platt'):
        """
        Args:
            db_path: path to paper_trades.db (for fitting)
            auto_fit: if True and data is sufficient, fit on init
            method: 'platt' (logistic, no deps) or 'isotonic' (needs sklearn)
        """
        self._model = None
        self._n_trades = 0
        self._fitted = False
        self._method = method

        # Platt scaling parameters: p = 1 / (1 + exp(a*x + b))
        self._platt_a = 0.0
        self._platt_b = 0.0

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
        When fitted, returns calibrated p.

        Args:
            confidence: raw confidence score ∈ [0, 1]

        Returns:
            Calibrated probability ∈ [0, 1]
        """
        if not self._fitted:
            return max(0.0, min(1.0, confidence))

        if self._method == 'platt':
            return self._platt_predict(confidence)

        # Isotonic fallback
        if self._model is not None:
            try:
                import numpy as np
                p = self._model.predict(np.array([[confidence]]))[0]
                return float(max(0.0, min(1.0, p)))
            except Exception:
                pass

        return max(0.0, min(1.0, confidence))

    def _platt_predict(self, x: float) -> float:
        """Platt scaling prediction: σ(a*x + b)."""
        z = self._platt_a * x + self._platt_b
        z = max(-20.0, min(20.0, z))  # clamp for numerical stability
        return 1.0 / (1.0 + math.exp(-z))

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

        if self._method == 'platt':
            return self._fit_platt()
        else:
            return self._fit_isotonic()

    def _load_training_data(self) -> Tuple[list, list]:
        """Load (confidence, win/loss) pairs from DB."""
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
            return [], []

        X = [r[0] for r in rows]
        y = [1 if r[1] > 0 else 0 for r in rows]
        return X, y

    def _fit_platt(self) -> bool:
        """Fit Platt scaling (logistic regression) via gradient descent.

        No sklearn dependency required — pure Python implementation.
        """
        try:
            X, y = self._load_training_data()
            if not X:
                return False

            # Newton's method for logistic regression: min -log-likelihood
            # L = Σ [y_i * log(σ(z_i)) + (1-y_i) * log(1 - σ(z_i))]
            a, b = -1.0, 0.0  # initial: identity-ish mapping
            lr = 0.01
            n = len(X)

            for _ in range(500):
                grad_a, grad_b = 0.0, 0.0
                for xi, yi in zip(X, y):
                    z = a * xi + b
                    z = max(-20.0, min(20.0, z))
                    p = 1.0 / (1.0 + math.exp(-z))
                    err = p - yi
                    grad_a += err * xi
                    grad_b += err
                a -= lr * grad_a / n
                b -= lr * grad_b / n

            self._platt_a = a
            self._platt_b = b
            self._fitted = True
            self._n_trades = n
            return True

        except Exception:
            return False

    def _fit_isotonic(self) -> bool:
        """Fit isotonic regression (requires sklearn)."""
        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression

            X, y = self._load_training_data()
            if not X:
                return False

            X_arr = np.array(X)
            y_arr = np.array(y)

            model = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds='clip')
            model.fit(X_arr, y_arr)

            self._model = model
            self._fitted = True
            self._n_trades = len(X)
            return True

        except ImportError:
            # sklearn not available — stay in passthrough
            return False
        except Exception:
            return False

    def fit_from_journal(self, journal) -> bool:
        """Fit calibrator from a PaperJournal instance (for live use)."""
        self._db_path = journal.db_path
        return self.try_fit()

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_trades(self) -> int:
        return self._n_trades

    @property
    def method(self) -> str:
        return self._method

