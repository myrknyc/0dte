"""
Intraday Volatility Seasonality — U-Shape Diurnal Pattern

Provides time-of-day variance scaling factors so the MC simulation
reflects the empirical pattern: high vol at open, low midday, rising
into close.
"""

import numpy as np
from config import DIURNAL_DEFAULT_PARAMS, TRADING_DAYS_PER_YEAR, MINUTES_PER_TRADING_DAY


def compute_diurnal_weights(session_fractions, params=None):
    """Compute variance scaling weights for given session-elapsed fractions.

    Parameters
    ----------
    session_fractions : array-like
        Values in [0, 1] where 0 = market open and 1 = market close.
    params : dict, optional
        Diurnal curve parameters {a, b, c1, d, c2}.
        Defaults to DIURNAL_DEFAULT_PARAMS.

    Returns
    -------
    np.ndarray — weights w(t) normalized so mean(w) = 1.0
    """
    if params is None:
        params = DIURNAL_DEFAULT_PARAMS

    t = np.asarray(session_fractions, dtype=float)

    a = params['a']
    b = params['b']
    c1 = params['c1']
    d = params['d']
    c2 = params['c2']

    # U-shape: high near t=0 (open), low midday, high near t=1 (close)
    w = a + b * np.exp(-c1 * t ** 2) + d * np.exp(-c2 * (1 - t) ** 2)

    # Normalize so mean(w) = 1.0  (preserves total variance budget)
    w_mean = np.mean(w)
    if w_mean > 0:
        w = w / w_mean

    return w


def simulation_time_to_session_fraction(sim_times, T, current_session_fraction):
    """Map simulation time grid [0, T] to session-elapsed fractions.

    Parameters
    ----------
    sim_times : array-like
        Simulation time points in trading-years, shape (n_steps+1,).
    T : float
        Total time horizon in trading-years.
    current_session_fraction : float
        Where we currently are in the session [0, 1].
        E.g. 0.5 = midday.

    Returns
    -------
    np.ndarray — session fractions for each sim time step, in [0, 1]
    """
    t = np.asarray(sim_times, dtype=float)

    # T represents time remaining to close, so session_fraction goes
    # from current_session_fraction to 1.0 as simulation progresses
    if T > 0 and len(t) > 1:
        frac = current_session_fraction + (1.0 - current_session_fraction) * (t / T)
    else:
        frac = np.full_like(t, current_session_fraction)

    return np.clip(frac, 0.0, 1.0)


def get_current_session_fraction():
    """Get the current position in the trading session [0, 1].

    0.0 = 9:30 AM ET,  1.0 = 4:00 PM ET.
    Before open returns 0.0, after close returns 1.0.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
    now = datetime.now(NY)

    market_open_min = 9 * 60 + 30   # 9:30
    market_close_min = 16 * 60      # 16:00
    now_min = now.hour * 60 + now.minute

    if now_min <= market_open_min:
        return 0.0
    if now_min >= market_close_min:
        return 1.0

    elapsed = now_min - market_open_min
    total = market_close_min - market_open_min  # 390

    return elapsed / total


def calibrate_diurnal_from_history(intraday_data, n_days_back=5):
    """Fit diurnal curve parameters from historical 1-min data.

    Parameters
    ----------
    intraday_data : pd.DataFrame
        1-minute OHLCV with DatetimeIndex.
    n_days_back : int
        How many days of data to use.

    Returns
    -------
    dict — fitted parameters {a, b, c1, d, c2}
    """
    import pandas as pd
    from scipy.optimize import minimize

    df = intraday_data.copy()

    # Log returns
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna(subset=['log_ret'])

    # Map each bar to session fraction
    if df.index.tz is None:
        from zoneinfo import ZoneInfo
        df.index = df.index.tz_localize(ZoneInfo('America/New_York'))

    minutes = df.index.hour * 60 + df.index.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60
    session_frac = (minutes - market_open) / (market_close - market_open)
    df['session_frac'] = np.clip(session_frac, 0, 1)

    # Keep only regular session
    df = df[(df['session_frac'] >= 0) & (df['session_frac'] <= 1)]

    # Group into 10-minute buckets for noise reduction
    df['bucket'] = (df['session_frac'] * 39).astype(int).clip(0, 38)
    vol_by_bucket = df.groupby('bucket')['log_ret'].std()

    if len(vol_by_bucket) < 10:
        # Not enough data — return defaults
        return DIURNAL_DEFAULT_PARAMS.copy()

    # Normalize to relative vol
    bucket_fracs = (vol_by_bucket.index + 0.5) / 39.0
    vol_vals = vol_by_bucket.values
    vol_norm = vol_vals / np.mean(vol_vals)

    # Fit parametric U-shape
    def loss(x):
        a, b, c1, d, c2 = x
        w = a + b * np.exp(-c1 * bucket_fracs ** 2) + d * np.exp(-c2 * (1 - bucket_fracs) ** 2)
        w = w / np.mean(w)
        return np.sum((w - vol_norm) ** 2)

    result = minimize(loss, [0.7, 0.5, 8.0, 0.4, 10.0],
                      bounds=[(0.1, 2), (0, 3), (1, 50), (0, 3), (1, 50)],
                      method='L-BFGS-B')

    a, b, c1, d, c2 = result.x
    return {'a': float(a), 'b': float(b), 'c1': float(c1),
            'd': float(d), 'c2': float(c2)}
