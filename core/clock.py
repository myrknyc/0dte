"""
Centralized timezone-aware clock utilities.

All time operations should go through this module to prevent
timezone bugs when running on non-ET servers.
"""

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def now_et() -> datetime:
    """Current time as timezone-aware Eastern datetime."""
    return datetime.now(ET)


def now_utc() -> datetime:
    """Current time as timezone-aware UTC datetime."""
    return datetime.now(UTC)


def now_utc_str() -> str:
    """Current time as UTC ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def to_utc_str(dt: datetime) -> str:
    """Convert any datetime to UTC ISO string. Assumes ET if naive."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(UTC).isoformat()


def is_market_open(dt: datetime = None) -> bool:
    """Check if the market is currently open (9:30 AM - 4:00 PM ET)."""
    if dt is None:
        dt = now_et()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)

    t = dt.time()
    return dt_time(9, 30) <= t < dt_time(16, 0)


def market_open_time() -> dt_time:
    """Return market open time (9:30 AM ET)."""
    return dt_time(9, 30)


def market_close_time() -> dt_time:
    """Return market close time (4:00 PM ET)."""
    return dt_time(16, 0)
