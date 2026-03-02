"""
Market Data Provider — unified interface for yfinance and IBKR.

Usage:
    provider = select_provider()        # Interactive menu
    provider = get_provider('yfinance') # Programmatic
    provider = get_provider('ibkr')

All providers expose the same methods so TradingSystem and
signal_generator work identically regardless of data source.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
#  Abstract base
# ---------------------------------------------------------------------------
class MarketDataProvider(ABC):
    """Common interface every data source must implement."""

    name: str = "base"

    @abstractmethod
    def get_spot_price(self, ticker: str) -> Dict[str, Any]:
        """Return latest spot price with freshness metadata.

        Returns
        -------
        dict  with keys:
            price : float         — last traded / quoted price
            timestamp : datetime  — when this price was observed
            age_seconds : float   — seconds since observation
        """

    @abstractmethod
    def get_intraday_data(self, ticker: str, period: str = '5d',
                          interval: str = '1m') -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex."""

    @abstractmethod
    def get_option_chain(self, ticker: str,
                         expiry_date: str = None) -> Dict[str, pd.DataFrame]:
        """Return {'calls': DataFrame, 'puts': DataFrame, 'expiry_date': str}."""

    # --- Derived helpers (work on any provider) ---

    def compute_returns(self, data: pd.DataFrame) -> np.ndarray:
        prices = data['Close'].values
        return np.diff(np.log(prices))

    def compute_vwap(self, data: pd.DataFrame,
                     date: str = None) -> float:
        df = data.copy()
        if date is not None:
            df = df[df.index.date == pd.to_datetime(date).date()]
        else:
            most_recent = df.index[-1].date()
            df = df[df.index.date == most_recent]
        return float((df['Close'] * df['Volume']).sum() / df['Volume'].sum())

    def get_historical_volatility(self, data: pd.DataFrame,
                                  periods_per_year: int = None) -> float:
        from config import TRADING_DAYS_PER_YEAR
        if periods_per_year is None:
            periods_per_year = 390 * TRADING_DAYS_PER_YEAR
        returns = self.compute_returns(data)
        return float(np.std(returns) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
#  yfinance provider
# ---------------------------------------------------------------------------
class YFinanceProvider(MarketDataProvider):
    """Delayed data via yfinance (free, no connection required)."""

    name = "yfinance"

    def __init__(self):
        import yfinance as yf
        self._yf = yf
        self._tickers = {}  # cache Ticker objects

    def _get_ticker(self, ticker: str):
        if ticker not in self._tickers:
            self._tickers[ticker] = self._yf.Ticker(ticker)
        return self._tickers[ticker]

    def get_spot_price(self, ticker: str) -> Dict[str, Any]:
        # Create a FRESH Ticker every time to avoid stale cache
        t = self._yf.Ticker(ticker)
        fetch_time = datetime.now()
        try:
            price = t.fast_info['lastPrice']
        except Exception:
            hist = t.history(period='1d', interval='1m')
            price = hist['Close'].iloc[-1]
        age = (datetime.now() - fetch_time).total_seconds()
        print(f"  {ticker} spot: ${price:.2f}  (yfinance, age={age:.1f}s)")
        return {'price': float(price), 'timestamp': fetch_time, 'age_seconds': age}

    def get_intraday_data(self, ticker: str, period: str = '5d',
                          interval: str = '1m') -> pd.DataFrame:
        t = self._get_ticker(ticker)
        data = t.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No intraday data for {ticker}")
        print(f"  Intraday: {len(data)} bars ({period} @ {interval})  (yfinance)")
        return data

    def get_option_chain(self, ticker: str,
                         expiry_date: str = None) -> Dict[str, Any]:
        t = self._get_ticker(ticker)
        if expiry_date is None:
            from datetime import datetime
            expiry_date = datetime.now().strftime('%Y-%m-%d')
        try:
            chain = t.option_chain(expiry_date)
        except Exception:
            # Fallback to nearest expiry
            chain = t.option_chain()
            expiry_date = t.options[0]
        return {
            'calls': chain.calls,
            'puts': chain.puts,
            'expiry_date': expiry_date,
        }


# ---------------------------------------------------------------------------
#  IBKR provider
# ---------------------------------------------------------------------------
class IBKRProvider(MarketDataProvider):
    """Real-time data via Interactive Brokers (requires TWS / IB Gateway)."""

    name = "ibkr"

    def __init__(self, host: str = '127.0.0.1', port: int = 4002,
                 client_id: int = 1):
        from data.streaming_data_feed import StreamingDataFeed
        self.feed = StreamingDataFeed(host=host, port=port,
                                       client_id=client_id)
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            print("  Connecting to IBKR...")
            if not self.feed.connect():
                raise ConnectionError(
                    "Could not connect to TWS/IB Gateway. "
                    "Is it running with API enabled?"
                )
            self._connected = True

    def get_spot_price(self, ticker: str) -> Dict[str, Any]:
        self._ensure_connected()
        self.feed.subscribe_stock(ticker)
        # Wait briefly for first tick
        import time as _t
        _t.sleep(1.5)
        fetch_time = datetime.now()
        price = self.feed.get_current_price(ticker)
        if price is None:
            raise ValueError(f"No price received for {ticker} from IBKR")
        age = (datetime.now() - fetch_time).total_seconds()
        print(f"  {ticker} spot: ${price:.2f}  (IBKR live, age={age:.1f}s)")
        return {'price': float(price), 'timestamp': fetch_time, 'age_seconds': age}

    def get_intraday_data(self, ticker: str, period: str = '5d',
                          interval: str = '1m') -> pd.DataFrame:
        self._ensure_connected()
        # Map yfinance-style args to IBKR
        dur_map = {
            '1d': '1 D', '2d': '2 D', '5d': '5 D',
            '10d': '10 D', '30d': '30 D', '60d': '60 D',
        }
        bar_map = {
            '1m': '1 min', '2m': '2 mins', '5m': '5 mins',
            '15m': '15 mins', '30m': '30 mins', '1h': '1 hour',
            '1d': '1 day',
        }
        duration = dur_map.get(period, '5 D')
        bar_size = bar_map.get(interval, '1 min')
        data = self.feed.get_historical_data(ticker, duration=duration,
                                              bar_size=bar_size)
        if data is None or data.empty:
            raise ValueError(f"No intraday data for {ticker} from IBKR")
        print(f"  Intraday: {len(data)} bars ({duration} @ {bar_size})  (IBKR)")
        return data

    def get_option_chain(self, ticker: str,
                         expiry_date: str = None) -> Dict[str, Any]:
        self._ensure_connected()
        chain_df = self.feed.get_option_chain(ticker, expiry_date=expiry_date)
        if chain_df is None or chain_df.empty:
            raise ValueError(f"No option chain for {ticker} from IBKR")

        # Reshape into calls/puts DataFrames matching yfinance format
        calls = chain_df[['strike', 'call_bid', 'call_ask', 'call_last']].copy()
        calls.columns = ['strike', 'bid', 'ask', 'lastPrice']

        puts = chain_df[['strike', 'put_bid', 'put_ask', 'put_last']].copy()
        puts.columns = ['strike', 'bid', 'ask', 'lastPrice']

        return {
            'calls': calls,
            'puts': puts,
            'expiry_date': expiry_date or 'nearest',
        }

    def disconnect(self):
        if self._connected:
            self.feed.disconnect()
            self._connected = False


# ---------------------------------------------------------------------------
#  Placeholder for future providers
# ---------------------------------------------------------------------------
class OtherProvider(MarketDataProvider):
    """Placeholder — extend for Alpaca, Polygon, Tradier, etc."""

    name = "other"

    def get_spot_price(self, ticker: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Other provider not configured. Subclass OtherProvider and "
            "implement get_spot_price / get_intraday_data / get_option_chain."
        )

    def get_intraday_data(self, ticker: str, **kw) -> pd.DataFrame:
        raise NotImplementedError("Other provider not configured.")

    def get_option_chain(self, ticker: str, **kw) -> dict:
        raise NotImplementedError("Other provider not configured.")


# ---------------------------------------------------------------------------
#  Factory + interactive menu
# ---------------------------------------------------------------------------
_PROVIDERS = {
    '1': ('yfinance', YFinanceProvider),
    '2': ('ibkr', IBKRProvider),
    '3': ('other', OtherProvider),
}


def get_provider(name: str = 'yfinance', **kwargs) -> MarketDataProvider:
    """Create a provider by name (programmatic)."""
    name = name.lower()
    for _, (pname, cls) in _PROVIDERS.items():
        if pname == name:
            return cls(**kwargs)
    raise ValueError(f"Unknown provider '{name}'. Options: yfinance, ibkr, other")


def select_provider(**kwargs) -> MarketDataProvider:
    """Interactive menu — call at startup to let user choose."""
    print("\n" + "=" * 50)
    print("SELECT DATA PROVIDER")
    print("=" * 50)
    print("  [1] yfinance   (free, 15-min delay)")
    print("  [2] IBKR       (real-time, requires TWS/Gateway)")
    print("  [3] Other      (custom / future)")
    print("=" * 50)

    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice in _PROVIDERS:
            name, cls = _PROVIDERS[choice]
            print(f"  → Using {name}\n")
            return cls(**kwargs)
        print("  Invalid choice. Try again.")
