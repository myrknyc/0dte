from ib_insync import IB, Stock, Option, util
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Callable
import threading
import queue
import time as time_module


class StreamingDataFeed:
    def __init__(self, host='127.0.0.1', port=4002, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self.subscriptions = {}  # ticker -> contract
        self.callbacks = {}      # ticker -> callback function
        self.price_data = {}     # ticker -> latest price info
        self._data_lock = threading.Lock()  # Thread safety for price_data
        
        self.is_connected = False
        self.auto_reconnect = True
        
    def connect(self, timeout=10):
        try:
            print(f"Connecting to IB at {self.host}:{self.port}...")
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            self.is_connected = True
            print("✓ Connected to Interactive Brokers")
            
            # Set up disconnection handler with proper binding
            self.ib.disconnectedEvent += lambda: self._on_disconnected()
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Is TWS/IB Gateway running?")
            print("  2. Is API enabled? (File -> Global Configuration -> API -> Settings)")
            print("  3. Check port number (7497=TWS paper, 7496=TWS live, 4002=Gateway)")
            return False
    
    def disconnect(self):
        if self.is_connected:
            self.ib.disconnect()
            self.is_connected = False
            print("Disconnected from IB")
    
    def _on_disconnected(self):
        print("⚠️ Disconnected from IB")
        self.is_connected = False
        
        if self.auto_reconnect:
            print("Attempting to reconnect...")
            self.connect()
    
    def subscribe_stock(self, symbol: str, callback: Optional[Callable] = None):
        if not self.is_connected:
            raise ConnectionError("Not connected to IB. Call connect() first.")
        
        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Request market data
        ticker = self.ib.reqMktData(contract, '', False, False)
        
        # Set up update handler
        def on_update(ticker):
            # Validate data quality before storing
            if not self._validate_price_data(symbol, ticker):
                return  # Skip invalid updates
            
            with self._data_lock:
                self.price_data[symbol] = {
                    'symbol': symbol,
                    'last': ticker.last,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'volume': ticker.volume,
                    'timestamp': datetime.now()
                }
            
            if callback:
                callback(symbol, self.price_data[symbol])
        
        ticker.updateEvent += on_update
        
        self.subscriptions[symbol] = contract
        self.callbacks[symbol] = callback
        
        print(f"✓ Subscribed to {symbol} real-time data")
        return ticker
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with NaN protection and fallback to midpoint."""
        with self._data_lock:
            if symbol not in self.price_data:
                return None
            
            data = self.price_data[symbol]
            last = data['last']
            
            # Handle NaN or None - fallback to midpoint
            if last is None or (isinstance(last, float) and np.isnan(last)):
                bid, ask = data['bid'], data['ask']
                if bid and ask and not np.isnan(bid) and not np.isnan(ask):
                    return (bid + ask) / 2.0
                return None
            
            return float(last)
    
    def _validate_price_data(self, symbol: str, ticker) -> bool:
        """Validate market data quality."""
        # Check for valid prices
        if ticker.bid is None or ticker.ask is None:
            return False
        
        if np.isnan(ticker.bid) or np.isnan(ticker.ask):
            return False
        
        # Check for crossed market (staleness indicator)
        if ticker.bid > ticker.ask + 0.01:  # Allow small epsilon
            print(f"⚠️ Crossed market detected for {symbol}: bid={ticker.bid:.2f} > ask={ticker.ask:.2f}")
            return False
        
        # Check for zero bid (likely stale)
        if ticker.bid <= 0:
            return False
        
        return True
    
    def _wait_for_valid_data(self, ticker, timeout=2.0):
        """Wait for ticker to receive valid market data."""
        start = time_module.time()
        while time_module.time() - start < timeout:
            if ticker.bid and ticker.ask:
                if not np.isnan(ticker.bid) and not np.isnan(ticker.ask):
                    return True
            self.ib.sleep(0.05)  # Poll every 50ms
        return False
    
    def is_data_stale(self, symbol: str, max_age_seconds=5.0) -> bool:
        """Check if price data is stale."""
        with self._data_lock:
            if symbol not in self.price_data:
                return True
            age = (datetime.now() - self.price_data[symbol]['timestamp']).total_seconds()
            return age > max_age_seconds
    
    def get_option_chain(self, symbol: str, expiry_date: str = None) -> pd.DataFrame:
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")
        
        # Get underlying contract
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        
        # Get option chain
        chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        
        if not chains:
            print(f"No option chains found for {symbol}")
            return None
        
        chain = chains[0]
        
        # Use specified expiry or nearest
        if expiry_date is None:
            expiry_date = min(chain.expirations)
        
        # Get all strikes
        strikes = sorted(chain.strikes)
        
        # Request option quotes
        options_data = []
        tickers_to_cancel = []  # Track for cleanup
        
        for strike in strikes:
            # Call
            call = Option(symbol, expiry_date, strike, 'C', 'SMART')
            self.ib.qualifyContracts(call)
            call_ticker = self.ib.reqMktData(call)
            tickers_to_cancel.append(call_ticker)
            
            # Put
            put = Option(symbol, expiry_date, strike, 'P', 'SMART')
            self.ib.qualifyContracts(put)
            put_ticker = self.ib.reqMktData(put)
            tickers_to_cancel.append(put_ticker)
            
            # Wait for valid data with timeout
            call_valid = self._wait_for_valid_data(call_ticker)
            put_valid = self._wait_for_valid_data(put_ticker)
            
            if not call_valid:
                print(f"⚠️ Warning: Incomplete data for {symbol} call strike {strike}")
            if not put_valid:
                print(f"⚠️ Warning: Incomplete data for {symbol} put strike {strike}")
            
            options_data.append({
                'strike': strike,
                'call_bid': call_ticker.bid,
                'call_ask': call_ticker.ask,
                'call_last': call_ticker.last,
                'put_bid': put_ticker.bid,
                'put_ask': put_ticker.ask,
                'put_last': put_ticker.last,
            })
        
        # CRITICAL: Cancel all subscriptions to prevent memory leak
        for ticker in tickers_to_cancel:
            self.ib.cancelMktData(ticker.contract)
        
        return pd.DataFrame(options_data)
    
    def get_historical_data(self, symbol: str, duration: str = '60 D', 
                           bar_size: str = '1 day') -> pd.DataFrame:
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")
        
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        
        bars = self.ib.reqHistoricalData(
            stock,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Convert to DataFrame
        df = util.df(bars)
        return df
    
    def subscribe_option(self, symbol: str, expiry: str, strike: float, 
                        right: str, callback: Optional[Callable] = None):
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")
        
        option = Option(symbol, expiry, strike, right, 'SMART')
        self.ib.qualifyContracts(option)
        
        ticker = self.ib.reqMktData(option)
        
        option_key = f"{symbol}_{expiry}_{strike}{right}"
        
        def on_update(ticker):
            with self._data_lock:
                self.price_data[option_key] = {
                    'symbol': symbol,
                    'strike': strike,
                    'right': right,
                    'expiry': expiry,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'iv': ticker.modelGreeks.impliedVol if ticker.modelGreeks else None,
                    'delta': ticker.modelGreeks.delta if ticker.modelGreeks else None,
                    'timestamp': datetime.now()
                }
            
            if callback:
                callback(option_key, self.price_data[option_key])
        
        ticker.updateEvent += on_update
        
        print(f"✓ Subscribed to {option_key} real-time data")
        return ticker
    
    def run(self):
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")
        
        print("\n" + "="*60)
        print("Streaming data feed active")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            self.ib.run()
        except KeyboardInterrupt:
            print("\nStopping stream...")
            self.disconnect()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Callback function
    def on_price_update(symbol, data):
        print(f"{data['timestamp'].strftime('%H:%M:%S')} | {symbol}: ${data['last']:.2f} "
              f"(Bid: ${data['bid']:.2f}, Ask: ${data['ask']:.2f})")
    
    # Initialize feed
    feed = StreamingDataFeed(
        host='127.0.0.1',
        port=4002,  # IB Gateway paper trading port
        client_id=1
    )
    
    # Connect
    if not feed.connect():
        sys.exit(1)
    
    # Subscribe to SPY
    feed.subscribe_stock('SPY', on_price_update)
    
    # Get historical data for calibration
    print("\nFetching 60 days of historical data...")
    hist_data = feed.get_historical_data('SPY', duration='60 D', bar_size='1 day')
    print(f"Retrieved {len(hist_data)} days")
    print(hist_data.tail())
    
    # Start streaming
    feed.run()
