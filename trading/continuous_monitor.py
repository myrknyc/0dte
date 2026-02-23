import time
import numpy as np
from datetime import datetime, time as dt_time
from typing import Dict
import threading

from data.streaming_data_feed import StreamingDataFeed
from trading.trading_system import TradingSystem
from signals.signal_logger import SignalLogger
from calibration.heston_calibrator import calibrate_heston_live
from calibration.jump_calibrator import calibrate_jumps_live
from config import get_time_to_expiry


class ContinuousTradingMonitor:
    def __init__(self, ticker='SPY', recalibrate_minutes=30, edge_threshold=0.02):
        self.ticker = ticker
        self.recalibrate_minutes = recalibrate_minutes
        self.edge_threshold = edge_threshold
        
        # Initialize components
        self.stream = StreamingDataFeed()
        self.trader = TradingSystem(ticker)
        self.logger = SignalLogger()
        
        # State
        self.last_calibration = None
        self.current_price = None
        self.latest_signals = {}
        
        self.running = False
        
    def connect(self):
        print("Initializing continuous trading monitor...")
        
        if not self.stream.connect():
            return False
        
        # Initial calibration
        print("\nPerforming initial calibration...")
        self.trader.calibrate_to_market()
        self.last_calibration = datetime.now()
        
        return True
    
    def _on_price_update(self, symbol, data):
        self.current_price = data['last']
        
        # Update trader's current price
        self.trader.S0 = self.current_price
        
        # Check if recalibration needed
        if self.last_calibration:
            minutes_since = (datetime.now() - self.last_calibration).seconds / 60
            
            if minutes_since >= self.recalibrate_minutes:
                print(f"\n{'='*60}")
                print(f"Auto-recalibrating ({self.recalibrate_minutes} min elapsed)")
                print(f"{'='*60}")
                self.trader.calibrate_to_market()
                self.last_calibration = datetime.now()
    
    def scan_strikes(self, strikes_to_check=5):
        """Scan option strikes for signals"""
        if not self.current_price:
            return
        
        # Generate strikes around ATM
        atm = round(self.current_price)
        strikes = [atm + i for i in range(-strikes_to_check//2, strikes_to_check//2 + 1)]
        
        print(f"\n{datetime.now().strftime('%H:%M:%S')} - Scanning {len(strikes)} strikes...")
        
        # Get option chain from IB
        try:
            option_chain = self.stream.get_option_chain(self.ticker)
            
            if option_chain is None:
                return
            
            signals_found = []
            T = get_time_to_expiry()
            
            for strike in strikes:
                chain_row = option_chain[option_chain['strike'] == strike]
                
                if len(chain_row) == 0:
                    continue
                
                row = chain_row.iloc[0]
                
                # Check calls
                if row['call_bid'] > 0 and row['call_ask'] > 0:
                    mkt_iv = row.get('call_iv', None)
                    
                    signal = self.trader.get_trading_signal(
                        strike=strike,
                        market_bid=row['call_bid'],
                        market_ask=row['call_ask'],
                        option_type='call',
                        market_iv=mkt_iv
                    )
                    
                    # Log every signal
                    self.logger.log_signal(
                        ticker=self.ticker,
                        strike=strike,
                        option_type='call',
                        action=signal['action'],
                        edge=signal['edge'],
                        confidence=signal['confidence'],
                        model_price=signal['model_price'],
                        market_bid=row['call_bid'],
                        market_ask=row['call_ask'],
                        market_mid=signal['market_mid'],
                        spread=signal['spread'],
                        std_error=signal['std_error'],
                        spot_price=self.current_price,
                        time_to_expiry=T,
                        iv=np.sqrt(self.trader.v0),
                        n_paths=signal.get('n_paths'),
                        vr_factor=signal.get('variance_reduction_factor'),
                        reason=signal['reason'],
                        source='continuous_monitor',
                        market_iv=mkt_iv,
                    )
                    
                    if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                        signals_found.append({
                            'strike': strike,
                            'type': 'CALL',
                            **signal
                        })
            
            # Display strong signals
            if signals_found:
                print(f"\n🔔 {len(signals_found)} SIGNAL(S) DETECTED:")
                for sig in signals_found:
                    print(f"  {sig['action']} {sig['strike']} {sig['type']} | "
                          f"Edge: {sig['edge']*100:.1f}% | "
                          f"Model: ${sig['model_price']:.2f} | "
                          f"Market: ${sig['market_mid']:.2f}")
                print()
            
            self.latest_signals = signals_found
            
        except Exception as e:
            print(f"Error scanning strikes: {e}")
    
    def start(self, scan_interval_seconds=60):
        """
        Start continuous monitoring
        
        Args:
            scan_interval_seconds: How often to check for signals
        """
        print(f"\n{'='*60}")
        print("CONTINUOUS TRADING MONITOR")
        print(f"{'='*60}")
        print(f"Ticker: {self.ticker}")
        print(f"Recalibrate every: {self.recalibrate_minutes} minutes")
        print(f"Scan interval: {scan_interval_seconds} seconds")
        print(f"Edge threshold: {self.edge_threshold*100:.1f}%")
        print(f"{'='*60}\n")
        
        # Subscribe to price updates
        self.stream.subscribe_stock(self.ticker, self._on_price_update)
        
        self.running = True
        
        # Scanning loop (runs in main thread)
        try:
            last_scan = datetime.now()
            
            while self.running:
                # Check if market is open (9:30 AM - 4:00 PM ET)
                now = datetime.now().time()
                market_open = dt_time(9, 30)
                market_close = dt_time(16, 0)
                
                if now < market_open or now > market_close:
                    print(f"Market closed. Waiting... (Current time: {datetime.now().strftime('%H:%M:%S')})")
                    time.sleep(60)
                    continue
                
                # Check if scan interval elapsed
                seconds_since_scan = (datetime.now() - last_scan).seconds
                
                if seconds_since_scan >= scan_interval_seconds:
                    self.scan_strikes()
                    last_scan = datetime.now()
                
                # Process IB events (non-blocking)
                self.stream.ib.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            self.stop()
    
    def stop(self):
        """Stop monitoring and disconnect"""
        self.running = False
        self.logger.summary(datetime.now().strftime('%Y-%m-%d'))
        self.logger.close()
        self.stream.disconnect()
        print("Monitor stopped")


# Run as continuous monitor
if __name__ == "__main__":
    print("="*60)
    print("0DTE CONTINUOUS TRADING MONITOR")
    print("="*60)
    print("\nREQUIREMENTS:")
    print("  1. Interactive Brokers account")
    print("  2. TWS or IB Gateway running")
    print("  3. API enabled in TWS settings")
    print("\n" + "="*60 + "\n")
    
    # Create monitor
    monitor = ContinuousTradingMonitor(
        ticker='SPY',
        recalibrate_minutes=30,  # Recalibrate every 30 min
        edge_threshold=0.02      # 2% min edge
    )
    
    # Connect and start
    if monitor.connect():
        print("\n✓ Connected and calibrated")
        print("Starting continuous monitoring...\n")
        monitor.start(scan_interval_seconds=60)  # Scan every 60 seconds
    else:
        print("\n❌ Failed to connect. Check IB setup.")
