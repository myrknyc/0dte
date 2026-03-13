import time
import numpy as np
from datetime import datetime, time as dt_time
from typing import Dict
import threading

from core.clock import now_et, now_utc, is_market_open, ET
from data.data_provider import MarketDataProvider, select_provider, IBKRProvider
from trading.trading_system import TradingSystem
from trading.paper_trader import PaperTrader
from trading.paper_journal import PaperJournal
from trading.eod_reporter import EODReporter
from signals.signal_logger import SignalLogger
from calibration.heston_calibrator import calibrate_heston_live
from calibration.jump_calibrator import calibrate_jumps_live
from config import get_time_to_expiry, SPOT_MAX_AGE_SECONDS


class ContinuousTradingMonitor:
    def __init__(self, ticker='SPY', recalibrate_minutes=30, edge_threshold=0.02,
                 provider: MarketDataProvider = None, enable_paper_trading=False):
        self.ticker = ticker
        self.recalibrate_minutes = recalibrate_minutes
        self.edge_threshold = edge_threshold
        
        # Data provider (chosen at startup)
        self.provider = provider
        
        # Initialize components
        self.trader = TradingSystem(ticker, provider=self.provider)
        self.logger = SignalLogger()
        
        # Paper trading (optional)
        self.paper_trader = None
        self._eod_reporter = None
        if enable_paper_trading:
            journal = PaperJournal()
            self.paper_trader = PaperTrader(journal=journal)
            self._eod_reporter = EODReporter(journal=journal)
        
        # For IBKR: keep reference to the streaming feed for live updates
        self._ibkr_feed = None
        if isinstance(self.provider, IBKRProvider):
            self._ibkr_feed = self.provider.feed
        
        # State
        self.last_calibration = None
        self.current_price = None
        self.latest_signals = {}
        
        self.running = False
        
    def connect(self):
        print("Initializing continuous trading monitor...")
        
        # IBKR: ensure connection is up
        if self._ibkr_feed and not self._ibkr_feed.is_connected:
            if not self._ibkr_feed.connect():
                return False
        
        # Initial calibration (uses provider automatically)
        print("\nPerforming initial calibration...")
        self.trader.calibrate_to_market()
        self.last_calibration = now_utc()
        
        return True
    
    def _on_price_update(self, symbol, data):
        self.current_price = data['last']
        self._price_timestamp = now_utc()
        
        # Update trader's current price + timestamp
        self.trader.S0 = self.current_price
        self.trader.spot_timestamp = self._price_timestamp
        
        # Check if recalibration needed
        if self.last_calibration:
            minutes_since = (now_utc() - self.last_calibration).total_seconds() / 60
            
            if minutes_since >= self.recalibrate_minutes:
                print(f"\n{'='*60}")
                print(f"Auto-recalibrating ({self.recalibrate_minutes} min elapsed)")
                print(f"{'='*60}")
                self.trader.calibrate_to_market()
                self.last_calibration = now_utc()
    
    def scan_strikes(self, strikes_to_check=5):
        """Scan option strikes for call and put signals."""
        if not self.current_price:
            return
        
        # Generate strikes around ATM
        atm = round(self.current_price)
        strikes = [atm + i for i in range(-strikes_to_check//2, strikes_to_check//2 + 1)]
        
        print(f"\n{now_et().strftime('%H:%M:%S')} ET - Scanning {len(strikes)} strikes (calls + puts)...")
        
        # Get option chain from provider
        try:
            chain = self.provider.get_option_chain(self.ticker)
            
            if chain is None:
                return
            
            calls = chain.get('calls')
            puts = chain.get('puts')
            
            signals_found = []
            all_signals = []    # every signal, for paper trader
            quote_map = {}      # (strike, option_type) → quote
            T = get_time_to_expiry()
            
            spot_age_val = (
                (now_utc() - self.trader.spot_timestamp).total_seconds()
                if getattr(self.trader, 'spot_timestamp', None) else 0
            )
            
            # Scan both calls and puts
            chain_pairs = []
            if calls is not None and not calls.empty:
                chain_pairs.append(('call', calls))
            if puts is not None and not puts.empty:
                chain_pairs.append(('put', puts))
            
            for opt_type, chain_df in chain_pairs:
                for strike in strikes:
                    chain_row = chain_df[chain_df['strike'] == strike]
                    
                    if len(chain_row) == 0:
                        continue
                    
                    row = chain_row.iloc[0]
                    
                    # Handle both yfinance and IBKR column names
                    bid_col = 'bid' if 'bid' in row.index else f'{opt_type}_bid'
                    ask_col = 'ask' if 'ask' in row.index else f'{opt_type}_ask'
                    if row[bid_col] > 0 and row[ask_col] > 0:
                        mkt_iv = row.get('impliedVolatility',
                                         row.get(f'{opt_type}_iv', None))
                        
                        signal = self.trader.get_trading_signal(
                            strike=strike,
                            market_bid=row[bid_col],
                            market_ask=row[ask_col],
                            option_type=opt_type,
                            market_iv=mkt_iv
                        )
                        
                        # Log every signal
                        spot_age = signal.get('spot_age_seconds', None)
                        self.logger.log_signal(
                            ticker=self.ticker,
                            strike=strike,
                            option_type=opt_type,
                            action=signal['action'],
                            edge=signal['edge'],
                            confidence=signal['confidence'],
                            model_price=signal['model_price'],
                            market_bid=row[bid_col],
                            market_ask=row[ask_col],
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
                            spot_timestamp=self.trader.spot_timestamp,
                            spot_age_seconds=spot_age,
                        )
                        
                        if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                            signals_found.append({
                                'strike': strike,
                                'type': opt_type.upper(),
                                **signal
                            })
                        all_signals.append(signal)
                        
                        # Build quote map with tuple keys
                        quote_map[(strike, opt_type)] = {
                            'bid': float(row[bid_col]),
                            'ask': float(row[ask_col]),
                            'spot': self.current_price,
                            'spot_age': spot_age_val,
                        }
            
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
            
            # ── Paper trading hook ──
            if self.paper_trader and all_signals:
                self.paper_trader.on_scan(
                    all_signals, quote_map, self.current_price, now_et(),
                    trading_system=self.trader
                )
            
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
        
        # Subscribe to price updates (IBKR: real-time; yfinance: poll)
        if self._ibkr_feed:
            self._ibkr_feed.subscribe_stock(self.ticker, self._on_price_update)
        else:
            # yfinance: update price at each scan cycle
            pass
        
        self.running = True
        
        # Scanning loop (runs in main thread)
        try:
            last_scan = now_utc()
            
            while self.running:
                # Check if market is open (9:30 AM - 4:00 PM ET)
                if not is_market_open():
                    print(f"Market closed. Waiting... (Current time: {now_et().strftime('%H:%M:%S')} ET)")
                    time.sleep(60)
                    continue
                
                # Check if scan interval elapsed
                seconds_since_scan = (now_utc() - last_scan).total_seconds()
                
                if seconds_since_scan >= scan_interval_seconds:
                    # For yfinance: refresh spot price each cycle
                    if not self._ibkr_feed:
                        try:
                            self.trader.update_spot()
                            self.current_price = self.trader.S0
                        except Exception:
                            pass
                    self.scan_strikes()
                    last_scan = now_utc()
                
                # Process events
                if self._ibkr_feed:
                    self._ibkr_feed.ib.sleep(1)
                else:
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            self.stop()
    
    def stop(self):
        """Stop monitoring and disconnect"""
        self.running = False
        self.logger.summary(now_et().strftime('%Y-%m-%d'))
        self.logger.close()
        if self._ibkr_feed:
            self._ibkr_feed.disconnect()
        # Paper trading EOD report
        if self.paper_trader and self._eod_reporter:
            self._eod_reporter.full_report(self.paper_trader.run_id)
        print("Monitor stopped")


# Run as continuous monitor
if __name__ == "__main__":
    print("="*60)
    print("0DTE CONTINUOUS TRADING MONITOR")
    print("="*60)
    print("\nREQUIREMENTS:")
    print("  • For IBKR: TWS or IB Gateway running with API enabled")
    print("  • For yfinance: internet connection (15-min delayed)")
    print("\n" + "="*60 + "\n")
    
    # Let user choose data source
    provider = select_provider()
    
    # Create monitor with chosen provider
    monitor = ContinuousTradingMonitor(
        ticker='SPY',
        recalibrate_minutes=30,
        edge_threshold=0.02,
        provider=provider,
        enable_paper_trading=True
    )
    
    # Connect and start
    if monitor.connect():
        print(f"\n✓ Connected and calibrated (using {provider.name})")
        print("Starting continuous monitoring...\n")
        monitor.start(scan_interval_seconds=60)
    else:
        print("\n❌ Failed to connect. Check your setup.")
