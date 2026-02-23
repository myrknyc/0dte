import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    YF_PERIOD, YF_INTERVAL, MIN_VOLUME, MIN_OPEN_INTEREST,
    MONEYNESS_RANGE, TRADING_DAYS_PER_YEAR
)

class DataLoader:
    
    def __init__(self, ticker="SPY"):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        self.spot_price = None
        self.intraday_data = None
        self.options_data = None
        
    def get_spot_price(self):
        try:
            self.spot_price = self.yf_ticker.fast_info['lastPrice']
        except Exception:
            try:
                hist = self.yf_ticker.history(period="1d", interval="1m")
                self.spot_price = hist['Close'].iloc[-1]
            except Exception:
                raise ValueError(f"Could not fetch spot price for {self.ticker}")
        
        print(f"Current {self.ticker} spot price: ${self.spot_price:.2f}")
        return self.spot_price
    
    def get_intraday_data(self, period=YF_PERIOD, interval=YF_INTERVAL):
        print(f"Fetching {period} of {interval} data for {self.ticker}...")
        
        self.intraday_data = self.yf_ticker.history(
            period=period, 
            interval=interval
        )
        
        if self.intraday_data.empty:
            raise ValueError(f"No intraday data returned for {self.ticker}")
        
        print(f"Retrieved {len(self.intraday_data)} data points")
        print(f"Date range: {self.intraday_data.index[0]} to {self.intraday_data.index[-1]}")
        
        return self.intraday_data
    
    def compute_returns(self, data=None):
        if data is None:
            data = self.intraday_data
        
        if data is None:
            raise ValueError("No data available. Call get_intraday_data() first.")
        
        prices = data['Close'].values
        returns = np.diff(np.log(prices))
        
        return returns
    
    def get_options_chain(self, expiry_date=None):
        print(f"Fetching options chain for {self.ticker}...")
        
        expiry_dates = self.yf_ticker.options
        
        if len(expiry_dates) == 0:
            raise ValueError(f"No options available for {self.ticker}")
        
        if expiry_date is None:
            expiry_date = expiry_dates[0]
        
        print(f"Using expiration date: {expiry_date}")
        
        opt_chain = self.yf_ticker.option_chain(expiry_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        print(f"Found {len(calls)} calls and {len(puts)} puts")
        
        self.options_data = {
            'calls': calls,
            'puts': puts,
            'expiry_date': expiry_date
        }
        
        return self.options_data
    
    def filter_options(self, option_type='call', min_volume=MIN_VOLUME,
                      min_oi=MIN_OPEN_INTEREST, moneyness_range=MONEYNESS_RANGE):
        if self.options_data is None:
            raise ValueError("No options data. Call get_options_chain() first.")
        
        if self.spot_price is None:
            self.get_spot_price()
        
        df = self.options_data['calls' if option_type == 'call' else 'puts'].copy()
        
        df = df[
            (df['volume'] >= min_volume) & 
            (df['openInterest'] >= min_oi)
        ]
        
        df['moneyness'] = df['strike'] / self.spot_price
        df = df[
            (df['moneyness'] >= moneyness_range[0]) & 
            (df['moneyness'] <= moneyness_range[1])
        ]
        
        print(f"After filtering: {len(df)} {option_type}s remain")
        
        return df
    
    def get_implied_volatilities(self, option_type='call'):
        df = self.filter_options(option_type)
        
        if 'impliedVolatility' in df.columns:
            df = df[df['impliedVolatility'] > 0]
            print(f"Mean IV for {option_type}s: {df['impliedVolatility'].mean():.4f}")
            return df
        else:
            print("Warning: No implied volatility data available")
            return df
    
    def get_historical_volatility(self, window=30):
        if self.intraday_data is None:
            self.get_intraday_data()
        
        returns = self.compute_returns()
        
        periods_per_year = 390 * TRADING_DAYS_PER_YEAR
        
        hist_vol = np.std(returns) * np.sqrt(periods_per_year)
        
        print(f"Historical volatility: {hist_vol:.4f} ({hist_vol*100:.2f}%)")
        
        return hist_vol
    
    def compute_vwap(self, date=None):
        if self.intraday_data is None:
            self.get_intraday_data()
        
        df = self.intraday_data.copy()
        
        if date is not None:
            df = df[df.index.date == pd.to_datetime(date).date()]
        else:
            most_recent_date = df.index[-1].date()
            df = df[df.index.date == most_recent_date]
        
        vwap = (df['Close'] * df['Volume']).sum() / df['Volume'].sum()
        
        print(f"VWAP: ${vwap:.2f}")
        
        return vwap
    
    def summary(self):
        print("\n" + "="*60)
        print(f"DATA LOADER SUMMARY - {self.ticker}")
        print("="*60)
        
        if self.spot_price:
            print(f"Spot Price: ${self.spot_price:.2f}")
        
        if self.intraday_data is not None:
            print(f"\nIntraday Data:")
            print(f"  Points: {len(self.intraday_data)}")
            print(f"  Range: {self.intraday_data.index[0]} to {self.intraday_data.index[-1]}")
            
            returns = self.compute_returns()
            print(f"  Returns: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
        
        if self.options_data is not None:
            print(f"\nOptions Data:")
            print(f"  Expiry: {self.options_data['expiry_date']}")
            print(f"  Calls: {len(self.options_data['calls'])}")
            print(f"  Puts: {len(self.options_data['puts'])}")
        
        print("="*60)



def quick_load_spy():
    loader = DataLoader("SPY")
    loader.get_spot_price()
    loader.get_intraday_data()
    loader.get_options_chain()
    return loader

def get_0dte_options():
    loader = DataLoader("SPY")
    loader.get_spot_price()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        loader.get_options_chain(today)
        print(f"Found 0DTE options expiring today ({today})")
        return loader
    except Exception:
        print(f"No 0DTE options expiring today. Using nearest expiry.")
        loader.get_options_chain()
        return loader
