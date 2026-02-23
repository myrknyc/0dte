"""
Signal Generator - SPY 0DTE
Shows trading signals - YOU decide whether to trade
"""

import yfinance as yf
import numpy as np
from trading.trading_system import TradingSystem
from signals.signal_logger import SignalLogger
from datetime import datetime

def display_signal(strike, signal, market_data):
    """Display a trading signal in clear format"""
    
    print(f"\n{'='*60}")
    print(f"🎯 SIGNAL: {strike} CALL")
    print(f"{'='*60}")
    print(f"Market Bid: ${market_data['bid']:.2f}")
    print(f"Market Ask: ${market_data['ask']:.2f}")
    print(f"Market Mid: ${signal['market_mid']:.2f}")
    print(f"\nModel Fair Value: ${signal['model_price']:.4f}")
    print(f"Model Std Error: ±${signal['std_error']:.6f}")
    
    # Diagnostic info
    if 'n_paths' in signal:
        print(f"  n_paths: {signal['n_paths']:,}")
    if 'variance_reduction_factor' in signal:
        print(f"  VR factor: {signal['variance_reduction_factor']:.2f}")
    if 'beta' in signal and signal['beta'] is not None:
        print(f"  CV beta: {signal['beta']:.4f}")
    
    if signal['action'] == 'BUY':
        print(f"\n💡 RECOMMENDATION: BUY")
        print(f"   Edge: {signal['edge']*100:+.1f}%")
        print(f"   Confidence: {signal['confidence']*100:.0f}%")
        print(f"   Reason: {signal['reason']}")
        print(f"\n   → Consider buying at ${market_data['ask']:.2f} or better")
        
    elif signal['action'] == 'SELL':
        print(f"\n💡 RECOMMENDATION: SELL")
        print(f"   Edge: {signal['edge']*100:+.1f}%")
        print(f"   Confidence: {signal['confidence']*100:.0f}%")
        print(f"   Reason: {signal['reason']}")
        print(f"\n   → Consider selling at ${market_data['bid']:.2f} or better")
        
    else:
        print(f"\n⏸️ HOLD - No edge detected")
        print(f"   Market is fairly priced")


def main():
    """Main signal generator"""
    
    print("="*60)
    print("SPY 0DTE SIGNAL GENERATOR")
    print("="*60)
    print("Shows trading signals - YOU execute manually")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Initialize trading system & logger
    trader = TradingSystem('SPY')
    trader.calibrate_to_market()
    logger = SignalLogger()
    
    print(f"✓ Model calibrated")
    print(f"  SPY: ${trader.S0:.2f}")
    print(f"  VWAP: ${trader.vwap:.2f}")
    print(f"  v0 (inst. vol): {np.sqrt(trader.v0):.2%}")
    
    # Time-to-expiry diagnostic
    from config import get_time_to_expiry, TRADING_DAYS_PER_YEAR
    T = get_time_to_expiry()
    T_hours = T * TRADING_DAYS_PER_YEAR * 6.5
    print(f"  T (to 4PM): {T:.6f} years ({T_hours:.2f} hours)")
    
    # BS sanity check for ATM
    from pricing.black_scholes import black_scholes
    from config import RISK_FREE_RATE
    atm_strike = round(trader.S0)
    bs_atm = black_scholes(trader.S0, atm_strike, T, RISK_FREE_RATE, np.sqrt(trader.v0), 'call')
    print(f"  BS ATM sanity ({atm_strike}C): ${bs_atm:.4f}")
    
    # Get market data — explicitly request today's 0DTE expiry
    spy = yf.Ticker('SPY')
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        options = spy.option_chain(today)
        calls = options.calls
        print(f"✓ Using 0DTE options expiring today ({today})")
    except Exception as e:
        print(f"\n⚠ No 0DTE options for {today}: {e}")
        print("Falling back to nearest expiry...")
        try:
            options = spy.option_chain()
            calls = options.calls
            print(f"  Using expiry: {spy.options[0]}")
        except Exception as e2:
            print(f"\n❌ Error fetching option data: {e2}")
            print("Note: yfinance has 15-min delay")
            return
    
    # Check strikes near ATM
    current = trader.S0
    strikes_to_check = [
        round(current - 2),
        round(current - 1),
        round(current),
        round(current + 1),
        round(current + 2)
    ]
    
    print(f"\n{'='*60}")
    print(f"Scanning {len(strikes_to_check)} strikes near ATM...")
    print(f"{'='*60}")
    
    signals_found = []
    logged_count = 0
    
    for strike in strikes_to_check:
        call_data = calls[calls['strike'] == strike]
        
        if len(call_data) > 0:
            bid = call_data['bid'].values[0]
            ask = call_data['ask'].values[0]
            mkt_iv = call_data['impliedVolatility'].values[0] if 'impliedVolatility' in call_data.columns else None
            
            if bid > 0 and ask > 0:  # Valid market
                signal = trader.get_trading_signal(strike, bid, ask, 'call',
                                                   market_iv=mkt_iv)
                
                # Log every signal (including HOLD) for backtesting
                logger.log_signal(
                    ticker='SPY',
                    strike=strike,
                    option_type='call',
                    action=signal['action'],
                    edge=signal['edge'],
                    confidence=signal['confidence'],
                    model_price=signal['model_price'],
                    market_bid=bid,
                    market_ask=ask,
                    market_mid=signal['market_mid'],
                    spread=signal['spread'],
                    std_error=signal['std_error'],
                    spot_price=trader.S0,
                    time_to_expiry=T,
                    iv=np.sqrt(trader.v0),
                    n_paths=signal.get('n_paths'),
                    vr_factor=signal.get('variance_reduction_factor'),
                    reason=signal['reason'],
                    source='signal_generator',
                    market_iv=mkt_iv,
                )
                logged_count += 1
                
                if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                    signals_found.append({
                        'strike': strike,
                        'signal': signal,
                        'market': {'bid': bid, 'ask': ask}
                    })
    
    # Display results
    if signals_found:
        print(f"\n✅ Found {len(signals_found)} trading signal(s):\n")
        
        for item in signals_found:
            display_signal(item['strike'], item['signal'], item['market'])
    else:
        print(f"\n⏸️ No strong signals detected at this time")
        print(f"   Market appears fairly priced")
        print(f"   Check again in 15-30 minutes")
    
    print(f"\n✓ {logged_count} signal(s) logged to {logger.db_path}")
    logger.summary(datetime.now().strftime('%Y-%m-%d'))
    
    print(f"{'='*60}")
    print(f"🔄 Run this script again anytime for fresh signals")
    print(f"💡 You decide whether to trade based on these signals")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
