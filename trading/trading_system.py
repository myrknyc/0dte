import numpy as np
from datetime import datetime

from config import (
    DEFAULT_PARAMS, RISK_FREE_RATE, N_PATHS_DEFAULT, N_STEPS_0DTE,
    TRADING_DAYS_PER_YEAR, get_time_to_expiry, USE_ANTITHETIC
)
from data.data_loader import DataLoader
from pricing.black_scholes import black_scholes

from core.time_grid import generate_time_grid
from core.random_numbers import generate_correlated_normals
from models.combined_model import simulate_combined_paths_fast as simulate_combined_paths
from pricing.european import price_european_option
from calibration.heston_calibrator import calibrate_to_realized_vol
from calibration.jump_calibrator import calibrate_from_returns
from calibration.mean_reversion_calibrator import calibrate_from_intraday


class TradingSystem:
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.loader = DataLoader(ticker)
        
        # Market state
        self.S0 = None
        self.vwap = None
        self.historical_vol = None
        
        # Model parameters (start with defaults)
        self.params = DEFAULT_PARAMS.copy()
        self.v0 = self.params.get('v0', 0.04)
        
        # Simulation settings
        self.n_paths = N_PATHS_DEFAULT
        self.n_steps = N_STEPS_0DTE
        
        # Cache for efficiency
        self._cached_paths = None
        self._cached_T = None
        
    def calibrate_to_market(self, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f"CALIBRATING MODEL TO MARKET - {self.ticker}")
            print(f"{'='*60}")
        
        # 1. Get current spot price
        self.S0 = self.loader.get_spot_price()
        
        # 2. Get intraday data for calibration
        self.loader.get_intraday_data()
        
        # 3. Compute VWAP
        self.vwap = self.loader.compute_vwap()
        
        # 4. Get historical volatility
        self.historical_vol = self.loader.get_historical_volatility()
        
        # 5. Calibrate Heston parameters
        prices = self.loader.intraday_data['Close'].values
        # Intraday 1-min data → annualize with 252 * 390 periods per year
        heston_params = calibrate_to_realized_vol(prices, periods_per_year=252 * 390)
        
        if verbose:
            print(f"\nHeston calibration:")
            print(f"  kappa (mean reversion): {heston_params['kappa']:.4f}")
            print(f"  theta_v (long-term var): {heston_params['theta_v']:.4f}")
            print(f"  sigma_v (vol of vol): {heston_params['sigma_v']:.4f}")
            print(f"  rho (correlation): {heston_params['rho']:.4f}")
            print(f"  v0 (initial var): {heston_params['v0']:.4f}")
        
        # Update params
        self.params.update(heston_params)
        self.v0 = heston_params['v0']
        
        # 5b. IV anchor: floor v0 at ATM implied vol from option chain
        try:
            from datetime import datetime as _dt
            today_str = _dt.now().strftime('%Y-%m-%d')
            import yfinance as yf
            spy = yf.Ticker(self.ticker)
            chain = spy.option_chain(today_str)
            calls = chain.calls
            atm_strike = round(self.S0)
            atm_row = calls[calls['strike'] == atm_strike]
            if len(atm_row) > 0 and 'impliedVolatility' in atm_row.columns:
                iv_atm = atm_row['impliedVolatility'].values[0]
                # Validate IV is finite and plausible for SPY 0DTE
                if np.isfinite(iv_atm) and 0.05 <= iv_atm <= 2.0:
                    iv_var = iv_atm ** 2
                    if iv_var > self.v0:
                        if verbose:
                            print(f"\n  IV anchor: ATM IV={iv_atm:.2%}, v0 calibrated={np.sqrt(self.v0):.2%}")
                            print(f"  → Overriding v0: {np.sqrt(self.v0):.2%} → {iv_atm:.2%}")
                        self.v0 = iv_var
                        self.params['v0'] = iv_var
                    elif verbose:
                        print(f"\n  IV anchor: ATM IV={iv_atm:.2%}, v0={np.sqrt(self.v0):.2%} (OK, no override)")
                elif verbose:
                    print(f"\n  IV anchor: ATM IV={iv_atm} out of range [5%-200%]. Ignoring.")
        except Exception as e:
            if verbose:
                print(f"\n  IV anchor: Could not fetch ATM IV ({e}). Using calibrated v0.")
        
        # 6. Calibrate jump parameters
        # Returns are 1-minute bars → dt must match that frequency
        returns = self.loader.compute_returns()
        intraday_dt = 1.0 / (TRADING_DAYS_PER_YEAR * 390)  # 1-min bar in trading years
        jump_params = calibrate_from_returns(returns, dt=intraday_dt)
        
        if verbose:
            print(f"\nJump calibration:")
            print(f"  lambda_jump: {jump_params['lambda_jump']:.4f}")
            print(f"  mu_jump: {jump_params['mu_jump']:.4f}")
            print(f"  sigma_jump: {jump_params['sigma_jump']:.4f}")
        
        self.params.update(jump_params)
        
        # 7. Calibrate mean reversion
        mr_params = calibrate_from_intraday(
            self.loader.intraday_data, 
            self.vwap
        )
        
        if verbose:
            print(f"\nMean reversion calibration:")
            print(f"  theta_mr: {mr_params['theta_mr']:.4f}")
            print(f"  mu_0 (VWAP): ${mr_params['mu_0']:.2f}")
        
        self.params.update(mr_params)
        
        # Clear cached paths (params changed)
        self._cached_paths = None
        
        if verbose:
            print(f"\n✓ Calibration complete")
            print(f"{'='*60}\n")
        
        return self.params
    
    def _generate_paths(self, T):
        # Round T to nearest trading minute for cache efficiency
        # (within a single scan, T changes by seconds between strikes — irrelevant)
        T_bucket = round(T * 252 * 390) / (252 * 390)
        
        # Check cache
        if self._cached_paths is not None and self._cached_T == T_bucket:
            return self._cached_paths
        
        # Generate time grid
        times, dt_array = generate_time_grid(T_bucket, self.n_steps)
        
        # Generate random numbers
        if USE_ANTITHETIC:
            half_paths = self.n_paths // 2
            Z1_half, Z2_half = generate_correlated_normals(
                half_paths, self.n_steps, self.params['rho']
            )
            # Antithetic pairs
            Z1 = np.vstack([Z1_half, -Z1_half])
            Z2 = np.vstack([Z2_half, -Z2_half])
        else:
            Z1, Z2 = generate_correlated_normals(
                self.n_paths, self.n_steps, self.params['rho']
            )
        
        # Simulate paths
        S_paths, V_paths = simulate_combined_paths(
            S0=self.S0,
            v0=self.v0,
            params=self.params,
            times=times,
            dt_array=dt_array,
            Z1=Z1,
            Z2=Z2,
            use_jumps=True,
            use_mean_reversion=False  # Risk-neutral: no mean reversion
        )
        
        # Cache results (include T_used so pricing uses the exact T that dt_array was built from)
        self._cached_paths = {
            'S_paths': S_paths,
            'V_paths': V_paths,
            'times': times,
            'dt_array': dt_array,
            'Z1': Z1,
            'Z2': Z2,
            'T_used': T_bucket
        }
        self._cached_T = T_bucket
        
        return self._cached_paths
    
    def price_option(self, strike, T=None, option_type='call', use_control_variate=True,
                     market_iv=None, verbose=False):
        if self.S0 is None:
            raise ValueError("Must call calibrate_to_market() first")
        
        if T is None:
            T = get_time_to_expiry()
        
        # Single-source r from params (H1)
        r = self.params.get('r', RISK_FREE_RATE)
        
        # Generate paths
        paths_data = self._generate_paths(T)
        
        # Use the exact T that dt_array was built from (C2 — prevents CV guard failures)
        T_used = paths_data['T_used']
        
        # CV sigma: prefer per-strike market IV when available (fix #2)
        if market_iv is not None and market_iv > 0:
            sigma_BS = market_iv
        else:
            sigma_BS = np.sqrt(self.v0)
        
        # --- Diagnostic: T and dt_array consistency ---
        if verbose:
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo as _ZI
            _now_ny = _dt.now(_ZI("America/New_York"))
            _minutes_left = max(0, (16*60) - (_now_ny.hour*60 + _now_ny.minute))
            _dt_sum = float(np.sum(paths_data['dt_array']))
            print(f"  [DIAG] now(NY)={_now_ny.strftime('%H:%M:%S')}, "
                  f"mins_to_close={_minutes_left}, T_raw={T:.8f}, T_used={T_used:.8f}")
            print(f"  [DIAG] dt.sum()={_dt_sum:.8f}, T_used={T_used:.8f}, "
                  f"|dt.sum()-T_used|={abs(_dt_sum - T_used):.2e}")
        
        result = price_european_option(
            S_paths=paths_data['S_paths'],
            K=strike,
            T=T_used,
            r=r,
            option_type=option_type,
            use_control_variate=use_control_variate,
            sigma_BS=sigma_BS,
            dt_array=paths_data['dt_array'],
            Z1=paths_data['Z1']
        )
        
        # --- Distribution diagnostics ---
        if verbose:
            S_T = paths_data['S_paths'][:, -1]
            prob_itm = np.mean(S_T > strike) if option_type == 'call' else np.mean(S_T < strike)
            print(f"  [DIST] {strike}{option_type[0].upper()}: "
                  f"mean(S_T)={np.mean(S_T):.2f}, std(S_T)={np.std(S_T):.2f}, "
                  f"P(ITM)={prob_itm:.2%}, FV=${result['price']:.4f}")
        
        # --- Sanity checks on model price ---
        
        # Intrinsic value floor
        if option_type == 'call':
            intrinsic = max(self.S0 - strike, 0)
        else:
            intrinsic = max(strike - self.S0, 0)
        
        # Upper bound: call <= S0, put <= K
        if option_type == 'call':
            upper_bound = self.S0
        else:
            upper_bound = strike
        
        price = result['price']
        
        if price < intrinsic:
            print(f"⚠ WARNING: Model price ${price:.4f} < intrinsic ${intrinsic:.2f}. "
                  f"Check calibration/inputs.")
            if use_control_variate:
                # Retry without control variate — CV may be over-correcting
                print("  → Retrying without control variate...")
                result = price_european_option(
                    S_paths=paths_data['S_paths'],
                    K=strike,
                    T=T_used,
                    r=r,
                    option_type=option_type,
                    use_control_variate=False
                )
                price = result['price']
                if price < intrinsic:
                    print(f"  → Still below intrinsic. Clamping to ${intrinsic:.2f}")
                    result['price'] = intrinsic
                else:
                    print(f"  → CV fallback price: ${price:.4f} (OK)")
        
        if result['price'] > upper_bound:
            print(f"⚠ WARNING: Model price ${result['price']:.4f} > upper bound ${upper_bound:.2f}. "
                  f"Clamping.")
            result['price'] = upper_bound
        
        return result
    
    def get_trading_signal(self, strike, market_bid, market_ask, option_type='call',
                           market_iv=None):
        # Get model price (pass per-strike IV for better CV performance)
        result = self.price_option(strike, option_type=option_type, market_iv=market_iv)
        model_price = result['price']
        std_error = result['std_error']
        
        # Market mid price and spread
        market_mid = (market_bid + market_ask) / 2
        spread = market_ask - market_bid
        
        # 1. LIQUIDITY FILTER
        # If spread is too wide (> $0.10 or > 10% of price), don't trade
        max_spread = 0.10
        if spread > max_spread and spread / market_mid > 0.10:
            return {
                'action': 'HOLD',
                'edge': 0.0,
                'confidence': 0.0,
                'reason': f"Spread too wide (${spread:.2f}) - Illiquid/After-hours",
                'model_price': model_price,
                'market_mid': market_mid,
                'std_error': std_error,
                'spread': spread
            }

        # 2. CALCULATE EDGE ON EXECUTION PRICE
        # To BUY, we must pay ASK. To SELL, we must hit BID.
        # Edge must exist against the PRICE WE PAY, not the mid.
        
        # Case A: Model thinks it's cheap -> WE BUY at ASK
        # Edge = (Model Value - Ask Price) / Ask Price
        if model_price > market_ask:
            edge = (model_price - market_ask) / market_ask
            action_candidate = 'BUY'
            execution_price = market_ask
        
        # Case B: Model thinks it's expensive -> WE SELL at BID
        # Edge = (Bid Price - Model Value) / Bid Price
        # (Using Bid as baseline since that's cash in hand)
        elif model_price < market_bid:
            edge = (market_bid - model_price) / market_bid
            action_candidate = 'SELL'
            execution_price = market_bid
            
        else:
            # Model value is inside the spread -> No edge
            edge = 0.0
            action_candidate = 'HOLD'
            execution_price = market_mid

        # 3. CONFIDENCE SCORING
        if abs(edge) > 0:
            # Signal-to-noise ratio: Edge $ / Model Error $
            edge_dollars = abs(model_price - execution_price)
            if np.isfinite(std_error) and std_error > 1e-6:
                snr = edge_dollars / std_error
            else:
                # NaN or near-zero stderr = unreliable model → low confidence
                snr = 0.0
            confidence = min(1.0, snr / 3.0)
            
            # Cap confidence if model is inconsistent with BS sanity
            r_bs = self.params.get('r', RISK_FREE_RATE)
            bs_check = black_scholes(self.S0, strike, get_time_to_expiry(), 
                                     r_bs, np.sqrt(self.v0), option_type)
            if bs_check > 0 and abs(model_price - bs_check) / bs_check > 0.50:
                confidence = min(confidence, 0.30)
        else:
            confidence = 0.0
        
        # 4. TRADING THRESHOLDS
        min_edge = 0.02  # 2% edge required
        min_confidence = 0.5
        

        if action_candidate != 'HOLD' and edge > min_edge and confidence >= min_confidence:
            action = action_candidate
            if action == 'BUY':
                reason = f"Model ${model_price:.2f} > Ask ${market_ask:.2f} ({edge*100:.1f}% edge)"
            else:
                reason = f"Model ${model_price:.2f} < Bid ${market_bid:.2f} ({edge*100:.1f}% edge)"
        else:
            action = 'HOLD'
            reason = "No significant edge vs execution price"
        
        return {
            'action': action,
            'edge': edge if action != 'HOLD' else 0,
            'confidence': confidence,
            'reason': reason,
            'model_price': model_price,
            'market_mid': market_mid,
            'std_error': std_error,
            'spread': spread,
            'n_paths': result.get('n_paths'),
            'variance_reduction_factor': result.get('variance_reduction_factor'),
            'beta': result.get('beta'),
        }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("LIVE TRADING EXAMPLE")
    print("="*60)
    
    # Initialize
    trader = TradingSystem('SPY')
    
    # Calibrate to market
    trader.calibrate_to_market()
    
    # Example: Price ATM call
    T = get_time_to_expiry()
    atm_strike = round(trader.S0)
    
    print(f"\nPricing {atm_strike} call (T={T:.6f} years)...")
    result = trader.price_option(atm_strike, T, 'call')
    
    print(f"\nResults:")
    print(f"  Model Price: ${result['price']:.2f}")
    print(f"  Std Error:   ${result['std_error']:.4f}")
    print(f"  95% CI:      ${result['confidence_interval'][0]:.2f} - ${result['confidence_interval'][1]:.2f}")
    
    # Compare to Black-Scholes
    sigma = np.sqrt(trader.v0)
    bs_price = black_scholes(trader.S0, atm_strike, T, RISK_FREE_RATE, sigma, 'call')
    print(f"\n  BS Price:    ${bs_price:.2f}")
    print(f"  Difference:  ${result['price'] - bs_price:.4f}")
    
    print("\n" + "="*60)
