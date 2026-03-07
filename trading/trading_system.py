import numpy as np
from datetime import datetime

from config import (
    DEFAULT_PARAMS, RISK_FREE_RATE, N_PATHS_DEFAULT, N_STEPS_0DTE,
    TRADING_DAYS_PER_YEAR, get_time_to_expiry, USE_ANTITHETIC,
    TRADING_THRESHOLDS, SPOT_MAX_AGE_SECONDS, SPOT_CHANGE_THRESHOLD
)
from data.data_provider import MarketDataProvider, get_provider
from pricing.black_scholes import black_scholes

from core.time_grid import generate_time_grid
from core.random_numbers import generate_correlated_normals
from models.combined_model import simulate_combined_paths_fast as simulate_combined_paths
from models.jump_diffusion import validate_bernoulli_approximation
from pricing.european import price_european_option
from calibration.heston_calibrator import calibrate_to_realized_vol
from calibration.jump_calibrator import calibrate_from_returns
from calibration.mean_reversion_calibrator import calibrate_from_intraday


class TradingSystem:
    def __init__(self, ticker='SPY', provider: MarketDataProvider = None):
        self.ticker = ticker
        self.provider = provider or get_provider('yfinance')
        
        # Market state
        self.S0 = None
        self.spot_timestamp = None   # when S0 was last observed
        self.vwap = None
        self.historical_vol = None
        
        # Model parameters (start with defaults)
        self.params = DEFAULT_PARAMS.copy()
        self.v0 = self.params.get('v0', 0.04)
        
        # Simulation settings
        self.n_paths = N_PATHS_DEFAULT
        self.n_steps = N_STEPS_0DTE
        
        # Diurnal vol seasonality
        self._diurnal_params = None
        
        # Cache for efficiency
        self._cached_paths = None
        self._cached_T = None
        self._cached_S0 = None  # spot used to build cached paths
        
    def calibrate_to_market(self, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f"CALIBRATING MODEL TO MARKET - {self.ticker}")
            print(f"{'='*60}")
        
        # 1. Get current spot price
        spot_data = self.provider.get_spot_price(self.ticker)
        self.S0 = spot_data['price']
        self.spot_timestamp = spot_data['timestamp']
        
        # 2. Get intraday data for calibration
        from config import YF_PERIOD, YF_INTERVAL
        self._intraday = self.provider.get_intraday_data(
            self.ticker, period=YF_PERIOD, interval=YF_INTERVAL
        )
        
        # 3. Compute VWAP
        self.vwap = self.provider.compute_vwap(self._intraday)
        
        # 4. Get historical volatility
        self.historical_vol = self.provider.get_historical_volatility(self._intraday)
        
        # 5. Calibrate Heston parameters
        prices = self._intraday['Close'].values
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
            # Try today's date; fall back to nearest available expiration
            try:
                chain = spy.option_chain(today_str)
            except Exception:
                avail = spy.options
                if avail:
                    chain = spy.option_chain(avail[0])
                    if verbose:
                        print(f"\n  IV anchor: {today_str} not available, "
                              f"using nearest expiry {avail[0]}")
                else:
                    raise ValueError("No option expirations available")
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
        
        # 5c. IV surface calibration (fit Heston CF to full chain)
        from config import USE_IV_SURFACE_CALIBRATION, IV_SURFACE_MIN_STRIKES, IV_SURFACE_MONEYNESS
        if USE_IV_SURFACE_CALIBRATION:
            try:
                from calibration.heston_cf import calibrate_to_iv_surface
                T_now = get_time_to_expiry()
                from datetime import datetime as _dt2
                today_str2 = _dt2.now().strftime('%Y-%m-%d')
                chain_data = self.provider.get_option_chain(self.ticker, expiry_date=today_str2)
                calls_df = chain_data['calls']
                
                iv_params, iv_quality = calibrate_to_iv_surface(
                    calls_df, self.S0, T_now, RISK_FREE_RATE,
                    moneyness_range=IV_SURFACE_MONEYNESS,
                    min_strikes=IV_SURFACE_MIN_STRIKES,
                    verbose=verbose
                )
                
                # Quality gate: only accept if fit quality is adequate
                if iv_quality.get('acceptable', False):
                    # Blend: use IV-surface ρ and σ_v (these are poorly estimated
                    # from realized vol), keep realized-vol κ and θ_v
                    self.params['rho'] = iv_params['rho']
                    self.params['sigma_v'] = iv_params['sigma_v']
                    self.v0 = iv_params['v0']
                    self.params['v0'] = iv_params['v0']
                    
                    if verbose:
                        print(f"  → IV surface overrides: ρ={iv_params['rho']:.2f}, "
                              f"σ_v={iv_params['sigma_v']:.2f}, "
                              f"v0={np.sqrt(iv_params['v0']):.1%}")
                else:
                    if verbose:
                        print(f"  → IV surface fit rejected (RMSE=${iv_quality['rmse']:.4f}, "
                              f"MAPE={iv_quality['mape']:.1%}). Keeping prior params.")
            except Exception as e:
                if verbose:
                    print(f"\n  IV surface calibration skipped: {e}")
        
        # 6. Calibrate jump parameters
        # Returns are 1-minute bars → dt must match that frequency
        returns = self.provider.compute_returns(self._intraday)
        intraday_dt = 1.0 / (TRADING_DAYS_PER_YEAR * 390)  # 1-min bar in trading years
        jump_params = calibrate_from_returns(returns, dt=intraday_dt)
        
        if verbose:
            print(f"\nJump calibration (raw):")
            print(f"  lambda_jump: {jump_params['lambda_jump']:.4f}")
            print(f"  mu_jump: {jump_params['mu_jump']:.4f}")
            print(f"  sigma_jump: {jump_params['sigma_jump']:.4f}")
        
        # 6b. Regime-adaptive jump blending (#5)
        from config import USE_REGIME_JUMPS
        if USE_REGIME_JUMPS:
            from calibration.regime_detector import classify, blend_jump_params
            # Compute intraday move % from open
            open_price = self._intraday['Open'].iloc[0]
            intraday_move_pct = abs(self.S0 - open_price) / open_price * 100
            self._current_regime = classify(intraday_move_pct)
            self._intraday_move_pct = intraday_move_pct
            
            # Variance stability: ratio of recent (last 30 bars) to full-session variance
            recent_n = min(30, len(returns))
            recent_var = float(np.var(returns[-recent_n:])) if recent_n > 1 else 0.0
            full_var = float(np.var(returns)) if len(returns) > 1 else 1e-8
            variance_stability = recent_var / max(full_var, 1e-8)
            
            blended = blend_jump_params(
                jump_params, self._current_regime,
                n_returns=len(returns),
                variance_stability=variance_stability
            )
            
            if verbose:
                from calibration.regime_detector import regime_blend_weight
                w = regime_blend_weight(len(returns), variance_stability)
                print(f"\n  Regime: {self._current_regime} "
                      f"(move={intraday_move_pct:.2f}%)")
                print(f"  Blend weight w={w:.2f} "
                      f"(n={len(returns)}, var_ratio={variance_stability:.2f})")
                print(f"  Blended λ={blended['lambda_jump']:.2f} "
                      f"(raw={jump_params['lambda_jump']:.2f})")
            
            jump_params = blended
        else:
            self._current_regime = 'unknown'
            self._intraday_move_pct = None
        
        self.params.update(jump_params)
        
        # 7. Calibrate mean reversion
        mr_params = calibrate_from_intraday(
            self._intraday,
            self.vwap
        )
        
        if verbose:
            print(f"\nMean reversion calibration:")
            print(f"  theta_mr: {mr_params['theta_mr']:.4f}")
            print(f"  mu_0 (VWAP): ${mr_params['mu_0']:.2f}")
        
        self.params.update(mr_params)
        
        # 8. Calibrate intraday vol seasonality
        from config import USE_VOL_SEASONALITY
        if USE_VOL_SEASONALITY:
            try:
                from calibration.vol_seasonality import calibrate_diurnal_from_history
                self._diurnal_params = calibrate_diurnal_from_history(self._intraday)
                if verbose:
                    print(f"\nVol seasonality calibrated:")
                    print(f"  a={self._diurnal_params['a']:.2f}, "
                          f"b={self._diurnal_params['b']:.2f}, "
                          f"c1={self._diurnal_params['c1']:.1f}, "
                          f"d={self._diurnal_params['d']:.2f}, "
                          f"c2={self._diurnal_params['c2']:.1f}")
            except Exception as e:
                if verbose:
                    print(f"\n  Vol seasonality skipped: {e}")
                self._diurnal_params = None
        
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
            # Invalidate cache if spot has moved meaningfully
            if self._cached_S0 is not None:
                spot_pct_change = abs(self.S0 - self._cached_S0) / self._cached_S0
                if spot_pct_change > SPOT_CHANGE_THRESHOLD:
                    if hasattr(self, '_verbose_cache') and self._verbose_cache:
                        print(f"  [CACHE] Spot moved {spot_pct_change:.4%} "
                              f"({self._cached_S0:.2f}→{self.S0:.2f}), invalidating")
                    self._cached_paths = None
                else:
                    return self._cached_paths
            else:
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
        use_jumps = True
        bernoulli_violated = False

        # Bernoulli guard: check if approximation holds at this λ and dt
        lambda_jump = self.params.get('lambda_jump', 0)
        if lambda_jump > 0 and len(dt_array) > 0:
            dt_max = float(np.max(dt_array))
            valid, prob_multi = validate_bernoulli_approximation(
                lambda_jump, dt_max, threshold=0.01
            )
            if not valid:
                print(f"  ⚠ Bernoulli approx FAILED: λ={lambda_jump:.1f}, "
                      f"dt_max={dt_max:.6f}, P(≥2 jumps/step)={prob_multi:.4f}")
                print(f"  → Disabling jump component for this simulation")
                use_jumps = False
                bernoulli_violated = True

        # Compute diurnal weights for vol seasonality
        diurnal_w = None
        if self._diurnal_params is not None:
            from calibration.vol_seasonality import (
                compute_diurnal_weights,
                simulation_time_to_session_fraction,
                get_current_session_fraction,
            )
            session_frac = get_current_session_fraction()
            step_fracs = simulation_time_to_session_fraction(
                times, T_bucket, session_frac
            )
            # weights for each step (n_steps values, one per dt interval)
            diurnal_w = compute_diurnal_weights(
                step_fracs[:-1], self._diurnal_params  # exclude terminal point
            )

        S_paths, V_paths = simulate_combined_paths(
            S0=self.S0,
            v0=self.v0,
            params=self.params,
            times=times,
            dt_array=dt_array,
            Z1=Z1,
            Z2=Z2,
            use_jumps=use_jumps,
            use_mean_reversion=False,  # Risk-neutral: no mean reversion
            diurnal_weights=diurnal_w,
        )
        
        # Cache results (include T_used so pricing uses the exact T that dt_array was built from)
        self._cached_paths = {
            'S_paths': S_paths,
            'V_paths': V_paths,
            'times': times,
            'dt_array': dt_array,
            'Z1': Z1,
            'Z2': Z2,
            'T_used': T_bucket,
            'bernoulli_violated': bernoulli_violated,
        }
        self._cached_T = T_bucket
        self._cached_S0 = self.S0  # remember which spot generated these paths
        
        return self._cached_paths
    
    def _generate_paths_adaptive(self, T, strike, spread):
        """Error-budget-driven MC: run pilot batch, scale up if needed.
        
        This method does NOT use the T-bucket cache because different
        strikes may need different path counts.
        """
        from config import (N_PILOT_PATHS, TARGET_ABS_ERROR,
                            TARGET_SPREAD_FRAC, MAX_PATHS)
        
        T_bucket = round(T * 252 * 390) / (252 * 390)
        r = self.params.get('r', RISK_FREE_RATE)
        
        # Target error: max of absolute target and fraction of spread
        target_error = max(TARGET_ABS_ERROR, TARGET_SPREAD_FRAC * spread)
        
        # --- Pilot batch ---
        times, dt_array = generate_time_grid(T_bucket, self.n_steps)
        
        n_pilot = N_PILOT_PATHS
        if USE_ANTITHETIC:
            half = n_pilot // 2
            Z1_h, Z2_h = generate_correlated_normals(
                half, self.n_steps, self.params['rho'])
            Z1 = np.vstack([Z1_h, -Z1_h])
            Z2 = np.vstack([Z2_h, -Z2_h])
        else:
            Z1, Z2 = generate_correlated_normals(
                n_pilot, self.n_steps, self.params['rho'])
        
        # Diurnal weights
        diurnal_w = None
        if self._diurnal_params is not None:
            from calibration.vol_seasonality import (
                compute_diurnal_weights,
                simulation_time_to_session_fraction,
                get_current_session_fraction,
            )
            sf = get_current_session_fraction()
            step_fracs = simulation_time_to_session_fraction(
                times, T_bucket, sf)
            diurnal_w = compute_diurnal_weights(
                step_fracs[:-1], self._diurnal_params)
        
        # Bernoulli check
        use_jumps = True
        bernoulli_violated = False
        lam = self.params.get('lambda_jump', 0)
        if lam > 0 and len(dt_array) > 0:
            valid, _ = validate_bernoulli_approximation(
                lam, float(np.max(dt_array)), 0.01)
            if not valid:
                use_jumps = False
                bernoulli_violated = True
        
        S_paths, V_paths = simulate_combined_paths(
            S0=self.S0, v0=self.v0, params=self.params,
            times=times, dt_array=dt_array, Z1=Z1, Z2=Z2,
            use_jumps=use_jumps, use_mean_reversion=False,
            diurnal_weights=diurnal_w,
        )
        
        # Estimate std_error from pilot payoffs
        S_T = S_paths[:, -1]
        payoffs = np.maximum(S_T - strike, 0) * np.exp(-r * T_bucket)
        pilot_std = np.std(payoffs, ddof=1)
        pilot_se = pilot_std / np.sqrt(n_pilot)
        
        # Do we need more paths?
        if pilot_se <= target_error or n_pilot >= MAX_PATHS:
            return {
                'S_paths': S_paths, 'V_paths': V_paths,
                'times': times, 'dt_array': dt_array,
                'Z1': Z1, 'Z2': Z2,
                'T_used': T_bucket,
                'bernoulli_violated': bernoulli_violated,
                'adaptive_n': n_pilot,
            }
        
        # --- Scale up ---
        n_needed = int(np.ceil((1.96 * pilot_std / target_error) ** 2))
        n_needed = min(n_needed, MAX_PATHS)
        n_extra = n_needed - n_pilot
        
        if n_extra > 0:
            if USE_ANTITHETIC:
                half_e = n_extra // 2
                Z1e_h, Z2e_h = generate_correlated_normals(
                    half_e, self.n_steps, self.params['rho'])
                Z1e = np.vstack([Z1e_h, -Z1e_h])
                Z2e = np.vstack([Z2e_h, -Z2e_h])
            else:
                Z1e, Z2e = generate_correlated_normals(
                    n_extra, self.n_steps, self.params['rho'])
            
            S_extra, V_extra = simulate_combined_paths(
                S0=self.S0, v0=self.v0, params=self.params,
                times=times, dt_array=dt_array, Z1=Z1e, Z2=Z2e,
                use_jumps=use_jumps, use_mean_reversion=False,
                diurnal_weights=diurnal_w,
            )
            
            S_paths = np.vstack([S_paths, S_extra])
            V_paths = np.vstack([V_paths, V_extra])
            Z1 = np.vstack([Z1, Z1e])
            Z2 = np.vstack([Z2, Z2e])
        
        return {
            'S_paths': S_paths, 'V_paths': V_paths,
            'times': times, 'dt_array': dt_array,
            'Z1': Z1, 'Z2': Z2,
            'T_used': T_bucket,
            'bernoulli_violated': bernoulli_violated,
            'adaptive_n': S_paths.shape[0],
        }
    
    def price_option(self, strike, T=None, option_type='call', use_control_variate=True,
                     market_iv=None, verbose=False):
        if self.S0 is None:
            raise ValueError("Must call calibrate_to_market() first")
        
        if T is None:
            T = get_time_to_expiry()
        
        # Single-source r from params (H1)
        r = self.params.get('r', RISK_FREE_RATE)
        
        # Generate paths (adaptive or fixed)
        from config import USE_ADAPTIVE_PATHS
        use_adaptive = USE_ADAPTIVE_PATHS
        
        # Auto-enable adaptive paths for OTM or wide-spread strikes
        # even when globally OFF — MC error dominates edge on these
        if not use_adaptive and hasattr(self, '_current_spread') and self._current_spread > 0:
            otm = abs(strike - self.S0) if self.S0 else 0
            if otm > 2.0 or self._current_spread > 0.10:
                use_adaptive = True
        
        if use_adaptive and hasattr(self, '_current_spread') and self._current_spread > 0:
            paths_data = self._generate_paths_adaptive(T, strike, self._current_spread)
        else:
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
        
        # Propagate Bernoulli violation flag for confidence penalty
        result['bernoulli_violated'] = paths_data.get('bernoulli_violated', False)
        
        return result
    
    def get_trading_signal(self, strike, market_bid, market_ask, option_type='call',
                           market_iv=None):
        # ── Spot freshness gate ──────────────────────────────────
        from datetime import datetime as _dt
        if self.spot_timestamp is not None:
            spot_age = (_dt.now() - self.spot_timestamp).total_seconds()
        else:
            spot_age = float('inf')  # unknown age → treat as stale

        if spot_age > SPOT_MAX_AGE_SECONDS:
            return {
                'action': 'HOLD',
                'edge': 0.0,
                'confidence': 0.0,
                'reason': f"Spot data stale ({spot_age:.0f}s old, max={SPOT_MAX_AGE_SECONDS}s)",
                'model_price': 0.0,
                'market_mid': (market_bid + market_ask) / 2,
                'market_bid': market_bid,
                'market_ask': market_ask,
                'std_error': float('nan'),
                'spread': market_ask - market_bid,
                'spot_age_seconds': spot_age,
                'strike': strike,
                'option_type': option_type,
                'ticker': self.ticker,
                'required_edge': 0,
                'market_iv': market_iv,
                'lambda_jump': self.params.get('lambda_jump'),
                'v0': self.v0,
                'kappa': self.params.get('kappa'),
                'theta_v': self.params.get('theta_v'),
                'sigma_v': self.params.get('sigma_v'),
                'rho': self.params.get('rho'),
                'otm_dollars': 0.0,
            }

        # Get model price (pass per-strike IV for better CV performance)
        self._current_spread = market_ask - market_bid  # for adaptive MC
        result = self.price_option(strike, option_type=option_type, market_iv=market_iv)
        model_price = result['price']
        std_error = result['std_error']
        
        # Market mid price and spread
        market_mid = (market_bid + market_ask) / 2
        spread = market_ask - market_bid
        
        # 1. LIQUIDITY FILTER  (thresholds from config)
        max_spread = TRADING_THRESHOLDS['max_spread']
        max_spread_pct = TRADING_THRESHOLDS['max_spread_pct']
        if spread > max_spread and spread / market_mid > max_spread_pct:
            return {
                'action': 'HOLD',
                'edge': 0.0,
                'confidence': 0.0,
                'reason': f"Spread too wide (${spread:.2f}) - Illiquid/After-hours",
                'model_price': model_price,
                'market_mid': market_mid,
                'market_bid': market_bid,
                'market_ask': market_ask,
                'std_error': std_error,
                'spread': spread,
                'spot_age_seconds': spot_age,
                'strike': strike,
                'option_type': option_type,
                'ticker': self.ticker,
                'required_edge': 0,
                'market_iv': market_iv,
                'lambda_jump': self.params.get('lambda_jump'),
                'v0': self.v0,
                'kappa': self.params.get('kappa'),
                'theta_v': self.params.get('theta_v'),
                'sigma_v': self.params.get('sigma_v'),
                'rho': self.params.get('rho'),
                'otm_dollars': 0.0,
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

        # 3. CONFIDENCE SCORING (multiplicative penalties)
        if abs(edge) > 0:
            # ── Base confidence: MC precision ──
            edge_dollars = abs(model_price - execution_price)
            if np.isfinite(std_error) and std_error > 1e-6:
                snr = edge_dollars / std_error
            else:
                snr = 0.0
            sim_confidence = min(1.0, snr / 3.0)

            # ── Base confidence: edge magnitude ──
            # Tiny edge → low confidence even if SNR is huge
            edge_confidence = min(1.0, abs(edge) / 0.10)

            # Base = min of simulation and edge magnitude
            confidence = min(sim_confidence, edge_confidence)

            # ── Multiplicative penalty: spread ──
            # Wider spread → less certainty about fair value
            spread_penalty = max(0.1, 1.0 - spread / max(market_mid, 0.01))
            confidence *= spread_penalty

            # ── Multiplicative penalty: OTM distance ──
            import math
            if option_type == 'call':
                otm_dollars = max(0.0, strike - self.S0)
            else:
                otm_dollars = max(0.0, self.S0 - strike)
            otm_conf_decay = TRADING_THRESHOLDS['otm_conf_decay']
            otm_penalty = math.exp(-otm_conf_decay * otm_dollars)
            confidence *= otm_penalty

            # ── Multiplicative penalty: BS divergence ──
            r_bs = self.params.get('r', RISK_FREE_RATE)
            bs_check = black_scholes(self.S0, strike, get_time_to_expiry(), 
                                     r_bs, np.sqrt(self.v0), option_type)
            bs_div_cap = TRADING_THRESHOLDS['bs_divergence_cap']
            bs_div_conf = TRADING_THRESHOLDS['bs_divergence_conf']
            if bs_check > 0 and abs(model_price - bs_check) / bs_check > bs_div_cap:
                confidence *= bs_div_conf

            # ── Multiplicative penalty: Bernoulli violation ──
            if result.get('bernoulli_violated', False):
                confidence *= 0.3
        else:
            confidence = 0.0
            otm_dollars = 0.0
        
        # 4. TRADING THRESHOLDS  (from config, with OTM scaling)
        min_edge = TRADING_THRESHOLDS['min_edge']
        otm_edge_scale = TRADING_THRESHOLDS['otm_edge_scale']
        if option_type == 'call':
            otm_dollars_for_edge = max(0.0, strike - self.S0)
        else:
            otm_dollars_for_edge = max(0.0, self.S0 - strike)
        required_edge = min_edge + otm_edge_scale * otm_dollars_for_edge
        min_confidence = TRADING_THRESHOLDS['min_confidence']
        

        if action_candidate != 'HOLD' and edge > required_edge and confidence >= min_confidence:
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
            'market_bid': market_bid,
            'market_ask': market_ask,
            'std_error': std_error,
            'spread': spread,
            'cvar_95': result.get('cvar_95'),
            'payoff_mean_pos': result.get('payoff_mean_pos', 0.0),
            'payoff_mean_zero': result.get('payoff_mean_zero', 0.0),
            'payoff_frac_pos': result.get('payoff_frac_pos', 0.0),
            'n_paths': result.get('n_paths'),
            'variance_reduction_factor': result.get('variance_reduction_factor'),
            'beta': result.get('beta'),
            'spot_age_seconds': spot_age,
            'bernoulli_violated': result.get('bernoulli_violated', False),
            'strike': strike,
            'option_type': option_type,
            'ticker': self.ticker,
            'required_edge': required_edge,
            'market_iv': market_iv,
            'lambda_jump': self.params.get('lambda_jump'),
            'v0': self.v0,
            'kappa': self.params.get('kappa'),
            'theta_v': self.params.get('theta_v'),
            'sigma_v': self.params.get('sigma_v'),
            'rho': self.params.get('rho'),
            'calibration_flags': result.get('calibration_flags'),
            'otm_dollars': otm_dollars,
        }

    def update_spot(self):
        """Fetch fresh spot, update S0 + timestamp, conditionally invalidate cache."""
        spot_data = self.provider.get_spot_price(self.ticker)
        new_price = spot_data['price']

        # Check if cache needs invalidation
        if self.S0 is not None and self.S0 > 0:
            pct_change = abs(new_price - self.S0) / self.S0
            if pct_change > SPOT_CHANGE_THRESHOLD:
                self._cached_paths = None

        self.S0 = new_price
        self.spot_timestamp = spot_data['timestamp']
        return spot_data


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
