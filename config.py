import numpy as np
from datetime import datetime, time
#market parameters
RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_TRADING_DAY = 390

#Heston Model Parameters
HESTON_PARAMS = {
    'kappa': 2.0,           # Volatility = reversion speed
    'theta_v': 0.04,        # Long-term variance (20% vol)
    'sigma_v': 0.3,         # Volatility of volatility
    'rho': -0.7,            # Price-vol correlation (leverage effect)
    'v0': 0.04,             # Initial variance
}

#jump diffusion parameters
JUMP_PARAMS = {
    'lambda_jump': 2.0,     # Average jumps per year (annualized intensity)
    'mu_jump': -0.01,       # Mean jump size 
    'sigma_jump': 0.02,     # Jump size std dev
}   

#mean reversals
MEAN_REVERSION_PARAMS = {
    'theta_mr': 3.0,
    'mu_0': None,
}

DEFAULT_PARAMS = {
    **HESTON_PARAMS,
    **JUMP_PARAMS,
    **MEAN_REVERSION_PARAMS,
    'r': RISK_FREE_RATE,
    'measure': 'risk_neutral',
}

N_PATHS_DEFAULT = 50000      # Number of Monte Carlo paths
N_STEPS_DEFAULT = 100        # Number of time steps
N_STEPS_0DTE = 200          # More steps for 0DTE

USE_ANTITHETIC = True #variance reduction

# --- Trading thresholds (configurable; previously hardcoded) ---
TRADING_THRESHOLDS = {
    'min_edge': 0.02,           # Minimum edge required (2%)
    'min_confidence': 0.50,     # Minimum confidence score (0-1)
    'max_spread': 0.10,         # Maximum absolute spread ($)
    'max_spread_pct': 0.10,     # Maximum spread as % of mid
    'bs_divergence_cap': 0.50,  # Cap confidence if model diverges > 50% from BS
    'bs_divergence_conf': 0.30, # Confidence cap value when BS divergence triggers
    'otm_edge_scale': 0.01,     # Extra min_edge required per $ OTM
    'otm_conf_decay': 0.15,     # Confidence exponential decay rate per $ OTM
}

# --- Spot freshness / cache invalidation ---
SPOT_MAX_AGE_SECONDS = 10       # Suppress signals if spot older than this
SPOT_CHANGE_THRESHOLD = 0.001   # Invalidate path cache on 0.1% spot move

# --- Risk manager defaults ---
RISK_DEFAULTS = {
    'sl_pct': 0.50,             # Default stop-loss percentage (50%)
    'tp_pct': 1.00,             # Default take-profit percentage (100%)
    'close_time': (15, 45),     # Hard time exit (3:45 PM) as (hour, minute)
    'max_daily_loss': 500,      # Maximum daily loss in dollars
}

#optimize
HESTON_BOUNDS = [
    (0.1, 10.0),      # kappa
    (0.01, 1.0),      # theta_v
    (0.01, 2.0),      # sigma_v
    (-0.99, 0.0),     # rho
    (0.01, 1.0),      # v0
]

JUMP_THRESHOLD = 3.0

# --- Jump intensity control ---
MAX_LAMBDA_JUMP = 200.0         # Hard safety cap (annualized)
LAMBDA_PRIOR = 5.0              # Conservative prior (jumps/year)
LAMBDA_SHRINKAGE_ALPHA = 0.15   # Shrinkage weight: λ_eff = α·λ_calib + (1-α)·λ_prior

YF_PERIOD = "5d"          
YF_INTERVAL = "1m" 

MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50

# Moneyness range 
MONEYNESS_RANGE = (0.95, 1.05) 
MAX_PRICE = 1e6
MIN_PRICE = 0.01
MIN_VARIANCE = 1e-8

MIN_IMPLIED_VOL = 0.01
MAX_IMPLIED_VOL = 5.0

PRICE_DECIMALS = 2
ERROR_DECIMALS = 4
PERCENT_DECIMALS = 2

def get_time_to_expiry(expiry_time=None):
    """
    Calculate time to expiry in trading years.
    
    Uses America/New_York timezone. Clamps to regular session (9:30-16:00).
    Pre-market: returns full session (390 minutes).
    After-hours: returns minimum T (1 minute).
    
    Parameters:
    -----------
    expiry_time : datetime.time, optional
        Expiration time (default: 4:00 PM ET)
    
    Returns:
    --------
    T : float
        Time to expiry in trading years
    """
    from zoneinfo import ZoneInfo
    
    NY = ZoneInfo("America/New_York")
    now = datetime.now(NY)
    
    if expiry_time is None:
        expiry_time = time(16, 0)  # 4:00 PM ET
    
    market_open = time(9, 30)
    market_close = expiry_time  # 4:00 PM
    
    # Build timezone-aware expiry datetime
    expiry_datetime = datetime.combine(now.date(), market_close, tzinfo=NY)
    open_datetime = datetime.combine(now.date(), market_open, tzinfo=NY)
    
    # After market close → minimum T (1 trading minute)
    if now >= expiry_datetime:
        return 1.0 / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    # Before market open → full session
    if now <= open_datetime:
        minutes_to_expiry = MINUTES_PER_TRADING_DAY  # 390
    else:
        # During session → count actual trading minutes remaining
        minutes_to_expiry = (expiry_datetime - now).total_seconds() / 60
    
    # Convert to trading years
    T = minutes_to_expiry / (TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY)
    
    return T

def is_0dte(T):
    return T < (1.0 / TRADING_DAYS_PER_YEAR)

def print_config():
    print("="*60)
    print("0DTE OPTIONS PRICING ENGINE - CONFIGURATION")
    print("="*60)
    print(f"\nMarket Parameters:")
    print(f"  Risk-free rate: {RISK_FREE_RATE*100:.2f}%")
    print(f"  Trading days/year: {TRADING_DAYS_PER_YEAR}")
    
    print(f"\nSimulation Parameters:")
    print(f"  Default paths: {N_PATHS_DEFAULT:,}")
    print(f"  Default steps: {N_STEPS_DEFAULT}")
    print(f"  0DTE steps: {N_STEPS_0DTE}")
    print(f"  Antithetic variates: {USE_ANTITHETIC}")
    
    print(f"\nModel Parameters:")
    for key, value in DEFAULT_PARAMS.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    print("="*60)

if __name__ == "__main__":
    print_config()
    
    # Test time 
    T = get_time_to_expiry()
    print(f"\nCurrent time to 4PM expiry: {T:.6f} years")
    print(f"Is 0DTE: {is_0dte(T)}")
    print(f"Hours remaining: {T * TRADING_DAYS_PER_YEAR * 6.5:.2f}")