import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import warnings

from config import MIN_IMPLIED_VOL, MAX_IMPLIED_VOL, RISK_FREE_RATE

class BlackScholesModel:
    
    def __init__(self, r=RISK_FREE_RATE):
        self.r = r
    
    def price(self, S, K, T, sigma, option_type='call'):
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    def delta(self, S, K, T, sigma, option_type='call'):
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0
    
    def gamma(self, S, K, T, sigma):
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(self, S, K, T, sigma):
        """Vega: option price sensitivity to a 1 percentage-point move in IV.
        
        Returns dV/dσ / 100, so vega=0.15 means the option gains $0.15 
        for a 1pp rise in implied vol. Multiply by 100 for raw dV/dσ.
        """
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return S * norm.pdf(d1) * np.sqrt(T) / 100 
    
    def theta(self, S, K, T, sigma, option_type='call'):
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365
        else:
            term2 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365
        
        return theta
    
    def implied_volatility(self, option_price, S, K, T, option_type='call',
                          initial_guess=0.3):
        if T <= 0:
            return np.nan
        
        if option_type == 'call':
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        
        if option_price <= intrinsic:
            return np.nan
        
        def objective(sigma):
            try:
                bs_price = self.price(S, K, T, sigma, option_type)
                return bs_price - option_price
            except Exception:
                return 1e10
        
        try:
            iv = brentq(
                objective, 
                MIN_IMPLIED_VOL, 
                MAX_IMPLIED_VOL,
                maxiter=100,
                xtol=1e-6
            )
            return iv
        except Exception:
            try:
                result = minimize(
                    lambda sigma: abs(objective(sigma)),
                    initial_guess,
                    bounds=[(MIN_IMPLIED_VOL, MAX_IMPLIED_VOL)],
                    method='L-BFGS-B'
                )
                if result.success and abs(result.fun) < 0.01:
                    return result.x[0]
                else:
                    return np.nan
            except Exception:
                return np.nan
    
    def price_chain(self, S, strikes, T, sigma, option_type='call'):
        strikes = np.array(strikes)
        
        if np.isscalar(sigma):
            sigma = np.full_like(strikes, sigma, dtype=float)
        else:
            sigma = np.array(sigma)
        
        prices = np.array([
            self.price(S, K, T, sig, option_type)
            for K, sig in zip(strikes, sigma)
        ])
        
        return prices
    
    def greeks(self, S, K, T, sigma, option_type='call'):
        """
        Calculate all Greeks at once
        
        Returns:
        --------
        dict : Dictionary of Greeks
        """
        return {
            'delta': self.delta(S, K, T, sigma, option_type),
            'gamma': self.gamma(S, K, T, sigma),
            'vega': self.vega(S, K, T, sigma),
            'theta': self.theta(S, K, T, sigma, option_type),
        }


def quick_price(S, K, T, sigma, option_type='call', r=RISK_FREE_RATE):
    model = BlackScholesModel(r=r)
    return model.price(S, K, T, sigma, option_type)

def quick_iv(option_price, S, K, T, option_type='call', r=RISK_FREE_RATE):
    model = BlackScholesModel(r=r)
    return model.implied_volatility(option_price, S, K, T, option_type)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    model = BlackScholesModel(r=r)
    return model.price(S, K, T, sigma, option_type)


