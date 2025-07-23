import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')

@dataclass
class Option:
    S: float  # spot price
    K: float  # strike
    T: float  # time to expiry (years)
    r: float  # risk free rate
    q: float = 0.0  # dividend yield
    sigma: float = None  # volatility
    type: str = 'call'  # call or put
    american: bool = False

class BlackScholes:
    def __init__(self):
        self.name = "Black-Scholes"
    
    def _d1_d2(self, S, K, T, r, q, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def price(self, opt):
        if opt.sigma is None:
            raise ValueError("Need volatility for BS pricing")
        
        if opt.T <= 0:
            # at expiration
            if opt.type.lower() == 'call':
                return max(opt.S - opt.K, 0)
            else:
                return max(opt.K - opt.S, 0)
        
        d1, d2 = self._d1_d2(opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma)
        
        if opt.type.lower() == 'call':
            price = (opt.S * np.exp(-opt.q * opt.T) * stats.norm.cdf(d1) - 
                    opt.K * np.exp(-opt.r * opt.T) * stats.norm.cdf(d2))
        else:
            price = (opt.K * np.exp(-opt.r * opt.T) * stats.norm.cdf(-d2) - 
                    opt.S * np.exp(-opt.q * opt.T) * stats.norm.cdf(-d1))
        
        return max(price, 0)
    
    def greeks(self, opt):
        if opt.sigma is None:
            raise ValueError("Need volatility for Greeks")
        
        if opt.T <= 0:
            return {greek: 0.0 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']}
        
        S, K, T, r, q, sigma = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        d1, d2 = self._d1_d2(S, K, T, r, q, sigma)
        
        nd1 = stats.norm.cdf(d1)
        nd2 = stats.norm.cdf(d2)
        nprime_d1 = stats.norm.pdf(d1)
        
        if opt.type.lower() == 'call':
            delta = np.exp(-q * T) * nd1
            rho = K * T * np.exp(-r * T) * nd2 / 100
        else:
            delta = -np.exp(-q * T) * stats.norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        gamma = (np.exp(-q * T) * nprime_d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * nprime_d1 * np.sqrt(T) / 100
        
        if opt.type.lower() == 'call':
            theta = ((-S * nprime_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * nd2 + 
                    q * S * np.exp(-q * T) * nd1) / 365
        else:
            theta = ((-S * nprime_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * stats.norm.cdf(-d2) - 
                    q * S * np.exp(-q * T) * stats.norm.cdf(-d1)) / 365
        
        return {
            'delta': delta,
            'gamma': gamma, 
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class MonteCarlo:
    def __init__(self, n_sims=100000, n_steps=252, antithetic=True):
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.name = "Monte Carlo"
    
    def _simulate_paths(self, opt):
        dt = opt.T / self.n_steps
        n_sims = self.n_sims // 2 if self.antithetic else self.n_sims
        
        paths = np.zeros((n_sims, self.n_steps + 1))
        paths[:, 0] = opt.S
        
        Z = np.random.standard_normal((n_sims, self.n_steps))
        
        drift = (opt.r - opt.q - 0.5 * opt.sigma**2) * dt
        diffusion = opt.sigma * np.sqrt(dt)
        
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
        # antithetic paths
        if self.antithetic:
            paths_anti = np.zeros((n_sims, self.n_steps + 1))
            paths_anti[:, 0] = opt.S
            
            for t in range(1, self.n_steps + 1):
                paths_anti[:, t] = paths_anti[:, t-1] * np.exp(drift - diffusion * Z[:, t-1])
            
            paths = np.vstack([paths, paths_anti])
        
        return paths
    
    def _payoff(self, S, K, option_type):
        if option_type.lower() == 'call':
            return np.maximum(S - K, 0)
        else:
            return np.maximum(K - S, 0)
    
    def _american_lsmc(self, opt, paths):
        # longstaff-schwartz for american options
        dt = opt.T / self.n_steps
        discount = np.exp(-opt.r * dt)
        
        exercise_vals = self._payoff(paths[:, -1], opt.K, opt.type)
        
        for t in range(self.n_steps - 1, 0, -1):
            S_t = paths[:, t]
            intrinsic = self._payoff(S_t, opt.K, opt.type)
            
            itm_mask = intrinsic > 0
            
            if np.sum(itm_mask) > 0:
                S_itm = S_t[itm_mask]
                # polynomial basis for regression
                X = np.column_stack([
                    np.ones(len(S_itm)),
                    S_itm,
                    S_itm**2,
                    np.maximum(S_itm - opt.K, 0) if opt.type.lower() == 'call' 
                    else np.maximum(opt.K - S_itm, 0)
                ])
                
                Y = exercise_vals[itm_mask] * discount
                
                try:
                    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
                    continuation_val = np.zeros(len(S_t))
                    continuation_val[itm_mask] = X @ coeffs
                except:
                    continuation_val = exercise_vals * discount
                
                exercise_mask = intrinsic > continuation_val
                exercise_vals[exercise_mask] = intrinsic[exercise_mask]
                exercise_vals[~exercise_mask] *= discount
            else:
                exercise_vals *= discount
        
        return np.mean(exercise_vals) * np.exp(-opt.r * dt)
    
    def price(self, opt):
        if opt.sigma is None:
            raise ValueError("Need volatility for MC pricing")
        
        paths = self._simulate_paths(opt)
        
        if opt.american:
            price = self._american_lsmc(opt, paths)
        else:
            # european
            terminal_payoffs = self._payoff(paths[:, -1], opt.K, opt.type)
            price = np.exp(-opt.r * opt.T) * np.mean(terminal_payoffs)
        
        return max(price, 0)
    
    def greeks(self, opt):
        # finite differences
        base_price = self.price(opt)
        
        # delta
        opt_up = Option(**opt.__dict__)
        opt_up.S *= 1.01
        delta = (self.price(opt_up) - base_price) / (0.01 * opt.S)
        
        # gamma
        opt_down = Option(**opt.__dict__)
        opt_down.S *= 0.99
        gamma = (self.price(opt_up) - 2 * base_price + self.price(opt_down)) / ((0.01 * opt.S)**2)
        
        # theta
        if opt.T > 1/365:
            opt_theta = Option(**opt.__dict__)
            opt_theta.T -= 1/365
            theta = self.price(opt_theta) - base_price
        else:
            theta = 0
        
        # vega
        if opt.sigma:
            opt_vega = Option(**opt.__dict__)
            opt_vega.sigma += 0.01
            vega = self.price(opt_vega) - base_price
        else:
            vega = 0
        
        # rho
        opt_rho = Option(**opt.__dict__)
        opt_rho.r += 0.01
        rho = self.price(opt_rho) - base_price
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class BinomialTree:
    def __init__(self, n_steps=100):
        self.n_steps = n_steps
        self.name = "Binomial Tree"
    
    def price(self, opt):
        if opt.sigma is None:
            raise ValueError("Need volatility for tree pricing")
        
        dt = opt.T / self.n_steps
        u = np.exp(opt.sigma * np.sqrt(dt))  # up move
        d = 1 / u  # down move
        p = (np.exp((opt.r - opt.q) * dt) - d) / (u - d)  # risk neutral prob
        
        if p < 0 or p > 1:
            raise ValueError("Invalid tree parameters")
        
        # stock price tree
        stock_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        stock_prices[0, 0] = opt.S
        
        for i in range(1, self.n_steps + 1):
            for j in range(i + 1):
                stock_prices[i, j] = opt.S * (u**(i-j)) * (d**j)
        
        # option values
        option_vals = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # terminal payoffs
        for j in range(self.n_steps + 1):
            if opt.type.lower() == 'call':
                option_vals[self.n_steps, j] = max(stock_prices[self.n_steps, j] - opt.K, 0)
            else:
                option_vals[self.n_steps, j] = max(opt.K - stock_prices[self.n_steps, j], 0)
        
        # work backwards
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                expected_val = (p * option_vals[i + 1, j] + 
                               (1 - p) * option_vals[i + 1, j + 1]) * np.exp(-opt.r * dt)
                
                if opt.american:
                    if opt.type.lower() == 'call':
                        intrinsic = max(stock_prices[i, j] - opt.K, 0)
                    else:
                        intrinsic = max(opt.K - stock_prices[i, j], 0)
                    
                    option_vals[i, j] = max(expected_val, intrinsic)
                else:
                    option_vals[i, j] = expected_val
        
        return option_vals[0, 0]
    
    def greeks(self, opt):
        # finite differences
        base_price = self.price(opt)
        
        h = 0.01 * opt.S
        opt_up = Option(**opt.__dict__)
        opt_up.S += h
        opt_down = Option(**opt.__dict__)
        opt_down.S -= h
        
        delta = (self.price(opt_up) - self.price(opt_down)) / (2 * h)
        gamma = (self.price(opt_up) - 2 * base_price + self.price(opt_down)) / (h**2)
        
        if opt.T > 1/365:
            opt_theta = Option(**opt.__dict__)
            opt_theta.T -= 1/365
            theta = self.price(opt_theta) - base_price
        else:
            theta = 0
        
        opt_vega = Option(**opt.__dict__)
        opt_vega.sigma += 0.01
        vega = self.price(opt_vega) - base_price
        
        opt_rho = Option(**opt.__dict__)
        opt_rho.r += 0.01
        rho = self.price(opt_rho) - base_price
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class ImpliedVol:
    def __init__(self, model=None):
        self.model = model or BlackScholes()
        self.max_iter = 100
        self.tol = 1e-6
    
    def _objective(self, sigma, opt, target_price):
        try:
            opt.sigma = sigma
            return self.model.price(opt) - target_price
        except:
            return np.inf
    
    def _vega(self, opt):
        try:
            greeks = self.model.greeks(opt)
            return greeks['vega'] * 100
        except:
            return 0.001
    
    def newton_raphson(self, opt, target_price, initial_guess=0.2):
        sigma = initial_guess
        
        for i in range(self.max_iter):
            opt.sigma = sigma
            price_diff = self._objective(sigma, opt, target_price)
            
            if abs(price_diff) < self.tol:
                return sigma
            
            vega = self._vega(opt)
            if abs(vega) < 1e-10:
                break
            
            sigma_new = sigma - price_diff / vega
            
            if sigma_new <= 0 or sigma_new > 5:
                break
            
            if abs(sigma_new - sigma) < self.tol:
                return sigma_new
            
            sigma = sigma_new
        
        return None
    
    def brent_method(self, opt, target_price, bounds=(0.001, 5.0)):
        try:
            opt.sigma = bounds[0]
            f_low = self._objective(bounds[0], opt, target_price)
            
            opt.sigma = bounds[1] 
            f_high = self._objective(bounds[1], opt, target_price)
            
            if f_low * f_high > 0:
                return None
            
            result = brentq(
                lambda sigma: self._objective(sigma, opt, target_price),
                bounds[0], bounds[1],
                xtol=self.tol,
                maxiter=self.max_iter
            )
            
            return result if 0 < result < 5 else None
            
        except:
            return None
    
    def calculate(self, opt, target_price, method='brent'):
        if target_price <= 0:
            return {'iv': None, 'method': None, 'error': 'Invalid price'}
        
        # check arbitrage bounds
        if opt.type.lower() == 'call':
            lower_bound = max(opt.S * np.exp(-opt.q * opt.T) - 
                            opt.K * np.exp(-opt.r * opt.T), 0)
            upper_bound = opt.S * np.exp(-opt.q * opt.T)
        else:
            lower_bound = max(opt.K * np.exp(-opt.r * opt.T) - 
                            opt.S * np.exp(-opt.q * opt.T), 0)
            upper_bound = opt.K * np.exp(-opt.r * opt.T)
        
        if target_price < lower_bound or target_price > upper_bound:
            return {'iv': None, 'method': None, 'error': 'Price outside bounds'}
        
        if method.lower() == 'brent':
            iv = self.brent_method(opt, target_price)
            if iv is not None:
                return {'iv': iv, 'method': 'Brent', 'status': 'success'}
            
            iv = self.newton_raphson(opt, target_price)
            if iv is not None:
                return {'iv': iv, 'method': 'Newton-Raphson (fallback)', 'status': 'success'}
        
        else:
            iv = self.newton_raphson(opt, target_price)
            if iv is not None:
                return {'iv': iv, 'method': 'Newton-Raphson', 'status': 'success'}
            
            iv = self.brent_method(opt, target_price)
            if iv is not None:
                return {'iv': iv, 'method': 'Brent (fallback)', 'status': 'success'}
        
        return {'iv': None, 'method': None, 'error': 'Failed to converge'}

class OptionsCalculator:
    def __init__(self):
        self.bs = BlackScholes()
        self.mc = MonteCarlo(n_sims=50000)
        self.tree = BinomialTree(n_steps=100)
        self.iv_calc = ImpliedVol()
    
    def price_option(self, opt, models=['bs']):
        results = {}
        
        model_map = {'bs': self.bs, 'mc': self.mc, 'tree': self.tree}
        
        for model_name in models:
            if model_name in model_map:
                try:
                    price = model_map[model_name].price(opt)
                    results[model_name] = price
                except Exception as e:
                    results[model_name] = f"Error: {str(e)}"
        
        return results
    
    def get_greeks(self, opt, model='bs'):
        model_map = {'bs': self.bs, 'mc': self.mc, 'tree': self.tree}
        
        if model not in model_map:
            raise ValueError(f"Unknown model: {model}")
        
        return model_map[model].greeks(opt)
    
    def implied_vol(self, opt, market_price):
        return self.iv_calc.calculate(opt, market_price)
    
    def scenario_analysis(self, opt, spot_range=None, vol_range=None, n_points=21):
        if spot_range is None:
            spot_range = (opt.S * 0.8, opt.S * 1.2)
        if vol_range is None:
            vol_range = (opt.sigma * 0.5, opt.sigma * 1.5)
        
        spots = np.linspace(spot_range[0], spot_range[1], n_points)
        vols = np.linspace(vol_range[0], vol_range[1], n_points)
        
        results = []
        for S_new in spots:
            for vol_new in vols:
                opt_scenario = Option(**opt.__dict__)
                opt_scenario.S = S_new
                opt_scenario.sigma = vol_new
                
                price = self.bs.price(opt_scenario)
                greeks = self.bs.greeks(opt_scenario)
                
                results.append({
                    'spot': S_new,
                    'vol': vol_new,
                    'price': price,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega']
                })
        
        return pd.DataFrame(results)
    
    def plot_payoff(self, opt, spot_range=None):
        if spot_range is None:
            spot_range = (opt.S * 0.7, opt.S * 1.3)
        
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        prices = []
        deltas = []
        
        for S_new in spots:
            opt_scenario = Option(**opt.__dict__)
            opt_scenario.S = S_new
            
            price = self.bs.price(opt_scenario)
            greeks = self.bs.greeks(opt_scenario)
            
            prices.append(price)
            deltas.append(greeks['delta'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(spots, prices, 'b-', linewidth=2, label='Option Price')
        ax1.axvline(x=opt.S, color='r', linestyle='--', alpha=0.7, label='Current Spot')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Option Price')
        ax1.set_title(f'{opt.type.title()} Option')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(spots, deltas, 'g-', linewidth=2, label='Delta')
        ax2.axvline(x=opt.S, color='r', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Spot Price')
        ax2.set_ylabel('Delta')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    print("Options Pricing Calculator")
    print("=========================\n")
    
    # example option
    opt = Option(
        S=100,
        K=105,
        T=0.25,  # 3 months
        r=0.05,
        q=0.02,
        sigma=0.2,
        type='call',
        american=False
    )
    
    calc = OptionsCalculator()
    
    # 1. price comparison
    print("1. Option Prices:")
    prices = calc.price_option(opt, models=['bs', 'mc', 'tree'])
    for model, price in prices.items():
        print(f"   {model.upper()}: ${price:.4f}")
    
    # 2. greeks
    print("\n2. Greeks (Black-Scholes):")
    greeks = calc.get_greeks(opt, model='bs')
    for greek, value in greeks.items():
        print(f"   {greek.title()}: {value:.4f}")
    
    # 3. implied vol
    print("\n3. Implied Volatility:")
    market_price = 3.50
    iv_result = calc.implied_vol(opt, market_price)
    if iv_result['iv']:
        print(f"   Market Price: ${market_price:.2f}")
        print(f"   Implied Vol: {iv_result['iv']:.2%}")
        print(f"   Method: {iv_result['method']}")
    
    # 4. american vs european
    print("\n4. American vs European:")
    opt_american = Option(**opt.__dict__)
    opt_american.american = True
    
    eur_price = calc.mc.price(opt)
    amer_price = calc.mc.price(opt_american)
    premium = amer_price - eur_price
    
    print(f"   European: ${eur_price:.4f}")
    print(f"   American: ${amer_price:.4f}")
    print(f"   Early Exercise Premium: ${premium:.4f}")
    
    # 5. plot
    print("\n5. Plotting payoff diagram...")
    calc.plot_payoff(opt)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
