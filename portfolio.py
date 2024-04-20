import pandas as pd
from datetime import datetime, timedelta, date
import pandas_datareader as pdr
import yfinance as yf
import os
import statsmodels.api as sm
import cvxpy as cp
from arch.__future__ import reindexing
from arch import arch_model
import numpy as np

class portfolio():
    def __init__(self, ohlc : pd.DataFrame()):
        self.equity = {}
        self.ohlc = ohlc
        self.w_min_risk = None
        self.pos = pd.Series()
        self.df_pos_ohlc = pd.DataFrame()
        
    def update_pos(self, symbol, volume):    
        self.equity[symbol] = self.equity.get(symbol, 0) + volume
        self.equity[symbol] = max(self.equity[symbol], 0)
        if self.equity[symbol] == 0:
            self.equity.pop(symbol)
        self.pos = pd.Series({k: v for k, v in self.equity.items()}, name="pos")
        self.update_values()
            
    def update_values(self):
        #pos = pd.Series({k: v for k, v in portf.equity.items()}, name="pos")
        self.df_pos_ohlc = self.ohlc[self.ohlc.index.get_level_values(1).isin(list(self.equity.keys()))]
        latest_price = self.df_pos_ohlc.loc[self.df_pos_ohlc.index.get_level_values(0)[-1:]]['Close'].reset_index().set_index('symbol')
        df_pos = self.pos.to_frame()
        df_pos.index.name = 'symbol'
        df = latest_price.merge(df_pos,left_index=True,right_index=True).reset_index().set_index('symbol')
        self.values =  (df['Close'] * df['pos']).sum()
        
    def get_values(self):
        return self.values
    
    def pct_change(self, lookback_days=252):
        df = self.ohlc[self.ohlc.index.get_level_values(1).isin(list(self.equity.keys()))]
        df =  df.loc[df.index.get_level_values(0)[-lookback_days:]]['Close']
        df_pct_change = df.groupby(level=[1]).pct_change().unstack().iloc[1:]
        return df_pct_change
    
    def weight(self):
        #df_pos = pd.Series({k: [v] for k, v in portf.equity.items()})
        #pos = pd.Series({k: v for k, v in portf.equity.items()}, name="pos")
        latest_price = self.df_pos_ohlc.loc[self.df_pos_ohlc.index.get_level_values(0)[-1:]]['Close'].reset_index().set_index('symbol')
        df = latest_price.merge(self.pos.to_frame(),left_index=True,right_index=True).reset_index().set_index('symbol')
        self.values =  (df['Close'] * df['pos']).sum()
        df['weight'] = df['Close'] * df['pos']
        df['weight'] = df['weight'] / self.get_values()
        return df['weight']
            
    def mu(self, lookback_days=252, method='hist'):
        #Annual expected return
        if method == 'hist':
            return self.pct_change(lookback_days).mean() *252
        elif method == 'capm':
            self.capm()
            return
        
    def capm_ols(self, y):
        df = y.to_frame().merge(self.bm.to_frame(), left_index=True,right_index=True)
        df_monthly =  df.resample('MS').first().pct_change()
        df_monthly =  df_monthly.merge(self.rf_rate.to_frame(), left_index=True,right_index=True).dropna()
        y = (df_monthly.iloc[:, 0] - df_monthly.iloc[:, 2] / 100 / 12).values
        X = (df_monthly.iloc[:, 1] - df_monthly.iloc[:, 2] / 100 / 12).values
        model = sm.OLS(y,X) #E
        results = model.fit() #F            
        return results.params[0]
        
    def capm(self):
        df_open = self.df_pos_ohlc['Open'].unstack() 
        df_beta = df_open.apply(self.capm_ols)
        return df_beta
    
    def garch_calc(self, rtn):
        am = arch_model(rtn.dropna().values*100)
        res = am.fit(disp=False)
        forecasts = res.forecast(horizon=63)
        vol_forecast = (forecasts.residual_variance.iloc[-1,:].sum() *252 / 63) ** 0.5 / 100
        return vol_forecast
        
    def garch(self):
        df_returns = self.df_pos_ohlc['Open'].unstack().pct_change()
        df_sigma = df_returns.apply(self.garch_calc)
        return df_sigma
    
    def sigma(self, method='hist'):
        if method == 'hist':
            return self.pct_change().std() *(252 ** 0.5)
        elif method == 'garch':
            return self.garch()
        
    def corr (self):
        return self.pct_change().corr()
    
    def cov (self):
        #np.diag(portf.sigma())@portf.corr()@np.diag(portf.sigma())
        return self.pct_change().cov() *252
    
    def mu_p(self):
        df =  self.weight().to_frame().merge(self.mu().to_frame(), left_index=True,right_index=True)
        return (df.iloc[:, 0] * df.iloc[:, 1]).sum()
    
    def sigma_p(self):
        w = self.weight().values
        c = self.cov().values
        return (w @ c @ w.T) ** 0.5
    
    def opt_min_risk(self):
        sigma = self.cov().values
        mu = self.mu().values
        mu_p = self.mu_p()
        N = len(mu)
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, sigma))
        constraints = [w.T @ mu >= mu_p, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
    
        df =  self.mu()
        if result != -np.inf:  
            self.w_min_risk = pd.Series(w.value.round(2), df.index)
            print (f"min_risk risk={round(result, 2)} Expected return={round((self.mu() * w.value.round(2)).sum(), 2)}")
            print (self.w_min_risk)
        
    def opt_max_mean_with_sigma(self, sigma_b=0.2):
        sigma = self.cov().values
        mu = self.mu().values
        N = len(mu)
        w = cp.Variable(N)
        objective = cp.Maximize(mu.T @ w)
        constraints = [cp.quad_form(w, sigma) <= sigma_b ** 2, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        df =  self.mu()
        if result != -np.inf:
            self.w_max_mean_with_sigma = pd.Series(w.value.round(2), df.index)
            print (f"max_mean_with_sigma {sigma_b} Expected return={result}")
            print (self.w_max_mean_with_sigma)    
    
    def optimize(self, goal="min_risk"):
        if goal == 'min_risk':
            self.opt_min_risk()
        elif goal == 'max_expect_mean':
            self.opt_max_mu()