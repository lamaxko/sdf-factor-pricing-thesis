from datetime import date

from numpy import save
from test.factor_portfolio import FactorPortfolio

save_dir = r"out/factor/"
n_quantiles = 5
indices = ["S&P500", "S&P400", "S&P600"]
dates = (date(2002, 1, 1), date(2022, 1, 1))
portfolio = FactorPortfolio("RollingOmega12M_lag1_neg", indices, n_quantiles, dates)
# portfolio = FactorPortfolio("RollingOmBeta12M_lag1_Deep", indices, n_quantiles, dates)
portfolio.create_plots(save_dir, long=n_quantiles, short=1, freq_premium="y", annual_labels=False, monthly_trend=True)
portfolio.sanity_check()
