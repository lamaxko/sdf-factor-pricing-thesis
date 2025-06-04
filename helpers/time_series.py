import pandas as pd

class RetTimeSeries:
    def __init__(self, pct_returns: pd.Series):
        self.pct_returns = pct_returns

    def cum_returns(self):
        return (1 + self.pct_returns).cumprod().sub(1).fillna(0)

    def monthly_returns(self):
        cr = self.cum_returns()
        first = cr.resample('ME').first()
        last = cr.resample('ME').last()
        monthly = (last + 1).div(first + 1).sub(1)
        monthly.index = monthly.index.to_period('M').astype(str)
        return monthly
 
    def annual_returns(self):
        cr = self.cum_returns()
        first = cr.resample('YE').first()
        last  = cr.resample('YE').last()
        annual = (last + 1).div(first + 1).sub(1)
        annual.index = annual.index.to_period('Y').astype(str)
        return annual

