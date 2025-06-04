import os
import numpy as np
import pandas as pd
from datetime import date

from test.factor_portfolio import FactorPortfolio
from apis.kfrench import KfApi
from helpers.db_manager import DB
from helpers.time_series import RetTimeSeries

indice_mapper = {
                "S&P 500": "S&P500",
                "S&P 400": "S&P400",
                "S&P 600": "S&P600"
                }


def get_mon_ret(name, indices):
    n_quantiles = 5
    dates = (date(2002, 1, 1), date(2022, 1, 1))
    portfolio = FactorPortfolio(name, indices, n_quantiles, dates)
    mon_ret = portfolio.monthly_returns
    mon_ret.index = pd.to_datetime(mon_ret.index)

    return mon_ret

def get_benchmarks_ret(indices=["S&P500", "S&P400", "S&P600"]):
    df = DB.fetch(f"""
        SELECT
          r.date,
          MAX(CASE WHEN sm.dscd = 'S&PMIDC' THEN r.return END) AS "S&P400",
          MAX(CASE WHEN sm.dscd = 'S&PCOMP' THEN r.return END) AS "S&P500",
          MAX(CASE WHEN sm.dscd = 'S&P600I' THEN r.return END) AS "S&P600"
        FROM
          returns_monthly r
          JOIN stock_mapper sm ON r.id_stock = sm.id_stock
        WHERE
          sm.dscd IN ('S&PMIDC', 'S&PCOMP', 'S&P600I')
        GROUP BY
          r.date
        ORDER BY
          r.date;
    """,
    output="df")
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns.name = 'index_name'
    df = df 
    df = df[[col for col in indices if col in df.columns]]

    # df = RetTimeSeries(df).cum_returns()

    return df 

def merge_ret(ret, feature):

    ret.index = pd.to_datetime(ret.index)
    feature.index = pd.to_datetime(feature.index)
    merged = ret.join(feature, how='left')
    merged = merged.dropna()

    return merged

def get_data(indices=["S&P500", "S&P400", "S&P600"], cache=False):
    save = r"out/stats/"

    if os.path.exists(save + "cache.csv") and cache:
        ret = pd.read_csv(save + "cache.csv", index_col="date")
        ret.index = pd.to_datetime(ret.index)
        return ret
        

    name_omeg = "RollingOmega12M_lag1_neg"
    name_beta = "RollingOmBeta12M_lag1_neg"

    returns_om = get_mon_ret(name_omeg, indices)
    returns_om.rename(columns={1: "Q1_Omega", 2: "Q2_Omega", 3: "Q3_Omega", 4: "Q4_Omega", 5: "Q5_Omega"},
                      inplace=True)

    returns_be = get_mon_ret(name_beta, indices)
    returns_be.rename(columns={1: "Q1_Beta", 2: "Q2_Beta", 3: "Q3_Beta", 4: "Q4_Beta", 5: "Q5_Beta"},
                      inplace=True)

    returns = merge_ret(returns_om, returns_be)

    rf = KfApi().clean_data_ff5()['RF'] / 100
    returns = merge_ret(returns, rf)

    bench = get_benchmarks_ret(indices=indices)
    returns = merge_ret(returns, bench)

    if cache:
        os.makedirs(save)
        returns.to_csv(save + "cache.csv")

    return returns


# Function to calculate annualized return
def annualized_return(series):
    return (1 + series.mean())**12 - 1

# Function to calculate annualized volatility
def annualized_volatility(series):
    return series.std() * np.sqrt(12)

# Function to calculate Sharpe Ratio
def sharpe_ratio(series, rf):
    excess_ret = series - rf
    return excess_ret.mean() / series.std() * np.sqrt(12)

# Function to calculate Beta
def beta(series, benchmark):
    cov = np.cov(series, benchmark)[0, 1]
    var = np.var(benchmark)
    return cov / var

# Function to calculate Jensen's Alpha
def jensens_alpha(series, benchmark, beta_val, rf):
    return series.mean() - (beta_val * benchmark.mean() + rf.mean())

# Function to calculate Max Drawdown
def max_drawdown(series):
    cum_returns = (1 + series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Function to calculate Value at Risk (95%)
def value_at_risk(series, alpha=0.05):
    return np.quantile(series, alpha)

# Function to calculate Expected Shortfall (95%)
def expected_shortfall(series, alpha=0.05):
    var = value_at_risk(series, alpha)
    return series[series <= var].mean()

# Wrap all metrics in one function
def compute_backtest_metrics(df, indices=["S&P500", "S&P400", "S&P600"]):
    results = []

    rf = df["RF"]


    benchmarks = {k: df[k] for k in ["S&P500", "S&P400", "S&P600"] if k in indices and k in df.columns}

    asset_cols = [col for col in df.columns if col.startswith("Q")]
    asset_cols = asset_cols + indices

    for col in asset_cols:
        data = df[col]
        row = {
            "Asset": col,
            "Annualized Return": annualized_return(data),
            "Annualized Volatility": annualized_volatility(data),
            "Sharpe Ratio": sharpe_ratio(data, rf),
            "Max Drawdown": max_drawdown(data),
            "VaR 95%": value_at_risk(data),
            "Expected Shortfall 95%": expected_shortfall(data),
        }

        for name, bench in benchmarks.items():
            b = beta(data, bench)
            alpha = jensens_alpha(data, bench, b, rf)
            row[f"Beta ({name})"] = b
            row[f"Alpha ({name})"] = alpha

        results.append(row)

    return pd.DataFrame(results)

if __name__ == "__main__":

    out_dir = r"out/stats/"

    ports = ["S&P500", "S&P400", "S&P600"]
    for port in ports:
        df = get_data(indices=[port,])
        print(df)
        stats_df = compute_backtest_metrics(df, indices=[port,])
        stats_df.to_csv(os.path.join(out_dir, f"backtest_metrics_{port}.csv"), index=False)
