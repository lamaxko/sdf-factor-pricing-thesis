import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
import matplotlib.pyplot as plt

from test.factor_portfolio import FactorPortfolio
from apis.kfrench import KfApi
from apis.fred import FredApi
from helpers.db_manager import DB
from helpers.time_series import RetTimeSeries
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# Colors
COLOR_SCATTER = "#328cc1"  # Blue
COLOR_LINE = "#c94c4c"     # Red

def apply_default_style():
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "axes.labelweight": "bold",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "axes.titleweight": "bold"
    })


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

def plot_ff_regression_grid(df, portfolio_cols=[1, 2, 3, 4, 5], factor_cols=["Mkt-RF", "SMB", "HML", "RMW", "CMA"], save_path=None):
    sns.set_style("whitegrid")

    # --- Styling ---
    def apply_default_style():
        plt.style.use("default")
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.labelweight": "bold",
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.titleweight": "bold"
        })

    apply_default_style()

    # --- Figure ---
    fig, axes = plt.subplots(len(portfolio_cols), len(factor_cols), figsize=(18, 14), sharex=False, sharey=False)

    BLUE = "#328cc1"
    RED = "#c94c4c"

    for row_idx, port in enumerate(portfolio_cols):
        for col_idx, factor in enumerate(factor_cols):
            ax = axes[row_idx, col_idx]
            x = df[factor].values.reshape(-1, 1)
            y = df[port].values

            # Scatterplot
            ax.scatter(x, y, color=BLUE, s=10, alpha=0.6)

            # Regression line
            model = LinearRegression().fit(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            ax.plot(x_fit, y_fit, color=RED, lw=2)

            # Show factor name only on top row
            if row_idx == 0:
                ax.set_title(factor, fontsize=11, color="#0b3c5d", weight="bold")
            
            # Show quantile label only on left column
            if col_idx == 0:
                ax.set_ylabel(f"Q{port}", fontsize=10, color="#0b3c5d", weight="bold")

            if row_idx < len(portfolio_cols) - 1:
                ax.set_xticklabels([])

    # --- Title ---
    fig.suptitle("Fama-French Factor Regressions by Omega Sorted Quantile (Excess Monthly Returns) (2002-2022), S&P 500/400/600", fontsize=16, weight="bold", color="#0b3c5d", y=1.02)
    plt.tight_layout()

    # --- Save or Show ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def save_ff_results_to_csv(reg_results, save_path):
    """
    Extracts R², alphas, betas, and their p-values for each quantile from regression results
    and saves to a CSV file.
    """
    rows = []
    for q, res in reg_results.items():
        row = {
            "Quantile": q,
            "R_squared": res.rsquared,
            "Alpha": res.params["const"],
            "Alpha_pval": res.pvalues["const"]
        }

        for factor in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
            row[f"{factor}"] = res.params[factor]
            row[f"{factor}_pval"] = res.pvalues[factor]

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = df_out.set_index("Quantile")
    df_out.to_csv(save_path)
    print(f"[Saved regression summary to] {save_path}")

def run_ff_regressions(df, portfolio_cols=[1, 2, 3, 4, 5], factor_cols=["Mkt-RF", "SMB", "HML", "RMW", "CMA"]):
    results = {}

    X = df[factor_cols]
    X = sm.add_constant(X)  # adds intercept (alpha)

    for port in portfolio_cols:
        y = df[port]
        model = sm.OLS(y, X, missing="drop").fit()
        results[port] = model

    return results

def get_mon_ret(name, indices):
    n_quantiles = 5
    dates = (date(2002, 1, 1), date(2022, 1, 1))
    portfolio = FactorPortfolio(name, indices, n_quantiles, dates)
    mon_ret = portfolio.monthly_returns
    mon_ret.index = pd.to_datetime(mon_ret.index)

    return mon_ret

def get_benchmarks_ret():
    df = DB.fetch(f"""
        SELECT
          r.date,
          MAX(CASE WHEN sm.dscd = 'S&PMIDC' THEN r.return END) AS "S&P 400",
          MAX(CASE WHEN sm.dscd = 'S&PCOMP' THEN r.return END) AS "S&P 500",
          MAX(CASE WHEN sm.dscd = 'S&P600I' THEN r.return END) AS "S&P 600"
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
    df = df / 100

    # df = RetTimeSeries(df).cum_returns()
    print(df)
    return df 

def get_recessions():

    fred = FredApi()
    recession = fred.recession_indicator()
    return recession

if __name__ == "__main__":

    name_omeg = "RollingOmega12M_lag1_neg"
    name_beta = "RollingOmBeta12M_lag1_neg"
    save_dir = r"out/ff/"

    ff = KfApi().clean_data_ff5() / 100
    bench = get_benchmarks_ret()

    returns = get_mon_ret(name_omeg, ["S&P500", "S&P400", "S&P600"])

    returns.index = pd.to_datetime(returns.index)
    ff.index = pd.to_datetime(ff.index)
    merged = returns.join(ff, how='left')  # this keeps returns's index
    merged = merged.dropna()

    for i in range(1,5+1):
        merged[i] = merged[i] - merged["RF"]
    merged.drop(columns=['RF'], inplace=True)
    print(merged)

    reg_res = run_ff_regressions(merged)
    print(reg_res)
    for q, res in reg_res.items():
        print(f"--------SUMMARY QUANTILE {q}--------")
        print(res.summary())

    save_ff_results_to_csv(reg_res, save_path=f"{save_dir}ff_summary.csv")

    plot_ff_regression_grid(
        df=merged,
        save_path=f"{save_dir}styled_ff_regression_grid.png"
    )
