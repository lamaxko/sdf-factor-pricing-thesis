import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apis.fred import FredApi
from helpers.db_manager import DB   

import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_predictive_regression_new(df, target: str, predictor: str):
    """
    Run OLS regression with a lagged predictor already present in the DataFrame.
    """
    df_clean = df[[target, predictor]].dropna()
    X = sm.add_constant(df_clean[predictor])
    y = df_clean[target]
    model = sm.OLS(y, X).fit()
    return model

def extract_regression_stats(model):
    """
    Extract alpha, beta, t-statistic, p-value, and R-squared from OLS result.
    """
    predictor_name = model.params.index[1]
    return {
        "alpha": model.params["const"],
        "beta": model.params[predictor_name],
        "t_stat_beta": model.tvalues[predictor_name],
        "p_value_beta": model.pvalues[predictor_name],
        "r_squared": model.rsquared
    }

def compute_dispersion_metrics(fitted_values: pd.Series):
    """
    Compute sigma (std) and coefficient of variation of predicted excess returns.
    """
    mean_pred = np.mean(fitted_values)
    sigma_pred = np.std(fitted_values)
    coef_of_var = sigma_pred / mean_pred if mean_pred != 0 else np.nan
    return {
        "sigma_pred": sigma_pred,
        "coef_of_variation": coef_of_var
    }

def plot_single_series(series, name="M", recession_series=None, save_path=None):
    """
    Plot a single time series with shaded recession periods and a horizontal red average line.
    """
    plt.style.use("default")
    COLOR_MAIN = "#328cc1"     # Blue for main line
    COLOR_AVG = "#c94c4c"      # Soft red for average line
    LABEL_COLOR = "#0b3c5d"
    MARKER = 'o'

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)

    # --- Recession shading ---
    if recession_series is not None:
        recession_series = recession_series.loc[series.index.min():series.index.max()]
        recession_series = recession_series.reindex(series.index, method='pad').fillna(0).astype(int)

        in_rec = False
        for i in range(len(recession_series)):
            if recession_series.iloc[i] == 1 and not in_rec:
                start = recession_series.index[i]
                in_rec = True
            elif recession_series.iloc[i] == 0 and in_rec:
                end = recession_series.index[i]
                ax.axvspan(start, end, color='lightgrey', alpha=0.4, zorder=0)
                in_rec = False
        if in_rec:
            ax.axvspan(start, series.index[-1], color='lightgrey', alpha=0.4, zorder=0)

    # --- Main line and markers ---
    x = series.index
    y = series.values
    scatter_idx = [i for i, date in enumerate(x) if date.month == 1]

    ax.plot(x, y, color=COLOR_MAIN, linewidth=2, linestyle="-", label=name)
    ax.scatter(x[scatter_idx], y[scatter_idx], marker=MARKER, s=40,
               facecolors='white', edgecolors=COLOR_MAIN, linewidths=1.2, zorder=3)

    # --- Average line ---
    avg = series.mean()
    ax.axhline(avg, color=COLOR_AVG, linestyle='--', linewidth=2, alpha=0.85, label="Average")

    ax.text(x[1], avg, f"{avg:.2f}", color=COLOR_AVG, fontsize=11, va='bottom', ha='right', weight='bold')

    # --- Styling ---
    ax.set_title(f"Time Series of {name} (SDF) (2000–2022)", fontsize=16, fontweight='bold', color=LABEL_COLOR, pad=10)
    ax.set_ylabel(name, fontsize=12, weight="bold", color=LABEL_COLOR)
    ax.set_xlabel("Date", fontsize=12, weight="bold", color=LABEL_COLOR)
    ax.tick_params(axis='x', labelsize=10, pad=6)
    ax.tick_params(axis='y', labelsize=10, pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

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
    # df = df / 100

    # df = RetTimeSeries(df).cum_returns()
    print(df)
    return df 

def merge_ret(ret, feature):

    ret.index = pd.to_datetime(ret.index)
    feature.index = pd.to_datetime(feature.index)
    merged = ret.join(feature, how='left')
    merged = merged.dropna()

    return merged

import pandas as pd
import statsmodels.api as sm

def run_predictive_regression(df, target="S&P 500", predictor="M", lags=[0, 1, 12, 60]):
    """
    Run predictive regressions of target index on lagged M values.

    Parameters:
    - df: DataFrame with date-indexed columns, including target and predictor
    - target: str, name of dependent variable (e.g., "S&P 500")
    - predictor: str, name of independent variable (e.g., "M")
    - lags: list of ints, lag lengths in months

    Returns:
    - Dictionary with lag label as key and regression result summary
    """
    results = {}

    for lag in lags:
        df_lagged = df[[target, predictor]].copy()
        df_lagged[f"{predictor}_lag{lag}"] = df_lagged[predictor].shift(lag)
        df_lagged = df_lagged.dropna()

        X = sm.add_constant(df_lagged[f"{predictor}_lag{lag}"])
        y = df_lagged[target]

        model = sm.OLS(y, X).fit()
        results[f"Lag {lag}"] = model

        print(f"\n--- Predictive Regression: {target} ~ {predictor} (Lag {lag} months) ---")
        print(model.summary())

    return results

path = r"C:\Users\lasse.kock\Desktop\msc_thesis\src\out\panel\panel_2000_2022.csv"
df = pd.read_csv(path)
print(df)
df_omega = df.pivot(index="date", columns="id_stock", values="omega")
df_return = df.pivot(index="date", columns="id_stock", values="ExcessRet")
print(df_omega)
print(df_return)


result = (df_omega.shift(1) * df_return).sum(axis=1)
print(result)
result.index = pd.to_datetime(result.index)
result.to_csv(r'out/pred_reg/M.csv')


fred = FredApi()
recession = fred.recession_indicator()

# plot_single_series(result, recession_series=recession, save_path="out/pred_reg/sdf_plot.png")

benchmarks = get_benchmarks_ret()

result = result.rename("M")
merged = merge_ret(pd.DataFrame(result), benchmarks)
print(merged)

ew_port = df_return.mean(axis=1)
ew_port = ew_port.rename("EW")
merged = merge_ret(merged, ew_port)


ew_omega = df_omega.mean(axis=1)
ew_omega = ew_omega.rename("Omega")
merged = merge_ret(merged, ew_omega)


merged["M_standardized"] = (merged["M"] - merged["M"].mean()) / merged["M"].std()
merged["Omega_standardized"] = (merged["Omega"] - merged["Omega"].mean()) / merged["Omega"].std()
reg_res = run_predictive_regression(merged, target="EW", predictor="Omega_standardized")
reg_res = run_predictive_regression(merged, target="EW", predictor="M_standardized")


# Create lagged predictors
merged["M_standardized_lag0"] = merged["Omega_standardized"]
merged["M_standardized_lag1"] = merged["Omega_standardized"].shift(1)
merged["M_standardized_lag12"] = merged["Omega_standardized"].shift(12)
merged["M_standardized_lag60"] = merged["Omega_standardized"].shift(60)

# Run models
results = {
    "no_lag": run_predictive_regression_new(merged, "S&P 400", "M_standardized_lag0"),
    "1m_lag": run_predictive_regression_new(merged, "S&P 400", "M_standardized_lag1"),
    "1y_lag": run_predictive_regression_new(merged, "S&P 400", "M_standardized_lag12"),
    "5y_lag": run_predictive_regression_new(merged, "S&P 400", "M_standardized_lag60"),
}


# Collect outputs
metrics = {}
for name, model in results.items():
    stats = extract_regression_stats(model)
    dispersion = compute_dispersion_metrics(model.fittedvalues)
    metrics[name] = {**stats, **dispersion}

# Convert to DataFrame for inspection/export
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)
metrics_df.to_csv("out/pred_reg/predictive_regression_metrics.csv")
