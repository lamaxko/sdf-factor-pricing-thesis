import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from test.factor_portfolio import FactorPortfolio
from apis.kfrench import KfApi
from apis.fred import FredApi
from helpers.db_manager import DB
from helpers.time_series import RetTimeSeries


def annualized_return(df):
    """
    Calculate annualized return for each portfolio.
    Assumes monthly returns.
    """
    total_months = df.shape[0]
    cumulative = (1 + df).prod()
    return cumulative ** (12 / total_months) - 1

def annualized_volatility(df):
    """
    Calculate annualized volatility for each portfolio.
    """
    return df.std() * (12 ** 0.5)

def max_drawdown(df):
    """
    Calculate max drawdown for each portfolio.
    Assumes df contains monthly returns.
    """
    cumulative = (1 + df).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()  # Most negative drawdown (worst)

def overall_volatility(df):
    """
    Calculate overall (total period) volatility for each column (portfolio).
    """
    df = df.copy()
    return df.std()

def yearly_volatility(df):
    """
    Calculate volatility (std) for each year for each column (portfolio).
    Returns a DataFrame with years as index and columns as in df.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['year'] = df.index.year
    return df.groupby('year').std()

def plot_double_volatility_heatmap(vol_df_top, vol_df_bottom, save_path=None):
    """
    Plot two vertically stacked heatmaps of yearly volatility (%) per portfolio (quantile),
    maintaining individual plot styles.
    """
    # Format data as percentage
    vol_df_top = (vol_df_top * 100).round(2)
    vol_df_bottom = (vol_df_bottom * 100).round(2)

    # Apply visual style
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "axes.labelweight": "bold",
        "axes.grid": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "axes.titleweight": "bold"
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

    for ax, data, title in zip(
        axes,
        [vol_df_top, vol_df_bottom],
        [
            "Annualized Volatility (%, Monthly Returns) of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
            "Annualized Volatility (%, Monthly Returns) of Beta-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe"
        ]
    ):
        sns.heatmap(
            data.T,  # Quantiles on x-axis, years on y-axis
            cmap="coolwarm",
            linewidths=0.4,
            linecolor='white',
            annot=True,
            fmt=".1f",
            cbar_kws={"shrink": 0.75, "format": '%.0f%%'},
            ax=ax
        )

        ax.set_title(title, fontsize=16, fontweight="bold", color="#0b3c5d", pad=14)
        ax.set_xlabel("Year", fontsize=12, weight="bold", color="#0b3c5d", labelpad=10)
        ax.set_ylabel("Quantile", fontsize=12, weight="bold", color="#0b3c5d", labelpad=10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax.tick_params(axis='y', labelrotation=0, labelsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Custom Palette
PALETTE_BASE = [
    "#0b3c5d",  # Deep Blue
    "#328cc1",  # Mid Blue
    "#6b0f1a",  # Deep Red
    "#c94c4c",  # Mid Red
    "#4c956c",  # Elegant Green
]

def apply_default_style():
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
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



def plot_double_cumulative_returns(df_top, df_bottom,
                                   recession_series=None,
                                   benchmarks_df=None,
                                   title_top="Cumulative Returns — Dataset 1",
                                   title_bottom="Cumulative Returns — Dataset 2",
                                   save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True)

    # --- Prepare Recession Periods ---
    recession_periods = []
    if recession_series is not None:
        recession_series = recession_series.reindex(df_top.index, method='pad').fillna(0).astype(int)
        mask = recession_series == 1
        in_recession = False
        for i in range(len(mask)):
            if mask.iloc[i] and not in_recession:
                start = mask.index[i]
                in_recession = True
            elif not mask.iloc[i] and in_recession:
                end = mask.index[i]
                recession_periods.append((start, end))
                in_recession = False
        if in_recession:
            recession_periods.append((start, mask.index[-1]))

    # --- Process Benchmarks ---
    if benchmarks_df is not None:
        # Align benchmark date range to match the portfolios
        start_date = min(df_top.index.min(), df_bottom.index.min())
        benchmarks_df = benchmarks_df[benchmarks_df.index >= start_date]
        # Compute cumulative returns starting at 0
        benchmarks_df = RetTimeSeries(benchmarks_df).cum_returns()

    # --- Determine shared y-axis limit ---
    ymax = max(df_top.max().max(), df_bottom.max().max())
    if benchmarks_df is not None:
        ymax = max(ymax, benchmarks_df.max().max())
    ymax *= 1.05  # Add 5% buffer

    # --- Plot each chart ---
    for df, ax, title in zip([df_top, df_bottom], axes, [title_top, title_bottom]):
        # Shade recessions
        for start, end in recession_periods:
            ax.axvspan(start, end, color='lightgrey', alpha=0.4, zorder=0)

        # Plot quantile lines
        for i, col in enumerate(df.columns):
            ax.plot(
                df.index,
                df[col],
                label=f"Quantile {col}",
                color=PALETTE_BASE[i % len(PALETTE_BASE)],
                lw=2,
                alpha=0.85,
                linestyle="-"
            )

        # Plot aligned benchmark lines
        colors = {
            'S&P 500': '#000000',     # black
            'S&P 400': '#666666',     # middle gray
            'S&P 600': '#bbbbbb'      # light gray
        }
        if benchmarks_df is not None:
            for bench_col in benchmarks_df.columns:
                if bench_col in df.columns:
                    continue
                bench_aligned = benchmarks_df[bench_col].reindex(df.index).dropna()
                ax.plot(
                    bench_aligned.index,
                    bench_aligned,
                    label=bench_col,
                    linestyle='--',
                    linewidth=1.8,
                    alpha=0.8,
                    color=colors.get(bench_col, 'black'),
                    zorder=5
                )

        # Apply consistent y-axis scaling
        ax.set_ylim(-2, ymax)

        # Styling
        ax.set_title(title, color="#0b3c5d", fontsize=16, weight="bold")
        ax.set_ylabel("Cumulative Return", color="#0b3c5d", weight="bold")
        ax.tick_params(pad=6)
        ax.legend(title="Quantiles / Benchmarks")
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    axes[-1].set_xlabel("Date", color="#0b3c5d", weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def get_recessions():

    fred = FredApi()
    recession = fred.recession_indicator()
    return recession

def benchmarks_ret():
    df = DB.fetch(f"""
        SELECT
          r.date,
          MAX(CASE WHEN sm.dscd = 'S&PMIDC' THEN r.return END) AS "S&P 400",
          MAX(CASE WHEN sm.dscd = 'S&PCOMP' THEN r.return END) AS "S&P 500",
          MAX(CASE WHEN sm.dscd = 'S&P600I' THEN r.return END) AS "S&P 600"
        FROM
          returns r
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

def get_mon_ret(name, indices):
    n_quantiles = 5
    dates = (date(2002, 1, 1), date(2022, 1, 1))
    portfolio = FactorPortfolio(name, indices, n_quantiles, dates)

    portfolio.sanity_check()

    cum_ret = portfolio.cum_returns
    cum_ret.index = pd.to_datetime(cum_ret.index)
    mon_ret = portfolio.monthly_returns
    mon_ret.index = pd.to_datetime(mon_ret.index)
    anu_ret = portfolio.annual_returns

    return mon_ret, cum_ret

def get_annual_ret(name, indices):
    n_quantiles = 5
    dates = (date(2002, 1, 1), date(2022, 1, 1))
    portfolio = FactorPortfolio(name, indices, n_quantiles, dates)

    portfolio.sanity_check()

    cum_ret = portfolio.cum_returns
    cum_ret.index = pd.to_datetime(cum_ret.index)
    mon_ret = portfolio.monthly_returns
    mon_ret.index = pd.to_datetime(mon_ret.index)
    anu_ret = portfolio.annual_returns
    print(anu_ret)
    print(anu_ret.index)


    return anu_ret
def plot_double_cumulative_returns(df_top, df_bottom,
                                   recession_series=None,
                                   benchmarks_df=None,
                                   title_top="Cumulative Returns — Dataset 1",
                                   title_bottom="Cumulative Returns — Dataset 2",
                                   save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), dpi=300, sharex=True)

    # --- Prepare Recession Periods ---
    recession_periods = []
    if recession_series is not None:
        recession_series = recession_series.reindex(df_top.index, method='pad').fillna(0).astype(int)
        mask = recession_series == 1
        in_recession = False
        for i in range(len(mask)):
            if mask.iloc[i] and not in_recession:
                start = mask.index[i]
                in_recession = True
            elif not mask.iloc[i] and in_recession:
                end = mask.index[i]
                recession_periods.append((start, end))
                in_recession = False
        if in_recession:
            recession_periods.append((start, mask.index[-1]))

    # --- Process Benchmarks ---
    if benchmarks_df is not None:
        # Align benchmark date range to match the portfolios
        start_date = min(df_top.index.min(), df_bottom.index.min())
        benchmarks_df = benchmarks_df[benchmarks_df.index >= start_date]
        # Compute cumulative returns starting at 0
        benchmarks_df = RetTimeSeries(benchmarks_df).cum_returns()

    # --- Determine shared y-axis limit ---
    ymax = max(df_top.max().max(), df_bottom.max().max())
    if benchmarks_df is not None:
        ymax = max(ymax, benchmarks_df.max().max())
    ymax *= 1.05  # Add 5% buffer

    # --- Plot each chart ---
    for df, ax, title in zip([df_top, df_bottom], axes, [title_top, title_bottom]):
        # Shade recessions
        for start, end in recession_periods:
            ax.axvspan(start, end, color='lightgrey', alpha=0.4, zorder=0)

        # Plot quantile lines
        for i, col in enumerate(df.columns):
            ax.plot(
                df.index,
                df[col],
                label=f"Quantile {col}",
                color=PALETTE_BASE[i % len(PALETTE_BASE)],
                lw=2,
                alpha=0.85,
                linestyle="-"
            )

        # Plot aligned benchmark lines
        colors = {
            'S&P 500': '#000000',     # black
            'S&P 400': '#666666',     # middle gray
            'S&P 600': '#bbbbbb'      # light gray
        }
        if benchmarks_df is not None:
            for bench_col in benchmarks_df.columns:
                if bench_col in df.columns:
                    continue
                bench_aligned = benchmarks_df[bench_col].reindex(df.index).dropna()
                ax.plot(
                    bench_aligned.index,
                    bench_aligned,
                    label=bench_col,
                    linestyle='--',
                    linewidth=1.8,
                    alpha=0.8,
                    color=colors.get(bench_col, 'black'),
                    zorder=5
                )

        # Apply consistent y-axis scaling
        ax.set_ylim(-2, ymax)

        # Styling
        ax.set_title(title, color="#0b3c5d", fontsize=16, weight="bold")
        ax.set_ylabel("Cumulative Return", color="#0b3c5d", weight="bold")
        ax.tick_params(pad=6)
        ax.legend(title="Quantiles / Benchmarks")
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    axes[-1].set_xlabel("Date", color="#0b3c5d", weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_grid_cumulative_returns(name_beta, name_omega, universes=["S&P500", "S&P400", "S&P600"], save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(3, 2, figsize=(16, 16), dpi=300, sharex=False)


    # Map display names to column names in benchmark df
    universe_column_map = {
        "S&P500": "S&P 500",
        "S&P400": "S&P 400",
        "S&P600": "S&P 600"
    }

    # Colors for benchmarks
    benchmark_colors = {
        "S&P500": "#000000",   # black
        "S&P400": "#666666",   # middle gray
        "S&P600": "#bbbbbb"    # light gray
    }

    # Get cumulative benchmark returns once
    benchmark_raw = benchmarks_ret()
    benchmark_raw = benchmark_raw[benchmark_raw.index >= "2002-01-01"]
    benchmark_cum = RetTimeSeries(benchmark_raw).cum_returns()

    for row_idx, universe in enumerate(universes):
        display_name = universe                # e.g., 'S&P500'
        column_name = universe_column_map[universe]  # e.g., 'S&P 500'

        # Fetch beta and omega portfolios for this universe only
        _, cum_ret_beta = get_mon_ret(name_beta, [display_name])
        _, cum_ret_omega = get_mon_ret(name_omega, [display_name])

        df_beta = cum_ret_beta
        df_omega = cum_ret_omega


        # Get matching benchmark return series
        benchmark_series = None
        if column_name in benchmark_cum.columns:
            benchmark_series = benchmark_cum[column_name].reindex(df_beta.index).dropna()

        # --- Shared y-scaling for this row ---
        row_data = [df_beta, df_omega]
        if benchmark_series is not None:
            row_data.append(benchmark_series.to_frame())

        row_min = min(df.min().min() for df in row_data)
        row_max = max(df.max().max() for df in row_data)
        ymin = row_min * 1.05 
        ymax = row_max * 1.05

        # --- Beta Column ---
        ax_beta = axes[row_idx, 0]
        for i, col in enumerate(df_beta.columns):
            ax_beta.plot(df_beta.index, df_beta[col], label=f"Q{col}",
                         color=PALETTE_BASE[i % len(PALETTE_BASE)], lw=2)

        if benchmark_series is not None:
            ax_beta.plot(benchmark_series.index, benchmark_series,
                         label=column_name,
                         color=benchmark_colors[display_name],
                         linestyle='--', linewidth=2, alpha=0.9, zorder=5)

        ax_beta.set_title(f"{display_name} — Beta Sorted", fontsize=13, weight="bold", color="#0b3c5d")
        ax_beta.set_ylim(ymin, ymax)
        ax_beta.grid(True, linestyle="--", alpha=0.3)
        ax_beta.legend(fontsize=8)
        ax_beta.spines['left'].set_alpha(0.3)
        ax_beta.spines['bottom'].set_alpha(0.3)

        # --- Omega Column ---
        ax_omega = axes[row_idx, 1]
        for i, col in enumerate(df_omega.columns):
            ax_omega.plot(df_omega.index, df_omega[col], label=f"Q{col}",
                          color=PALETTE_BASE[i % len(PALETTE_BASE)], lw=2)

        if benchmark_series is not None:
            ax_omega.plot(benchmark_series.index, benchmark_series,
                          label=column_name,
                          color=benchmark_colors[display_name],
                          linestyle='--', linewidth=2, alpha=0.9, zorder=5)

        ax_omega.set_title(f"{display_name} — Omega Sorted", fontsize=13, weight="bold", color="#0b3c5d")
        ax_omega.set_ylim(ymin, ymax)
        ax_omega.grid(True, linestyle="--", alpha=0.3)
        ax_omega.legend(fontsize=8)
        ax_omega.spines['left'].set_alpha(0.3)
        ax_omega.spines['bottom'].set_alpha(0.3)

    # Shared axis labels
    for ax in axes[-1]:
        ax.set_xlabel("Date", fontsize=11, color="#0b3c5d", weight="bold")
    for col in range(2):
        for row in range(3):
            axes[row, col].set_ylabel("Cumulative Return", fontsize=11, color="#0b3c5d", weight="bold")

    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression

def plot_double_geo_mean_returns(df_top, df_bottom,
                                  title_top="Geometric Mean — Dataset 1",
                                  title_bottom="Geometric Mean — Dataset 2",
                                  show_trend=False,
                                  save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=False)

    for df, ax, title in zip([df_top, df_bottom], axes, [title_top, title_bottom]):
        # Compute geometric means
        geo_means = {}
        for col in df.columns:
            vals = df[col].dropna() + 1
            vals = vals[vals > 0]
            geo_means[int(col)] = np.prod(vals) ** (1 / len(vals)) - 1 if not vals.empty else np.nan

        quantiles = sorted(geo_means.keys())
        geo_df = pd.DataFrame({
            "Quantile": [f"Q{q}" for q in quantiles],
            "GeometricMean": [geo_means[q] for q in quantiles]
        })

        # Use PALETTE_BASE for coloring by quantile
        palette = {f"Q{q}": PALETTE_BASE[i % len(PALETTE_BASE)] for i, q in enumerate(quantiles)}

        bars = sns.barplot(
            data=geo_df,
            x="Quantile",
            y="GeometricMean",
            palette=palette,
            edgecolor="white",
            linewidth=1.5,
            ax=ax,
            zorder=3
        )

        for bar in bars.patches:
            bar.set_width(bar.get_width() * 0.6)
            height = bar.get_height()
            if pd.notnull(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2%}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    zorder=4
                )

        if show_trend:
            x_vals = [bar.get_x() + bar.get_width() / 2 for bar in bars.patches]
            y_vals = [bar.get_height() for bar in bars.patches]
            model = LinearRegression().fit(np.array(x_vals).reshape(-1, 1), y_vals)
            x_fit = np.array([min(x_vals), max(x_vals)]).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            ax.plot(
                x_fit.flatten(),
                y_fit,
                color="#666666",
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                zorder=2
            )

        ax.set_title(title, color="#0b3c5d", fontsize=16, weight="bold")
        ax.set_xlabel("Quantile", color="#0b3c5d", weight="bold")
        ax.set_ylabel("Geometric Mean Return", color="#0b3c5d", weight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
        ax.tick_params(pad=6)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression
import os


def plot_grid_geo_mean_returns(name_beta, name_omega, universes=["S&P500", "S&P400", "S&P600"], save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=300, sharex=False)

    for row_idx, universe in enumerate(universes):
        # --- Fetch monthly returns ---
        mon_ret_beta, _ = get_mon_ret(name_beta, [universe])
        mon_ret_omega, _ = get_mon_ret(name_omega, [universe])
        df_beta = mon_ret_beta
        df_omega = mon_ret_omega

        # --- Compute geometric means ---
        def compute_geo_means(df):
            geo_means = {}
            for col in df.columns:
                vals = df[col].dropna() + 1
                vals = vals[vals > 0]
                if not vals.empty:
                    geo_means[int(col)] = np.prod(vals) ** (1 / len(vals)) - 1
                else:
                    geo_means[int(col)] = np.nan
            return geo_means

        geo_means_beta = compute_geo_means(df_beta)
        geo_means_omega = compute_geo_means(df_omega)

        # --- Row-wide y-limits ---
        all_vals = list(geo_means_beta.values()) + list(geo_means_omega.values())
        ymax = max(all_vals) * 1.05 if all_vals else 0.1
        ymin = min(all_vals) * 1.05 if min(all_vals) < 0 else 0

        # --- Build DataFrames ---
        def build_geo_df(geo_dict):
            return pd.DataFrame({
                "Quantile": [f"Q{k}" for k in sorted(geo_dict)],
                "GeoMean": [geo_dict[k] for k in sorted(geo_dict)]
            })

        beta_df = build_geo_df(geo_means_beta)
        omega_df = build_geo_df(geo_means_omega)

        # --- Beta plot ---
        ax_beta = axes[row_idx, 0]
        bars = sns.barplot(
            data=beta_df,
            x="Quantile",
            y="GeoMean",
            palette=PALETTE_BASE[:len(beta_df)],
            edgecolor="white",
            linewidth=1.5,
            ax=ax_beta,
            zorder=3
        )
        for bar in bars.patches:
            bar.set_width(bar.get_width() * 0.6)
            height = bar.get_height()
            if pd.notnull(height):
                ax_beta.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2%}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    zorder=4
                )
        ax_beta.set_title(f"{universe} — Beta Sorted", fontsize=13, weight="bold", color="#0b3c5d")
        ax_beta.set_ylabel("Geo. Mean Return", fontsize=11, color="#0b3c5d", weight="bold")
        ax_beta.set_ylim(ymin, ymax)
        ax_beta.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
        ax_beta.grid(True, linestyle="--", alpha=0.3)
        ax_beta.spines['left'].set_alpha(0.3)
        ax_beta.spines['bottom'].set_alpha(0.3)

        # --- Omega plot ---
        ax_omega = axes[row_idx, 1]
        bars = sns.barplot(
            data=omega_df,
            x="Quantile",
            y="GeoMean",
            palette=PALETTE_BASE[:len(omega_df)],
            edgecolor="white",
            linewidth=1.5,
            ax=ax_omega,
            zorder=3
        )
        for bar in bars.patches:
            bar.set_width(bar.get_width() * 0.6)
            height = bar.get_height()
            if pd.notnull(height):
                ax_omega.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2%}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    zorder=4
                )
        ax_omega.set_title(f"{universe} — Omega Sorted", fontsize=13, weight="bold", color="#0b3c5d")
        ax_omega.set_ylabel("Geo. Mean Return", fontsize=11, color="#0b3c5d", weight="bold")
        ax_omega.set_ylim(ymin, ymax)
        ax_omega.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
        ax_omega.grid(True, linestyle="--", alpha=0.3)
        ax_omega.spines['left'].set_alpha(0.3)
        ax_omega.spines['bottom'].set_alpha(0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Quantile", fontsize=11, color="#0b3c5d", weight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_double_annual_returns(df_top, df_bottom,
                                title_top="Annual Returns — Beta Sorted",
                                title_bottom="Annual Returns — Omega Sorted",
                                show_labels=True,
                                save_path=None):
    apply_default_style()
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=300, sharex=True)

    # --- Determine shared y-limits across both charts ---
    combined = pd.concat([df_top, df_bottom])
    ymin = combined.min().min() * 1.05 if combined.min().min() < 0 else 0
    ymax = combined.max().max() * 1.05

    for df, ax, title in zip([df_top, df_bottom], axes, [title_top, title_bottom]):
        df_plot = df.copy()
        df_plot.index = df_plot.index.astype(str)
        df_plot = df_plot.reset_index().melt(id_vars="date", var_name="Quantile", value_name="Return")

        barplot = sns.barplot(
            data=df_plot,
            x="date",
            y="Return",
            hue="Quantile",
            palette=PALETTE_BASE[:df.shape[1]],
            ax=ax,
            edgecolor="white",
            linewidth=1.5,
            zorder=3
        )

        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, color="#0b3c5d", fontsize=16, weight="bold")
        ax.set_xlabel("Year", color="#0b3c5d", weight="bold")
        ax.set_ylabel("Return", color="#0b3c5d", weight="bold")
        ax.tick_params(axis='x', labelrotation=45, pad=6)
        ax.tick_params(axis='y', pad=6)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        # Selective labels only for Quantile 1 and 5
        if show_labels:
            for bar in barplot.patches:
                quantile = bar.get_label() if hasattr(bar, "get_label") else None
                label = barplot.get_legend_handles_labels()[1]
                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                hue = bar.get_label()
                try:
                    hue_val = int(label[int(bar.get_hatch() or 0)])  # fallback for order
                except:
                    hue_val = None

                if not pd.isna(height):
                    # Get quantile number from the bar
                    bar_index = int(barplot.patches.index(bar)) % df.shape[1]
                    quant_label = df_plot["Quantile"].unique()[bar_index]
                    if quant_label in ['1', '5']:
                        ax.text(
                            x,
                            height + 0.005,
                            f"{height:.0%}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            zorder=4
                        )

        ax.legend(title="Quantiles")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        
if __name__ =="__main__":

    name_omega = "RollingOmega12M_lag1_neg"
    name_beta = "RollingOmBeta12M_lag1_neg"
    save_dir = r"out/"

    an_ret_beta = get_annual_ret(name_beta, ["S&P500", "S&P400", "S&P600"])
    an_ret_omega = get_annual_ret(name_omega, ["S&P500", "S&P400", "S&P600"])

    plot_double_annual_returns(
        df_top=an_ret_beta,
        df_bottom=an_ret_omega,
        title_top="Annual Returns (%) of Beta-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
        title_bottom="Annual Returns (%) of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
        save_path=r"out/plots/double_annual_returns.png"
    )

    # plot_grid_cumulative_returns(name_beta, name_omega, save_path=fr"{save_dir}plots/grid_return_plot.png")
    plot_grid_geo_mean_returns(name_beta, name_omega, save_path=fr"{save_dir}plots/grid_geo_mean_return_plot.png")

    recession = get_recessions()
    print(recession)

    benchmarks = benchmarks_ret()

    mon_ret_beta_all, cum_ret_beta_all = get_mon_ret(name_beta, ["S&P500", "S&P400", "S&P600"])
    mon_ret_omega_all, cum_ret_omega_all = get_mon_ret(name_omega, ["S&P500", "S&P400", "S&P600"])
    print(cum_ret_omega_all)

    plot_double_geo_mean_returns(df_top=mon_ret_beta_all, df_bottom=mon_ret_omega_all,
                             title_top="Geo Mean (%, Monthly Returns) of Beta-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
                             title_bottom="Geo Mean (%, Monthly Returns) of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
                             save_path="out/plots/double_geo_means.png", show_trend=True)



    plot_double_volatility_heatmap(yearly_volatility(mon_ret_omega_all), yearly_volatility(mon_ret_beta_all), save_path=r"out/plots/double_yearly_volatility_heatmap.png")
    plot_double_cumulative_returns(cum_ret_omega_all, cum_ret_beta_all, title_top="Cumulative Returns (%, Monthly Returns) of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
                                   title_bottom="Cumulative Returns (%, Monthly Returns) of Beta-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe",
                                   save_path=r"out/plots/double_cum_return.png", recession_series=recession,
                                   benchmarks_df=benchmarks,)

    mon_ret_omega_500 = get_mon_ret(name_omega, ["S&P500",])
    mon_ret_omega_400 = get_mon_ret(name_omega, ["S&P400",])
    mon_ret_omega_600 = get_mon_ret(name_omega, ["S&P600",])


    mon_ret_beta_500 = get_mon_ret(name_beta, ["S&P500",])
    mon_ret_beta_400 = get_mon_ret(name_beta, ["S&P400",])
    mon_ret_beta_600 = get_mon_ret(name_beta, ["S&P600",])

    mon_rf = KfApi().clean_data_ff5()["RF"]
    print(mon_rf)

# Overall volatility
    vola_all = overall_volatility(mon_ret)
    print("Overall Volatility:\n", vola_all)

# Yearly volatility
    vola_year = yearly_volatility(mon_ret)
    print("\nYearly Volatility:\n", vola_year)
# plot_yearly_volatility_heatmap(vola_year, save_path=r"out/plots/pyearly_volatility_heatmap.png")

# Calculate yearly metrics
    vola_year = annualized_volatility(mon_ret)
    ret_year = annualized_return(mon_ret)
    dd_year = max_drawdown(mon_ret)

# Print results
    print("\nYearly Annualized Volatility:\n", vola_year)
    print("\nYearly Annualized Return:\n", ret_year)
    print("\nYearly Max Drawdown:\n", dd_year)
