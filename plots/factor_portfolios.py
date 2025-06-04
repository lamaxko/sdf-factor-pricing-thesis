import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression

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


def plot_monthly_geometric_means(df, save_path=None, show_trend=False):
    apply_default_style()
    geo_means = {}
    for col in df.columns:
        vals = df[col].dropna() + 1
        vals = vals[vals > 0]
        geo_means[int(col)] = np.prod(vals) ** (1 / len(vals)) - 1 if not vals.empty else np.nan

    geo_df = pd.DataFrame({
        "Quantile": sorted(geo_means.keys()),
        "GeometricMean": [geo_means[k] for k in sorted(geo_means.keys())]
    })
    geo_df["Quantile"] = geo_df["Quantile"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(
        data=geo_df,
        x="Quantile",
        y="GeometricMean",
        palette=["#0b3c5d"] * len(geo_df),  # Deep Blue
        edgecolor="white",
        linewidth=1.5,
        ax=ax,
        zorder=3
    )

    # Uniform bar width + consistently placed labels
    for bar in bars.patches:
        bar.set_width(bar.get_width() * 0.6)  # thinner bars
        height = bar.get_height()
        if pd.notnull(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,  # consistently just above bar
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
            color="#4c956c",  # Elegant Green
            linestyle='--',
            linewidth=2,
            alpha=0.9,
            zorder=2
        )

    ax.set_title("Geometric Mean Monthly Returns by Quantile", color="#0b3c5d", fontsize=16, weight="bold")
    ax.set_xlabel("Quantile", color="#0b3c5d", weight="bold")
    ax.set_ylabel("Geometric Mean Return (%)", color="#0b3c5d", weight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.tick_params(pad=6)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Factor Premium Plot with Proper External Labels
def plot_factor_premium(series, freq="m", save_path=None):
    apply_default_style()
    series = series.copy()
    series.index = series.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = sns.barplot(
        x=series.index,
        y=series.values,
        color="#0b3c5d",  # Deep Blue
        edgecolor="white",
        linewidth=1.5,
        ax=ax,
        zorder=3
    )

    # External labels accounting for negatives
    for bar in bars.patches:
        height = bar.get_height()
        label_y = height + 0.002 if height >= 0 else height - 0.004
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.2%}",
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=9,
            zorder=4
        )

    ax.set_title(f"Factor Premium ({freq.upper()})", color="#0b3c5d", fontsize=16, weight="bold")
    ax.set_xlabel("Period", color="#0b3c5d", weight="bold")
    ax.set_ylabel("Premium", color="#0b3c5d", weight="bold")
    ax.tick_params(axis='x', rotation=45, pad=6)
    ax.tick_params(axis='y', pad=6)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# 1. Cumulative Returns Plot
def plot_cumulative_returns(df, save_path=None):
    apply_default_style()
    fig, ax = plt.subplots(figsize=(14, 6))

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

    ax.set_title("Cumulative Returns by Quantile", color="#0b3c5d", fontsize=16, weight="bold")
    ax.set_xlabel("Date", color="#0b3c5d", weight="bold")
    ax.set_ylabel("Cumulative Return", color="#0b3c5d", weight="bold")
    ax.tick_params(pad=6)
    ax.legend(title="Quantiles")
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# 2. Annual Returns Bar Chart
def plot_annual_returns(df, save_path=None, show_labels=True):
    apply_default_style()
    df_plot = df.copy()
    df_plot.index = df_plot.index.astype(str)
    df_plot = df_plot.reset_index().melt(id_vars="date", var_name="Quantile", value_name="Return")

    fig, ax = plt.subplots(figsize=(16, 7))
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

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Annual Returns by Quantile", color="#0b3c5d", fontsize=16, weight="bold")
    ax.set_xlabel("Year", color="#0b3c5d", weight="bold")
    ax.set_ylabel("Return", color="#0b3c5d", weight="bold")
    ax.tick_params(axis='x', labelrotation=45, pad=6)
    ax.tick_params(axis='y', pad=6)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    if show_labels:
        for bar in barplot.patches:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.002,
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    zorder=4
                )

    ax.legend(title="Quantiles")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_industry_composition(icbin_dict, save_path=None):
    """
    Plots stacked industry composition per quantile over years.

    Parameters:
    -----------
    icbin_dict : dict
        Dictionary with quantile keys and DataFrame values.
        Each DataFrame should have:
            - index: stock_id
            - columns: years
            - values: industry codes or names

    save_path : str or None
        If provided, saves the plot. Otherwise, displays it.
    """
    apply_default_style()
    num_quantiles = len(icbin_dict)
    fig, axes = plt.subplots(num_quantiles, 1, figsize=(16, 3.4 * num_quantiles), sharex=True)

    if num_quantiles == 1:
        axes = [axes]

    # Collect unique industries for consistent coloring
    all_industries = pd.concat(icbin_dict.values(), axis=0).stack().unique()
    palette = sns.color_palette(PALETTE_BASE + sns.color_palette("tab20"))
    color_map = {industry: palette[i % len(palette)] for i, industry in enumerate(sorted(all_industries))}

    # For final shared legend
    legend_handles = []
    legend_labels = []

    for i, (quantile, df) in enumerate(icbin_dict.items()):
        melted = df.reset_index().melt(id_vars='id_stock', var_name='Year', value_name='Industry')

        count_df = (
            melted.groupby(['Year', 'Industry'])
            .size()
            .unstack(fill_value=0)
            .apply(lambda x: x / x.sum(), axis=1)
        )

        count_df = count_df[sorted(count_df.columns)]  # consistent industry order
        ax = axes[i]

        count_df.sort_index().plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[color_map[ind] for ind in count_df.columns],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            legend=False  # << REMOVE subplot legend
        )

        ax.set_ylabel(f"Quantile {quantile}", fontsize=12, weight="bold", color="#0b3c5d")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.tick_params(axis='y', pad=6)
        ax.set_title(f"Industry Composition - Quantile {quantile}",
                     fontsize=13, weight='bold', color="#0b3c5d", pad=10)

        if i == num_quantiles - 1:
            legend_handles = [
                plt.Line2D([0], [0], color=color_map[ind], lw=6)
                for ind in count_df.columns
            ]
            legend_labels = list(count_df.columns)

    axes[-1].set_xlabel("Year", fontsize=12, labelpad=6, color="#0b3c5d")

    # Shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=6,
        fontsize=10,
        title="Industry",
        title_fontsize=11
    )

    fig.suptitle("Industry Composition by Quantile of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe", fontsize=16, weight='bold', color="#0b3c5d", y=0.94)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_market_cap_distribution(mc_df, save_path=None):
    """
    Plots stacked market cap decile distribution per quantile over years.
    Each subplot is one quantile. Bars show proportion of stocks in each market cap decile.

    Parameters:
    -----------
    mc_df : DataFrame
        Columns: ['year', 'quantile', 'id_stock', 'market_cap']

    save_path : str or None
        If provided, saves the plot. Otherwise, displays it.
    """
    apply_default_style()

    # Ensure fresh DataFrame to avoid SettingWithCopyWarning
    mc_df = mc_df.dropna(subset=['market_cap']).copy()

    # Assign deciles per year
    mc_df['decile'] = mc_df.groupby('year')['market_cap'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=False)
    ).astype(int)

    num_quantiles = mc_df['quantile'].nunique()
    quantile_order = sorted(mc_df['quantile'].unique())
    fig, axes = plt.subplots(num_quantiles, 1, figsize=(16, 3.4 * num_quantiles), sharex=True)

    if num_quantiles == 1:
        axes = [axes]

    # Define decile labels and custom stepped green gradient
    decile_labels = list(range(10))
    green_gradient = [
        "#a8e6a2", "#8ed993", "#75cc83", "#5dbf74", "#4cb667",
        "#3cae5a", "#2c984d", "#1e8241", "#116d36", "#065a2c"
    ]
    color_map = {d: green_gradient[d] for d in decile_labels}

    for i, quantile in enumerate(quantile_order):
        ax = axes[i]
        df_q = mc_df[mc_df['quantile'] == quantile]

        # Count per decile per year
        count_df = (
            df_q.groupby(['year', 'decile'])
            .size()
            .unstack(fill_value=0)
            .apply(lambda x: x / x.sum(), axis=1)
        )

        count_df = count_df[sorted(count_df.columns)]
        count_df.sort_index().plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[color_map[d] for d in count_df.columns],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            legend=False
        )

        ax.set_ylabel(f"Quantile {quantile}", fontsize=12, weight="bold", color="#0b3c5d")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.tick_params(axis='y', pad=6)
        ax.set_title(f"Market Cap Decile Distribution – Quantile {quantile}",
                     fontsize=13, weight='bold', color="#0b3c5d", pad=10)

    axes[-1].set_xlabel("Year", fontsize=12, labelpad=6, color="#0b3c5d")

    # Shared legend below all subplots
    legend_handles = [
        plt.Line2D([0], [0], color=color_map[d], lw=6) for d in decile_labels
    ]
    legend_labels = [f"Decile {d+1} (of market cap)" for d in decile_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=10,
        title="Market Cap Decile (Higher Decile = Higher Market Cap)",
        title_fontsize=11
    )

    fig.suptitle("Market Cap Composition by Quantile of Omega-Sorted Portfolios (2002–2022), S&P 500/400/600 Universe", fontsize=16, weight='bold', color="#0b3c5d", y=0.94)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
