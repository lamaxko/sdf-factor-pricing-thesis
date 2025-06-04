import pandas as pd
from datetime import datetime
from helpers.db_manager import DB

def read(syear, eyear):
    panel_df = pd.read_csv(fr"out/panel/panel_{syear}_{eyear}.csv", parse_dates=["date"])
    df_omega = panel_df.pivot(index="date", columns="id_stock", values="omega")
    df_excess_ret = panel_df.pivot(index="date", columns="id_stock", values="ExcessRet")
    return df_omega, df_excess_ret

def get_range():
    start_year = 2000
    years_to_load = list(range(2006, 2024))

    dfs_omega = []
    dfs_excess_ret = []

    latest_date = None

    for eyear in years_to_load:
        df_omega, df_excess_ret = read(start_year, eyear)

        if latest_date is not None:
            df_omega = df_omega[df_omega.index > latest_date]
            df_excess_ret = df_excess_ret[df_excess_ret.index > latest_date]

        if not df_omega.empty:
            latest_date = df_omega.index.max()

        dfs_omega.append(df_omega)
        dfs_excess_ret.append(df_excess_ret)

    omega_full = pd.concat(dfs_omega).sort_index()
    excess_ret_full = pd.concat(dfs_excess_ret).sort_index()

    return omega_full, excess_ret_full

def compute_rolling_avg_omega(omega_df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    rolling_avg = omega_df.rolling(window=window, min_periods=window).mean()

    # Add year column for annual aggregation
    rolling_avg['year'] = rolling_avg.index.year
    omega_annual = rolling_avg.groupby('year').last().drop(columns='year', errors='ignore')

    return omega_annual

def compute_factor_return(omega_df: pd.DataFrame, excess_ret_df: pd.DataFrame) -> pd.Series:
    omega_lagged = omega_df.shift(1)

    factor_returns = []

    for date in omega_lagged.index:
        omega_row = omega_lagged.loc[date]
        ret_row = excess_ret_df.loc[date]

        valid = omega_row.notna() & ret_row.notna()
        if valid.sum() == 0:
            factor_returns.append(None)
            continue

        omega_valid = omega_row[valid]
        ret_valid = ret_row[valid]

        weights = omega_valid / omega_valid.sum()
        factor_return = (weights * ret_valid).sum()
        factor_returns.append(factor_return)

    return pd.Series(factor_returns, index=omega_lagged.index)


def compute_rolling_betas(excess_ret: pd.DataFrame, factor_return: pd.Series, window: int = 12) -> pd.DataFrame:
    factor_return = factor_return.reindex(excess_ret.index)
    betas = {}

    for stock in excess_ret.columns:
        stock_ret = excess_ret[stock]
        combined = pd.concat([stock_ret, factor_return], axis=1, keys=["r", "f"]).dropna()

        if combined.empty or len(combined) < window:
            continue

        cov_rf = combined["r"].rolling(window).cov(combined["f"])
        var_f = combined["f"].rolling(window).var()

        beta_series = cov_rf / var_f
        betas[stock] = beta_series

    beta_df = pd.DataFrame(betas).dropna(how="all")

    # Keep only the December values (rolling beta based on data from Jan–Dec)
    beta_df = beta_df[beta_df.index.month == 12]

    # Group by year and use the December value (which is the last in the year)
    beta_df['year'] = beta_df.index.year
    beta_annual = beta_df.groupby('year').first().drop(columns='year', errors='ignore')

    return beta_annual

def insert_factor_if_not_exists(factor_name):
    query = "SELECT id_factor FROM factor_mapper WHERE factor_name = ?"
    existing_id = DB.fetch(query, params=(factor_name,), output="one")

    if existing_id:
        return existing_id

    max_id = DB.fetch("SELECT MAX(id_factor) FROM factor_mapper", output="one") or 0
    new_id = max_id + 1

    df_new_factor = pd.DataFrame({
        "id_factor": [new_id],
        "factor_name": [factor_name]
    })
    DB.upload_df(df_new_factor, "factor_mapper")

    return new_id

def upload_factor(beta_df: pd.DataFrame, factor_name: str, invert_values=False):
    id_factor = insert_factor_if_not_exists(factor_name)

    beta_long = beta_df.reset_index().melt(id_vars="year", var_name="id_stock", value_name="value")
    beta_long = beta_long.dropna(subset=["value"])


    beta_long["id_factor"] = id_factor

    beta_long["year"] = beta_long["year"].astype(int)
    beta_long["id_stock"] = beta_long["id_stock"].astype(int)
    beta_long["id_factor"] = beta_long["id_factor"].astype(int)

    beta_long = beta_long[["year", "id_factor", "id_stock", "value"]]
    if invert_values:
        beta_long["value"] = -beta_long["value"]
    print(beta_long)

    DB.upload_df(beta_long, "factors")

    print(f"Uploaded {len(beta_long)} rows for factor '{factor_name}' (id {id_factor})")
    return

if __name__ == "__main__":
    omega, excess_ret = get_range()
    print(omega, excess_ret)

    factor_return = compute_factor_return(omega, excess_ret)
    print(factor_return)
    factor_return.to_csv("out/factor_return_series.csv")

    # Compute Rolling Omega
    omega_avg_df = compute_rolling_avg_omega(omega)
    print(omega_avg_df)
    omega_avg_df.to_csv("out/rolling_avg_omega_by_year.csv")

    factor_name = "RollingOmega12M_lag1_neg"
    upload_factor(omega_avg_df, factor_name, invert_values=True)


    # Compute Rolling Beta
    beta_df = compute_rolling_betas(excess_ret, factor_return)
    print(beta_df)
    beta_df.to_csv("out/betas_by_year.csv")


    factor_name = "RollingOmBeta12M_lag1_neg"
    upload_factor(beta_df, factor_name, invert_values=True)
