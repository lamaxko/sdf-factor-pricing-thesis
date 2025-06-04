import numpy as np
import pandas as pd
from apis.equities import EquitiesAPI
from helpers.db_manager import DB
from apis.kfrench import KfApi

class FactorCalc:
    def __init__(self, recalc=False, syear=2000, eyear=2023, lag=0):

        self.lag = lag
        if recalc:
            returns = EquitiesAPI().returns_monthly
            self.returns = returns

            self.price_index = (1 + self.returns).cumprod()
            self.index_returns = self._get_index_returns()

            self.upload_panel()
        self.syear = syear
        self.eyear = eyear
        self.panel = self.fetch_panel()

    def short_term_momentum(self):
        return self.returns.shift(2)

    def intermediate_momentum(self):
        return (
            (1 + self.returns.shift(2))
            .rolling(window=10, min_periods=10)
            .apply(np.prod, raw=True) - 1
        )

    def long_term_momentum(self):
        return (
            (1 + self.returns.shift(13))
            .rolling(window=24, min_periods=24)
            .apply(np.prod, raw=True) - 1
        )

    def short_term_reversal(self):
        return -self.returns.shift(1)

    def long_term_reversal(self):
        return -(
            (1 + self.returns.shift(1))
            .rolling(window=12, min_periods=12)
            .apply(np.prod, raw=True) - 1
        )

    def rel2high(self):
        """Price relative to 12-month high"""
        max_price = self.price_index.shift(1).rolling(window=12, min_periods=12).max()
        current_price = self.price_index
        return current_price / max_price

    def variance(self):
        """Rolling 12-month variance of returns"""
        return self.returns.rolling(window=12, min_periods=12).var()

    def _get_index_returns(self):
        df = DB.fetch(f"""
            SELECT 
                r.date,
                MAX(CASE WHEN sm.dscd = 'S&PMIDC' THEN r.return END) AS "S&PMIDC",
                MAX(CASE WHEN sm.dscd = 'S&PCOMP' THEN r.return END) AS "S&PCOMP",
                MAX(CASE WHEN sm.dscd = 'S&P600I' THEN r.return END) AS "S&P600I"
            WHERE sm.dscd IN (
                'S&PMIDC', 'S&PCOMP', 'S&P600I'
            )
            FROM returns_monthly r
            JOIN stock_mapper sm ON r.id_stock = sm.id_stock
            GROUP BY r.date
            ORDER BY r.date;
        """,
        output="df")
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns.name = 'index_name'
        return df 

    def rolling_beta_sp500(self):
        """12-month rolling beta with S&P 500 Composite Index"""
        return self._compute_rolling_beta(self.index_returns["S&PCOMP"])

    def rolling_beta_sp400(self):
        """12-month rolling beta with S&P 400 MidCap Index"""
        return self._compute_rolling_beta(self.index_returns["S&PMIDC"])

    def rolling_beta_sp600(self):
        """12-month rolling beta with S&P 600 SmallCap Index"""
        return self._compute_rolling_beta(self.index_returns["S&P600I"])


    def _compute_rolling_beta(self, index_series, window=12):
        """
        Efficient rolling beta computation using rolling covariance and variance.
        Returns DataFrame [date x stock_id].
        """
        # Align index with stock returns
        index_series.name = "index"
        aligned = self.returns.join(index_series, how="inner")
        stock_returns = aligned.drop(columns=["index"])
        index_returns = aligned["index"]

        # Rolling mean of index
        index_mean = index_returns.rolling(window).mean()

        # Rolling variance of index
        index_var = (index_returns - index_mean).pow(2).rolling(window).mean()

        # Rolling mean of stock returns
        stock_mean = stock_returns.rolling(window).mean()

        # Rolling covariance between each stock and the index
        cov = (
            (stock_returns.subtract(stock_mean, axis=0))
            .multiply(index_returns - index_mean, axis=0)
            .rolling(window).mean()
        )

        # Rolling beta = cov / var
        beta = cov.divide(index_var, axis=0)

        return beta

    def _get_annual_factor(self, factor_name):
        """
        Generic loader and formatter for annual firm-level factors.
        Maps year Y to monthly data in Y+1.
        """
        # Load factor values for the given factor name
        df = DB.fetch(f"""
            SELECT f.year, f.id_stock, f.value
            FROM factors f
            JOIN factor_mapper fm ON f.id_factor = fm.id_factor
            WHERE fm.factor_name = '{factor_name}'
            AND year >= 1999 AND year <= 2022
        """, output="df")

        # Drop missing values
        df = df.dropna(subset=["value"])

        # Create monthly dates for year+1 (Jan to Dec)
        records = []
        for _, row in df.iterrows():
            for month in range(1, 13):
                records.append({
                    "date": pd.Timestamp(year=int(row["year"]) - self.lag, month=int(month), day=1),
                    "id_stock": int(row["id_stock"]),
                    "value": row["value"]
                })

        monthly_df = pd.DataFrame(records)

        # Pivot to [date x id_stock]
        result = monthly_df.pivot(index="date", columns="id_stock", values="value")
        result.index.name = "date"
        result.columns.name = None
        result.index = pd.to_datetime(result.index)

        return result

    def forward_eps(self):
        return self._get_annual_factor("ForwardEPS")

    def trailing_eps(self):
        return self._get_annual_factor("TrailingEPS")

    def market_cap(self):
        return self._get_annual_factor("MarketCap")

    def excess_return(self):
        """
        Subtracts the Fama-French RF from each stock's return to compute excess return.
        """
        returns = self.returns.copy()
        rf = KfApi().ff5["RF"] / 100  # Convert % to decimal

        aligned = returns.join(rf, how="left")
        aligned_rf = aligned["RF"]

        excess = aligned.drop(columns="RF").subtract(aligned_rf, axis=0)
        excess[returns == 0] = np.nan

        return excess

    def all_factors(self, fillna_value=0):
        """Returns a dictionary of all seven factors"""
        factors = {
            "r2_1": self.short_term_momentum(),
            "r12_2": self.intermediate_momentum(),
            "r36_13": self.long_term_momentum(),
            "ST_Rev": self.short_term_reversal(),
            "LT_Rev": self.long_term_reversal(),
            "Rel2High": self.rel2high(),
            "Variance": self.variance(),
            "BetaSP500": self.rolling_beta_sp500(),
            "BetaSP400": self.rolling_beta_sp400(),
            "BetaSP600": self.rolling_beta_sp600(),
            "ForwardEPS": self.forward_eps(),
            "TrailingEPS": self.trailing_eps(),
            "MarketCap": self.market_cap(),
            "ExcessRet": self.excess_return(),
        }

        for name in factors:
            factors[name] = factors[name].fillna(fillna_value)
            print(factors[name])

        return factors
    
    def get_const(self):
        df = DB.fetch(
            """
            SELECT c.id_stock, c.year
            FROM constituents c
            JOIN index_mapper im ON c.id_index = im.id_index
            WHERE im.index_name IN ('S&P500', 'S&P400', 'S&P600')
            GROUP BY c.id_stock, c.year
            ORDER BY c.id_stock, c.year;
            """, output="df")
        return (
            df.groupby("year")["id_stock"]
            .apply(list)
            .to_dict()
        )

    def panel(self):
        """
        Creates a monthly panel of decile-normalized factors (0.1 to 1.0) for FFN/GAN.
        Only includes stocks that were constituents in the previous year.
        """
        all_factors = self.all_factors()
        constituents = self.get_const()

        panel_data = []

        # Get all months available
        all_dates = all_factors[next(iter(all_factors))].index

        for date in all_dates:
            year = date.year
            valid_stocks = constituents.get(year - 1, [])

            if not valid_stocks:
                continue

            # Will hold a dict of {id_stock: {factor_name: quantile_value}}
            stock_rows = {}

            for factor_name, df in all_factors.items():
                if date not in df.index:
                    continue

                factor_row = df.loc[date]
                factor_row = factor_row[factor_row.index.isin(valid_stocks)].dropna()

                if factor_row.empty:
                    continue

                # Rank and bin into deciles (0.1, 0.2, ..., 1.0)
                quantiles = pd.qcut(factor_row.rank(method="first"), 10, labels=[0.1 * (i + 1) for i in range(10)])

                for stock_id, q in quantiles.items():
                    if stock_id not in stock_rows:
                        stock_rows[stock_id] = {"date": date, "id_stock": stock_id}
                    
                    if factor_name == 'ExcessRet':
                        stock_rows[stock_id][factor_name] = float(factor_row[stock_id])
                    else:
                        stock_rows[stock_id][factor_name] = float(q)

            panel_data.extend(stock_rows.values())

        # Combine into DataFrame
        panel_df = pd.DataFrame(panel_data)

        # Ensure types
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        panel_df["id_stock"] = panel_df["id_stock"].astype(int)

        return panel_df.sort_values(["date", "id_stock"])

    def upload_panel(self):
        panel = self.panel()
        table_name = "panel_data"

        panel["date"] = panel["date"].dt.date

        float_cols = [col for col in panel.columns if col not in ["date", "id_stock", "ExcessRet"]]
        panel[float_cols] = panel[float_cols].round(1)

        columns = {
            "date": "DATE",
            "id_stock": "INTEGER",
            "ExcessRet": "REAL"
        }
        for col in float_cols:
            columns[col] = "REAL"

        constraints = [
            "PRIMARY KEY (date, id_stock)"
        ]

        DB.drop_table(table_name)
        DB.create_table(table_name, columns, constraints=constraints)
        DB.upload_df(panel, table_name)

    def fetch_panel(self):
        panel_df = DB.fetch(f"SELECT * FROM panel_data WHERE date BETWEEN '{self.syear}-01-01' AND '{self.eyear}-12-31' ", output="df")
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        panel_df["id_stock"] = panel_df["id_stock"].astype(int)
        return panel_df

