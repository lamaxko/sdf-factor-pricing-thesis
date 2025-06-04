import os
import pandas as pd
from datetime import date
from helpers.time_series import RetTimeSeries
from helpers.db_manager import DB
from plots.factor_portfolios import plot_factor_premium, plot_monthly_geometric_means, plot_annual_returns, plot_cumulative_returns, plot_industry_composition, plot_market_cap_distribution

class FactorPortfolio:
    def __init__(self, factor: str, indices: list[str], num_quantiles: int, date_range: tuple):
        self.factor = factor
        self.indices = indices
        self.num_quantiles = num_quantiles
        self.start_date, self.end_date = date_range

        self.years = self._years_between()
        self.quantiles = self._quantiles()

        self.constituents = self._get_constituents()
        self.quantile_consts = self._get_quantile_consts()
        self.returns_raw = self._get_returns()

        self.pct_returns = self._calc_pct_returns()
        self.TS = RetTimeSeries(self.pct_returns)

        self.cum_returns = self.TS.cum_returns()
        self.monthly_returns = self.TS.monthly_returns()
        self.annual_returns = self.TS.annual_returns()

    def _years_between(self):
        start, end = sorted((self.start_date.year, self.end_date.year))
        return list(range(start, end + 1))

    def _quantiles(self):
        return range(1, self.num_quantiles + 1)

    def _get_constituents(self, factor_lag=1):
        constituents = {}
        for year in self.years:
            parts = []
            for index in self.indices:
                query = f"""
                    WITH params AS (
                      SELECT
                        '{self.factor}' AS factor_name,
                        '{index}' AS index_name,
                        {year} AS year,
                        {year - factor_lag} AS year_lagged
                    )
                    SELECT
                      c.id_stock,
                      f.value AS factor_value
                    FROM
                      params
                    JOIN index_mapper im ON im.index_name = params.index_name
                    JOIN constituents c ON c.id_index = im.id_index AND c.year = params.year
                    JOIN factor_mapper fm ON fm.factor_name = params.factor_name
                    JOIN factors f ON f.id_stock = c.id_stock AND f.id_factor = fm.id_factor AND f.year = params.year_lagged;
                """
                part_df = DB.fetch(query, output="df")
                parts.append(part_df)

            full_df = pd.concat(parts, ignore_index=True).drop_duplicates()
            constituents[year] = full_df
        return constituents

    def _get_quantile_consts(self):
        quantile_consts = {}
        for year in self.years:
            df = self.constituents[year].copy()
            df['quantile'] = pd.qcut(
                df['factor_value'],
                q=self.num_quantiles,
                labels=False,
                duplicates='drop'
            ) + 1
            quantile_consts[year] = {
                q: grp['id_stock'].tolist()
                for q, grp in df.groupby('quantile', sort=True)
            }
        return quantile_consts

    def _get_returns(self):
        returns = {}
        for year in self.years:
            quantiles = {}
            for quantile in self.quantiles:
                values = ", ".join(f"({const})" for const in self.quantile_consts[year][quantile])
                query = f"""
                    WITH stock_list(id_stock) AS (
                      VALUES
                        {values}
                    )
                    SELECT
                      r.id_stock,
                      r.date,
                      r.return
                    FROM
                      stock_list sl
                    JOIN
                      returns r INDEXED BY idx_returns_stock_date
                        ON r.id_stock = sl.id_stock
                    WHERE
                      r.date BETWEEN '{year}-01-01' AND '{year}-12-31'
                    ORDER BY
                      r.id_stock, r.date;
                """
                df = DB.fetch(query, output="df")
                df['date'] = pd.to_datetime(df['date'])
                ret_year = df.pivot(index='date', columns='id_stock', values='return')
                quantiles[quantile] = ret_year / 100
            returns[year] = quantiles
        return returns

    def _calc_pct_returns(self):
        pct_returns = {}
        for quantile in self.quantiles:
            pct_returns_quantile = []
            for year in self.years:
                returns_df = self.returns_raw[year][quantile]
                initial_date = returns_df.index.min() - pd.Timedelta(days=1)
                returns_with_initial = returns_df.copy()
                returns_with_initial.loc[initial_date] = 0
                returns_with_initial = returns_with_initial.sort_index()
                growth_index = (1 + returns_with_initial).cumprod() * 100
                growth_index['total'] = growth_index.sum(axis=1)
                pct_returns_quantile.append(growth_index['total'].pct_change().drop(initial_date))
            pct_returns[quantile] = pd.concat(pct_returns_quantile).sort_index()
        return pd.DataFrame(pct_returns)

    def premium(self, long_quantile: int, short_quantile: int, freq="d"):
        if freq == "d":
            returns = self.pct_returns.copy()
        elif freq == "m":
            returns = self.monthly_returns.copy()
        elif freq in ("y", "a"):
            returns = self.annual_returns.copy()
        return returns[long_quantile] - returns[short_quantile]

    def create_plots(self, save_dir, long=None, short=None, freq_premium="y", annual_labels=False, monthly_trend=False):
        os.makedirs(save_dir, exist_ok=True)

        plot_cumulative_returns(self.cum_returns, save_path=os.path.join(save_dir, "cum_returns_plot.png"))
        plot_annual_returns(self.annual_returns, show_labels=annual_labels, save_path=os.path.join(save_dir, "annual_returns_plot.png"))
        plot_monthly_geometric_means(self.monthly_returns, show_trend=monthly_trend, save_path=os.path.join(save_dir, "monthly_geo_means_plot.png"))
        plot_industry_composition(self.get_icbins(), save_path=os.path.join(save_dir, "icbin_comp_plot.png"))
        plot_market_cap_distribution(self.get_market_cap_distribution(),save_path=os.path.join(save_dir, "market_cap_comp_plot.png"))
        if long and short:
            plot_factor_premium(self.premium(long, short, freq=freq_premium), freq=freq_premium, save_path=os.path.join(save_dir, "factor_premium_y_plot.png"))


    def get_icbins(self):
        """
        Returns a dictionary:
            - keys: quantiles (e.g., 1 to 5)
            - values: DataFrames with:
                - index = id_stock
                - columns = years (as int)
                - values = industry (icbin)
        Fetches each (year, quantile) pair separately and merges without dropping any id_stock.
        """
        icbin_data = {q: [] for q in self.quantiles}

        for quantile in self.quantiles:
            rows = []

            for year in self.years:
                id_stocks = [int(s) for s in self.quantile_consts[year][quantile]]
                if not id_stocks:
                    continue

                values = ", ".join(f"({id_})" for id_ in id_stocks)

                query = f"""
                    WITH stock_list(id_stock) AS (
                        VALUES
                            {values}
                    )
                    SELECT
                        sl.id_stock,
                        sm.icbin
                    FROM stock_list sl
                    LEFT JOIN stock_mapper sm
                    ON sl.id_stock = sm.id_stock;
                """
                df = DB.fetch(query, output="df")
                df['id_stock'] = df['id_stock'].astype(int)
                df['year'] = int(year)
                rows.append(df[['id_stock', 'icbin', 'year']])

            # Concatenate all years for this quantile
            full_df = pd.concat(rows, ignore_index=True)

            # Pivot to: index = id_stock, columns = year, values = icbin
            pivot_df = full_df.pivot(index='id_stock', columns='year', values='icbin')
            pivot_df.columns = pivot_df.columns.astype(int)
            pivot_df.index.name = 'id_stock'

            icbin_data[quantile] = pivot_df

        return icbin_data


    def sanity_check(self):
        """
        Prints a summary of how many stocks are in each quantile per year.
        Helps verify consistency and coverage of quantile assignments.
        """
        print("\nSanity Check: Stock Counts per Quantile per Year")

        for quantile in self.quantiles:
            summary = {
                year: len(self.quantile_consts.get(year, {}).get(quantile, []))
                for year in self.years
            }
            df = pd.DataFrame.from_dict(summary, orient='index', columns=[f"Quantile {quantile}"])
            print(f"\nQuantile {quantile} Summary:")
            print(df.T)


    def get_market_cap_distribution(self):
        """
        Fetches market cap data across quantiles and years.
        Returns a DataFrame with columns: ['year', 'quantile', 'id_stock', 'market_cap']
        """
        all_rows = []

        for year in self.years:
            # Step 1: get all stocks for this year
            year_constituents = self.constituents[year]

            if year_constituents.empty:
                continue

            stock_ids = year_constituents['id_stock'].unique().tolist()
            stock_values = ", ".join(f"({int(s)})" for s in stock_ids)

            # Step 2: get market cap values for this year
            query = f"""
                WITH stock_list(id_stock) AS (
                    VALUES
                        {stock_values}
                ),
                mc_factor AS (
                    SELECT f.id_stock, f.value AS market_cap
                    FROM stock_list sl
                    JOIN factor_mapper fm ON fm.factor_name = 'MarketCap'
                    JOIN factors f ON f.id_stock = sl.id_stock AND f.id_factor = fm.id_factor AND f.year = {year}
                )
                SELECT * FROM mc_factor;
            """
            mc_df = DB.fetch(query, output="df")
            if mc_df.empty:
                continue

            mc_df['id_stock'] = mc_df['id_stock'].astype(int)
            mc_df['year'] = year

            # Step 3: assign quantile to each stock
            for q in self.quantiles:
                ids_in_q = self.quantile_consts[year].get(q, [])
                if not ids_in_q:
                    continue
                subset = mc_df[mc_df['id_stock'].isin(ids_in_q)].copy()
                subset['quantile'] = q
                all_rows.append(subset)

        # Final structure
        result_df = pd.concat(all_rows, ignore_index=True)
        return result_df[['year', 'quantile', 'id_stock', 'market_cap']]

    def get_factor_distribution(self):
        """
        Fetches factor data across quantiles and years.
        Returns a DataFrame with columns: ['year', 'quantile', 'id_stock', 'market_cap']
        """
        all_rows = []

        for year in self.years:
            year_constituents = self.constituents[year]

            if year_constituents.empty:
                continue

            stock_ids = year_constituents['id_stock'].unique().tolist()
            stock_values = ", ".join(f"({int(s)})" for s in stock_ids)

            query = f"""
                WITH stock_list(id_stock) AS (
                    VALUES
                        {stock_values}
                ),
                mc_factor AS (
                    SELECT f.id_stock, f.value AS value
                    FROM stock_list sl
                    JOIN factor_mapper fm ON fm.factor_name = '{self.factor}'
                    JOIN factors f ON f.id_stock = sl.id_stock AND f.id_factor = fm.id_factor AND f.year = {year}
                )
                SELECT * FROM mc_factor;
            """
            mc_df = DB.fetch(query, output="df")
            if mc_df.empty:
                continue

            mc_df['id_stock'] = mc_df['id_stock'].astype(int)
            mc_df['year'] = year

            # Step 3: assign quantile to each stock
            for q in self.quantiles:
                ids_in_q = self.quantile_consts[year].get(q, [])
                if not ids_in_q:
                    continue
                subset = mc_df[mc_df['id_stock'].isin(ids_in_q)].copy()
                subset['quantile'] = q
                all_rows.append(subset)

        # Final structure
        result_df = pd.concat(all_rows, ignore_index=True)
        return result_df[['year', 'quantile', 'id_stock', 'value']]


    def create_plots(self, save_dir, long=None, short=None, freq_premium="y", annual_labels=False, monthly_trend=False):

        os.makedirs(save_dir, exist_ok=True)

        plot_cumulative_returns(self.cum_returns, save_path=os.path.join(save_dir, "cum_returns_plot.png"))
        plot_annual_returns(self.annual_returns, show_labels=annual_labels, save_path=os.path.join(save_dir, "annual_returns_plot.png"))
        plot_monthly_geometric_means(self.monthly_returns, show_trend=monthly_trend, save_path=os.path.join(save_dir, "monthly_geo_means_plot.png"))
        plot_industry_composition(self.get_icbins(), save_path=os.path.join(save_dir, "icbin_comp_plot.png"))
        plot_market_cap_distribution(self.get_market_cap_distribution(),save_path=os.path.join(save_dir, "market_cap_comp_plot.png"))
        if long and short:
            plot_factor_premium(self.premium(long, short, freq=freq_premium), freq=freq_premium, save_path=os.path.join(save_dir, "factor_premium_y_plot.png"))
