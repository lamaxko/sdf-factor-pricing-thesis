from numpy import outer
import pandas as pd
from datetime import date
from helpers import DB
from helpers import RetTimeSeries

class EquitiesAPI:
    def __init__(self):
        self.start_date = date(2000, 1, 1)
        self.end_date = date(2023, 1, 1)
        self.consts = self.get_const()
        self.factors = self.get_factors()
        self.returns_annual = self.get_annual_returns()
        self.data_annual = (
            pd.merge(self.consts, self.factors, on=['year', 'id_stock'], how='inner')
              .merge(self.returns_annual, on=['year', 'id_stock'], how='inner')
        )

        # self.returns_daily = self.get_daily_returns()
        # self.returns_monthly = self.calc_monthly_returns()
        # self.upload_monthly_returns()
        self.returns_monthly = self.get_monthly_returns()
        self.data_monthly = self.build_monthly_panel()


    def get_const(self):
        query = f"""
        SELECT c.year, im.index_name AS index_name, c.id_stock
        FROM constituents c
        JOIN index_mapper im ON c.id_index = im.id_index
        WHERE c.year >= {self.start_date.year} AND c.year <= {self.end_date.year};
        """
        df = DB.fetch(query, output="df")
        index_cols = [f'is_{name}' for name in df['index_name'].unique()]
        df_pivot = (
            df.assign(value=1)
              .pivot_table(index=['year', 'id_stock'], columns='index_name', values='value', fill_value=0)
              .rename(columns=lambda col: f'is_{col}')
              .reset_index()
        )
        df_pivot[index_cols] = df_pivot[index_cols].astype(int)
        return df_pivot
    
    def get_factors(self):
        query = f"""
        SELECT f.year, f.id_stock, f.value, m.factor_name 
        FROM factors f
        JOIN factor_mapper m ON f.id_factor = m.id_factor
        WHERE f.value != 'None' AND f.year >= {self.start_date.year} AND f.year <= {self.end_date.year};
        """
        df = DB.fetch(query, output="df")
        df_wide = df.pivot_table(
            index=['year', 'id_stock'],
            columns='factor_name',
            values='value'
        ).reset_index()
        return df_wide

    def get_daily_returns(self):
        query = f"""
            SELECT
              r.id_stock,
              r.date,
              r.return
            FROM
              returns AS r
            WHERE
              r.date BETWEEN '{self.start_date.year}-01-01' AND '{self.end_date.year}-12-31'
            ORDER BY
              r.id_stock,
              r.date;
        """
        df = DB.fetch(query, output="df")
        df = df.pivot(index="date", columns="id_stock", values="return")
        df = df / 100
        df.index = pd.to_datetime(df.index)
        return df

    def calc_monthly_returns(self):
        return RetTimeSeries(self.returns_daily).monthly_returns()

    def upload_monthly_returns(self):
        table_name = "returns_monthly"
        columns = {"date": "DATE",
                   "id_stock": "INTEGER",
                   "return": "REAL"}

        constraints = [
            "PRIMARY KEY (date, id_stock)",
        ]
        print(self.returns_monthly)
        df = self.returns_monthly.reset_index().melt(id_vars="date", var_name="id_stock", value_name="return")
        print(df)


        DB.drop_table(table_name)
        DB.create_table(table_name, columns, constraints=constraints)
        DB.upload_df(df, table_name)
        return

    def get_annual_returns(self):
        query = f"""
                SELECT year, id_stock, return FROM returns_annual
                WHERE year >= {self.start_date.year} AND year <= {self.end_date.year};
                """
        df = DB.fetch(query, output="df")
        return df
    
    def get_monthly_returns(self):
        query = f"""
                SELECT date, id_stock, return FROM returns_monthly
                WHERE date BETWEEN '{self.start_date.year}-01' AND '{self.end_date.year}-12'
                """
        df = DB.fetch(query, output="df")
        df = df.pivot(index="date", columns="id_stock", values="return")
        df.index = pd.to_datetime(df.index)
        return df

    def build_monthly_panel(self):
        # Step 1: Stack wide-form returns to long-form
        df_returns = self.returns_monthly.copy()
        df_long = df_returns.stack().reset_index()
        df_long.columns = ['date', 'id_stock', 'return']
        df_long['id_stock'] = df_long['id_stock'].astype(int)
        df_long['year'] = df_long['date'].dt.year

        # Step 2: Merge annual factor data (on year, id_stock)
        df_factors = self.factors.copy()
        df_factors['id_stock'] = df_factors['id_stock'].astype(int)
        df_factors['year'] = df_factors['year'].astype(int)

        df_long = df_long.merge(df_factors, on=['year', 'id_stock'], how='left')

        # Step 3: Merge index membership (constituents)
        df_consts = self.consts.copy()
        df_consts['id_stock'] = df_consts['id_stock'].astype(int)
        df_consts['year'] = df_consts['year'].astype(int)

        df_long = df_long.merge(df_consts, on=['year', 'id_stock'], how='left')
        const_columns = df_consts.columns.difference(['id_stock', 'year'])
        df_long[const_columns] = df_long[const_columns].fillna(0)
        df_long[const_columns] = df_long[const_columns].astype(int)
        df_long = df_long.loc[df_long[const_columns].any(axis=1)]

        # Step 4: Forward-fill factor and membership info per stock
        df_long = df_long.sort_values(['id_stock', 'date'])
        df_long.update(df_long.groupby(['id_stock', 'year']).ffill())

        return df_long
