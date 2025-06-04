import requests
import pandas as pd
import numpy as np
from io import StringIO
from statsmodels.tsa.stattools import adfuller
from prep import make_stationary


class FredApi:
    def __init__(self):
        self.url = (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            "fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64"
        )
        self.data_raw = self.fetch_data()
        self.tcode_row = self.data_raw.iloc[0]
        self.data = self.clean_data()
        self.tcodes = self.get_tcodes()
        self.data_stationary = self.apply_stationarity()

    def fetch_data(self):
        response = requests.get(self.url)
        response.raise_for_status()

        csv_data = StringIO(response.text)

        return pd.read_csv(csv_data)

    def clean_data(self):
        df = self.data_raw.drop(index=0).reset_index(drop=True)
        df.columns = df.columns.map(str)
        datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df.set_index(datetime_col, inplace=True)

        return df

    def get_tcodes(self):
        return {
            col: int(float(val))
            for col, val in self.tcode_row.items()
            if col != self.data_raw.columns[0]
            and pd.notna(val)
            and str(val).strip().replace('.', '', 1).isdigit()
        }

    def apply_stationarity(self):
        transformed_cols = []

        for col in self.data.columns:
            tcode = self.tcodes.get(col)
            series = self.data[col]

            try:
                tcode = int(tcode)
                transformed = make_stationary(series, tcode=tcode)
                transformed = transformed.replace([np.inf, -np.inf], np.nan)
                transformed.name = col
                transformed_cols.append(transformed)
            except Exception as e:
                print(f"Warning: Could not transform {col} (tcode={tcode}): {e}")
                transformed = pd.Series(np.nan, index=self.data.index, name=col)
                transformed_cols.append(transformed)

        df_transformed = pd.concat(transformed_cols, axis=1)
        return df_transformed.copy()

    def adf_test(self, alpha=0.05):
        results = []

        for col in self.data_stationary.columns:
            series = self.data_stationary[col].replace([np.inf, -np.inf], np.nan).dropna()

            if len(series) < 20:
                results.append((col, None, 'Too short'))
                continue

            try:
                adf_result = adfuller(series, autolag='AIC')
                p_value = adf_result[1]
                status = 'Stationary' if p_value < alpha else 'Non-stationary'
                results.append((col, p_value, status))
            except Exception as e:
                results.append((col, None, f"Error: {str(e)}"))

        df_results = pd.DataFrame(results, columns=["Variable", "p-value", "Status"])
        return df_results.sort_values("p-value", na_position='last')

    def recession_indicator(self):
        """
        Fetches NBER US recession indicator (binary: 1 = recession, 0 = no recession).
        Returns:
            pd.Series: Date-indexed recession indicator (monthly frequency)
        """
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id=USREC&scale=left&cosd=1854-12-01&coed=2025-04-01&fq=Monthly"
        )
        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df.columns = ['date', 'recession']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return df['recession'].astype(int)
