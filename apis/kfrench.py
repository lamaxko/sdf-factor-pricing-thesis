import requests
import pandas as pd
from io import BytesIO, StringIO
from zipfile import ZipFile

class KfApi:
    def __init__(self):
        self.url_ff5 = (
            "https://mba.tuck.dartmouth.edu/pages/faculty"
            "/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        )
        self.raw_ff5 = self.fetch_data_ff5()
        self.ff5 = self.clean_data_ff5()

    def fetch_data_ff5(self):
        r = requests.get(self.url_ff5)
        r.raise_for_status()
        with ZipFile(BytesIO(r.content)) as z:
            name = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(name) as f:
                content = f.read().decode('utf-8')

        lines = content.splitlines()

        start = next(i for i, line in enumerate(lines) if line.strip().startswith("Date") or "Mkt-RF" in line)
        end = next(i for i, line in enumerate(lines) if "Annual Factors" in line)

        data_block = '\n'.join(lines[start:end])
        df = pd.read_csv(StringIO(data_block))
        return df

    def clean_data_ff5(self):
        df = self.raw_ff5.copy()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        df.set_index("Date", inplace=True)
        return df
