import os
import sqlite3
import pandas as pd
from pathlib import Path

class DB:
    def __init__(self, db_path=Path(os.getenv("MS_DB_PATH"))):
        self.db_path = db_path 
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns, constraints=None):
        columns_def = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())

        if constraints:
            constraints_def = ", " + ", ".join(constraints)
        else:
            constraints_def = ""

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_def}{constraints_def})"

        self.cursor.execute(query)
        self.conn.commit()

    def create_index(self, index_name, table_name, columns, unique=False):
        unique_sql = "UNIQUE" if unique else ""
        cols = ", ".join(columns)
        query = f"CREATE {unique_sql} INDEX IF NOT EXISTS {index_name} ON {table_name} ({cols})"
        self.cursor.execute(query)
        self.conn.commit()

    def upload_df(self, df, table_name, if_exists="append", index=False):
        df = df.where(pd.notnull(df), None)
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=index)
    
    def fetch(self, query, params=None, output="df", parse_dates=None):
        """
        Fetches query results in various formats.

        output options:
        - 'df'     : returns a pandas DataFrame
        - 'tuples' : returns list of tuples (raw cursor.fetchall())
        - 'list'   : returns flattened list (first column only)
        - 'one'    : returns a single item (first row, first column)
        """
        output = output.lower()

        if output == "df":
            return pd.read_sql(query, self.conn, params=params, parse_dates=parse_dates)

        # for other outputs, execute the query directly
        self.cursor.execute(query, params or [])

        if output == "tuples":
            return self.cursor.fetchall()

        if output == "list":
            rows = self.cursor.fetchall()
            return [row[0] for row in rows]

        if output == "one":
            row = self.cursor.fetchone()
            return row[0] if row else None

        raise ValueError(f"Unsupported output format: {output}")

    def drop_table(self, table_name):
        query = f"DROP TABLE IF EXISTS {table_name}"
        self.cursor.execute(query)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()

DB = DB()
