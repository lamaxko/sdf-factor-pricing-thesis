import numpy as np
import pandas as pd

def make_stationary(series, tcode=None, tname=None):
    """
    Transform a pandas Series to a stationary series using FRED-MD transformation conventions.

    Args:
        series (pd.Series): Raw time series data.
        tcode (int, optional): Transformation code (1–7).
        tname (str, optional): Descriptive name for transformation. Ignored if tcode is provided.

    Returns:
        pd.Series: Transformed series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    # Normalize tname if provided
    if tcode is None and tname is not None:
        tname = tname.strip().lower()
        tname_map = {
            "log diff": 1,
            "first diff": 2,
            "second diff": 3,
            "log second diff": 4,
            "log": 5,
            "percent change": 6,
            "second diff percent change": 7,
            "second diff of pct change": 7,
        }
        tcode = tname_map.get(tname)
        if tcode is None:
            raise ValueError(f"Unknown tname '{tname}'. Valid options: {list(tname_map.keys())}")

    if tcode is None:
        raise ValueError("You must specify either a valid tcode or tname.")

    # Apply transformation safely
    def safe_log(s):
        with np.errstate(divide='ignore', invalid='ignore'):
            logged = np.log(s.replace(0, np.nan))
        return pd.Series(logged, index=s.index)

    if tcode == 1:
        return safe_log(series).diff()
    elif tcode == 2:
        return series.diff()
    elif tcode == 3:
        return series.diff().diff()
    elif tcode == 4:
        return safe_log(series).diff().diff()
    elif tcode == 5:
        return safe_log(series)
    elif tcode == 6:
        return series.pct_change(fill_method=None) * 100
    elif tcode == 7:
        return series.pct_change(fill_method=None).diff().diff() * 100
    else:
        raise ValueError(f"Invalid tcode '{tcode}'. Must be an integer between 1 and 7.")
