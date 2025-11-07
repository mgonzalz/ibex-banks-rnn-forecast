# Notes:
#   This module provides utility functions for handling time and date operations
#   within the IBEX-Banks-RNN project. These utilities are essential to ensure that
#   all time series data share the same timezone and are correctly delimited within
#   training and test date ranges defined in config/data.yml.
#
# Purpose:
#   To guarantee temporal consistency, reproducibility, and clean slicing of datasets
#   across the project. These functions help align OHLCV data from Yahoo Finance
#   and ensure that time boundaries (e.g., train/test) are applied uniformly.


import pandas as pd


def localize_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Localizes the DataFrame index to the specified timezone.
    Parameters:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        tz (str): Timezone string (e.g., 'Europe/Madrid').
    Returns:
        pd.DataFrame: DataFrame with localized index.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("[ERROR] DataFrame index must be a DatetimeIndex.")
    # Localize the index to the specified timezone
    df.index = df.index.tz_localize(tz)
    return df


def _align_ts_to_index_tz(ts_like, index_tz):
    """
    Aligns a timestamp-like object to the timezone of a given index.
    Parameters:
        ts_like: A timestamp-like object (str, pd.Timestamp, etc.).
        index_tz: The timezone of the index to align to (pytz timezone or string).
    Returns:
        pd.Timestamp: The timestamp aligned to the index timezone.
    """
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        # If the timestamp is naive, it is located directly at the tz of the index
        return ts.tz_localize(index_tz)
    # If it already has a timezone, it is converted (keeping the absolute instant)
    return ts.tz_convert(index_tz)


def clip_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Clips the DataFrame to only include rows between the start and end dates.
    Parameters:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        start (str): Start date (inclusive).
        end (str): End date (inclusive).
    Returns:
        pd.DataFrame: Clipped DataFrame.
    """
    index_tz = df.index.tz
    # If the index is naive, we accept naive timestamps
    if index_tz is None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    else:
        # Align the cutoff dates to the same tz as the index
        start_ts = _align_ts_to_index_tz(start, index_tz)
        end_ts = _align_ts_to_index_tz(end, index_tz)

    # Apply the time filter
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
