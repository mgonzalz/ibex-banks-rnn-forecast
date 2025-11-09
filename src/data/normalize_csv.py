# Notes:
#   - Keeps Close as-is (no dividend adjustment).
#   - Ensures required columns exist with consistent dtypes.
#   - Fills missing "Adj Close" with "Close"; missing "Dividends" / "Stock Splits" with 0.0.
#   - Sorts by Date and drops rows without Close.
#   - Writes outputs via repo utilities to respect .cache policy and lineage logging.
#
# Purpose:
# Normalize loosely-formatted OHLCV CSV exports into a consistent schema and
# save them under a cached path structure. Designed to tolerate "weird" CSVs
# (e.g., two-row headers with tickers on the second row).

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io_utils import write_csv_cache
from src.utils.logging_utils import log_lineage

REQUIRED_COLUMNS = ["Date"] + load_config().columns

def _read_loose_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, header=[0, 1])
        if isinstance(df.columns, pd.MultiIndex):
            # Join non-empty parts (e.g. ("Date","") → "Date")
            df.columns = [
                c[0] if c[0] else c[1] if c[1] else "Unnamed" for c in df.columns
            ]
        else:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df


def load_clean_csv(path: str) -> pd.DataFrame:
    """
    Normalize columns and dtypes:
        - Parse dates in 'Date'.
        - Convert numeric columns to float/int.
        - Create missing columns:
            - Adj Close := Close (if missing or entirely NaN)
            - Dividends / Stock Splits := 0.0 if missing
        - Ensure all REQUIRED_COLUMNS exist (OHLCV may be created as NaN if absent).
        - Sort by date and drop rows without Close.
    Does NOT adjust prices for dividends (keeps Close as-is).
    Args:
        path: Path to the CSV file.
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized columns and types.
    """
    df = _read_loose_csv(path)

    # Rename columns (strip whitespace)
    rename_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure 'Date' exists
    if "Date" not in df.columns:
        raise ValueError("[ERROR] Missing 'Date' column in CSV.")

    # Parse dates and drop invalid rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()

    # Create missing columns
    if "Close" not in df.columns:
        raise ValueError("[ERROR] Missing 'Close' column in CSV.")

    if "Adj Close" not in df.columns or df["Adj Close"].isna().all():
        df["Adj Close"] = df["Close"]

    for opt in ["Dividends", "Stock Splits"]:
        if opt not in df.columns:
            df[opt] = 0.0

    # Ensure all required columns exist (if any OHLCV is missing then create NaN)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # Numeric types
    num_cols = [c for c in REQUIRED_COLUMNS if c != "Date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort and clean
    df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

    # Final selection and column order
    df = df[REQUIRED_COLUMNS]
    return df


def normalize_and_save(
    path: str | Path, symbol: str, out_dir: str | Path = ".cache/processed"
) -> Path:
    """
    Load, normalize, and save to .cache/processed/{symbol}.csv (safe write via repo utils).
    Args:
        path: Path to the input CSV file.
        symbol: Ticker symbol (e.g., "BBVA.MC") for naming the output file.
        out_dir: Output directory (default: .cache/processed).
    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    out_dir = Path(out_dir)
    df = load_clean_csv(str(path))
    out_path = out_dir / f"{symbol}.csv"
    write_csv_cache(df, out_path)  # respeta política .cache/*
    log_lineage(
        step="data.normalize_csv",
        params={"symbol": symbol, "required_columns": REQUIRED_COLUMNS},
        inputs={"source_csv": str(path)},
        outputs={"normalized_csv": str(out_path)},
    )
    return df
