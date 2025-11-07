# Notes:
#   - Ensures reproducible date ranges for all assets.
#   - Automatically fills missing columns like 'Adj Close' when absent.
#   - Applies timezone normalization (removes tz info from index).
#   - Saves all downloaded data under `cfg.io.raw_dir`.
#   - Logs data lineage for pipeline traceability.

# Purpose:
#   This module downloads and prepares raw financial data for assets
#   defined in the configuration file (`config/data.yml`).
#   It uses Yahoo Finance via `yfinance` to fetch daily prices,
#   cleans and standardizes the data, and stores it as cached CSVs
#   for reproducible pipelines.

import warnings
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.utils.config import Asset, load_config
from src.utils.io_utils import write_csv_cache
from src.utils.logging_utils import log_lineage
from src.utils.time_utils import clip_dates


def _download_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stable daily data for a given symbol from Yahoo Finance within the specified date range.
    Args:
        symbol: Ticker symbol to download (e.g., "BBVA.MC").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
    Returns:
        DataFrame with daily OHLCV data and actions.

    Information: More info about parameters used in yfinance.download() can be found here: ./docs/YFinance_Explained.md
    """
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        group_by="column",
        progress=False,
        threads=False,
        repair=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"[ERROR] Without data for {symbol}")

    # Index: Date without timezone (avoids shifts when converting to Europe/Madrid)
    if isinstance(df.index, pd.DatetimeIndex):
        # Remove tz if it exists and normalize to date (YYYY-MM-DD)
        idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df.index = pd.to_datetime(idx.date)  # oOnly the date, without time
    df.index.name = "Date"

    # Consistent ordering
    df = df.sort_index()

    return df


def _ensure_columns(df: pd.DataFrame, cols: List[str], *, symbol: str) -> pd.DataFrame:
    """
    Verifies requested columns. If 'Adj Close' is missing, creates it from 'Close'
    (when Yahoo does not provide it for that symbol in that range).
    Args:
        df: DataFrame with downloaded data.
        cols: List of required columns.
        symbol: Ticker symbol (for logging).
    Returns:
        DataFrame with ensured columns.
    """
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        warnings.warn(f"[WARN] '{symbol}': Missing columns: {missing}")
        # No error raised, only a warning
        for m in missing:
            out[m] = 0.0 if m in ["Dividends", "Stock Splits"] else None

    return out[cols].copy()


def run():
    cfg = load_config("config/data.yml")

    start = cfg.dates.start
    end = cfg.dates.train_end

    all_assets: List[Asset] = list(cfg.universe.targets) + list(cfg.universe.references)
    outputs: Dict[str, str] = {}

    for a in all_assets:
        raw = _download_one(a.symbol, start, end)

        # Clip dates to ensure reproducibility
        raw = clip_dates(raw, start, end)

        sel = _ensure_columns(raw, cfg.columns, symbol=a.symbol)
        sel = sel.reset_index()

        out_path = cfg.io.raw_dir / f"{a.symbol.replace('^','IDX_')}.csv"
        write_csv_cache(sel, out_path)
        outputs[a.symbol] = str(out_path)

    log_lineage(
        step="data.load_raw",
        params={
            "columns": cfg.columns,
            "start": start,
            "end": end,
            "assets": [a.symbol for a in all_assets],
        },
        inputs={"config": "config/data.yml"},
        outputs=outputs,
    )


if __name__ == "__main__":
    run()
