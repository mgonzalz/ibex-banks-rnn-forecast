# src/data/build_exogenos.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_config
from src.utils.io_utils import read_csv, write_csv_cache
from src.utils.logging_utils import log_lineage

# Cols to delete from raw price data
COLS_DELETE = [
    "Adj Close",
    "Dividends",
    "Stock Splits",
    "DiffDays",
    "IsGap",
    "Weekday",
    "Month",
    "Quarter",
]


def _sanitize_col(s: str) -> str:
    """Sanitize string to be a valid column name."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _load_prices(path) -> Dict[str, pd.DataFrame]:
    """
    Load raw price data for BBVA.MC and SAN.MC, clean unnecessary columns.
    Arguments:
        cfg: Config object loaded from config/data.yml
    Returns:
        Dict with DataFrames for each symbol.
    """
    paths = {
        "BBVA.MC": path / "BBVA.MC_enriched.csv",
        "SAN.MC": path / "SAN.MC_enriched.csv",
    }
    out = {}
    for sym, p in paths.items():
        df = read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # Eliminate unnecessary columns
        drop_cols = [c for c in COLS_DELETE if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        out[sym] = df
    return out


def _build_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Build daily calendar index from start to end dates.
    Arguments:
        start: Start date (inclusive).
        end: End date (inclusive).
    Returns:
        pd.DatetimeIndex with daily frequency.
    """
    return pd.date_range(start, end, freq="D")


def _build_events_df(
    events_list: List[dict],
    calendar_index: pd.DatetimeIndex,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a DataFrame of event indicators (0/1) for each event in events_list.
    Arguments:
        events_list: List of event dicts with keys 'name', 'start', 'end'.
        calendar_index: pd.DatetimeIndex for the full date range.
        date_start: Start date of the project (for clipping).
        date_end: End date of the project (for clipping).
    Returns:
        pd.DataFrame with event indicator columns.
    """
    events_df = pd.DataFrame(index=calendar_index)
    if not events_list:
        return events_df

    for ev in events_list:
        name = ev.get("name", "EVT")
        start = pd.to_datetime(ev.get("start", date_start))
        end = pd.to_datetime(ev.get("end", ev.get("start", date_start)))
        start = max(start, date_start)
        end = min(end, date_end)
        col = f"EVT_{_sanitize_col(name)}"
        mask = (calendar_index >= start) & (calendar_index <= end)
        events_df[col] = np.where(mask, 1, 0)

    # Ensure integer type and fill NaNs with 0
    for c in events_df.columns:
        events_df[c] = events_df[c].fillna(0).astype(int)
    return events_df


def _load_macro_daily(
    macro_dir: Path, calendar_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Read macroeconomic CSV files and convert to daily frequency.
    Arguments:
        macro_dir: Directory where macro CSV files are stored.
        calendar_index: pd.DatetimeIndex for daily frequency.
    Returns:
        pd.DataFrame with daily macroeconomic indicators.
    """
    macro_dir = Path(macro_dir)
    files = {
        "MACRO_ECB_Deposit_Rate": macro_dir / "MACRO_ECB_Deposit_Rate.csv",
        "MACRO_Inflation_HICP_EA": macro_dir / "MACRO_Inflation_HICP_EA.csv",
        "MACRO_IBEX_Close": macro_dir / "MACRO_IBEX_Close.csv",
    }

    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] Missing macro file: {p}")

    # ECB Deposit Rate
    ecb = read_csv(files["MACRO_ECB_Deposit_Rate"])
    ecb["Date"] = pd.to_datetime(ecb["Date"])
    ecb = (
        ecb.rename(columns={"DepositRate": "MACRO_ECB_Deposit_Rate"})
        .set_index("Date")
        .sort_index()
    )

    # Inflation HICP EA
    hicp = read_csv(files["MACRO_Inflation_HICP_EA"])
    hicp["Date"] = pd.to_datetime(hicp["Date"])
    hicp = (
        hicp.rename(columns={"Inflation": "MACRO_Inflation_HICP_EA"})
        .set_index("Date")
        .sort_index()
    )

    # IBEX 35 Close
    ibex = read_csv(files["MACRO_IBEX_Close"])
    ibex["Date"] = pd.to_datetime(ibex["Date"])
    ibex = (
        ibex.rename(columns={"IBEX_Close": "MACRO_IBEX35"})
        .set_index("Date")
        .sort_index()
    )

    macro_daily = pd.DataFrame(index=calendar_index)

    def _to_daily(df: pd.DataFrame) -> pd.Series:
        tmp = df.reindex(calendar_index)
        # If missing values, interpolate linearly and ffill/bfill
        if tmp.isna().sum().sum() > 0:
            tmp = tmp.interpolate(method="linear").ffill().bfill()
        return tmp.iloc[:, 0]

    macro_daily["MACRO_ECB_Deposit_Rate"] = _to_daily(ecb)
    macro_daily["MACRO_Inflation_HICP_EA"] = _to_daily(hicp)
    macro_daily["MACRO_IBEX35"] = _to_daily(ibex)

    return macro_daily


def _merge_all(
    price_df: pd.DataFrame, events_df: pd.DataFrame, macro_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge previous + events + macro to daily frequency and fill reasonable gaps.
    Arguments:
        price_df: DataFrame with price data (indexed by Date).
        events_df: DataFrame with event indicators (indexed by Date).
        macro_daily: DataFrame with daily macro indicators (indexed by Date).
    Returns:
        Merged DataFrame with all exogenous variables.
    """
    out = price_df.merge(events_df, left_index=True, right_index=True, how="left")
    out = out.merge(macro_daily, left_index=True, right_index=True, how="left")

    evt_cols = [c for c in events_df.columns] if not events_df.empty else []
    if evt_cols:
        out[evt_cols] = out[evt_cols].fillna(0).astype(int)

    macro_cols = list(macro_daily.columns)
    out[macro_cols] = out[macro_cols].ffill().bfill()

    return out


def run():
    # Load configuration
    cfg = load_config("config/data.yml")
    DATE_START = pd.Timestamp(cfg.dates.start)
    DATE_END = pd.Timestamp(cfg.dates.train_end)

    PROC_DIR = Path(".cache/processed")
    EXO_DIR = cfg.io.exo_dir
    MACRO_DIR = ".cache/macro"
    EXO_DIR.mkdir(parents=True, exist_ok=True)

    # Load clean prices
    prices = _load_prices(PROC_DIR)
    bbva = prices["BBVA.MC"]
    san = prices["SAN.MC"]

    # Build calendar
    calendar_index = _build_calendar(DATE_START, DATE_END)

    # Read exogenous events from YAML
    exo_yaml = Path("config/exogenous_events.yml")
    if not exo_yaml.exists():
        raise FileNotFoundError(f"[ERROR] Missing YAML: {exo_yaml}")
    with open(exo_yaml, "r", encoding="utf-8") as f:
        exo_cfg = yaml.safe_load(f) or {}

    events_list = exo_cfg.get("events", []) or []
    events_df = _build_events_df(events_list, calendar_index, DATE_START, DATE_END)

    # 5) Load daily macro from .cache/macro
    macro_daily = _load_macro_daily(MACRO_DIR, calendar_index)

    # 6) Merge final por símbolo
    bbva_exo = _merge_all(bbva, events_df, macro_daily)
    san_exo = _merge_all(san, events_df, macro_daily)

    # 7) Guardar datasets
    out_bbva = EXO_DIR / "BBVA.MC.csv"
    out_san = EXO_DIR / "SAN.MC.csv"
    write_csv_cache(bbva_exo.reset_index(), out_bbva)
    write_csv_cache(san_exo.reset_index(), out_san)

    # 8) Logging de linaje (tu función exacta)
    log_lineage(
        step="data.build_exogenous",
        params={
            "date_start": str(DATE_START.date()),
            "date_end": str(DATE_END.date()),
            "symbols": ["BBVA.MC", "SAN.MC"],
            "num_events": len(events_df.columns),
            "num_macros": len(macro_daily.columns),
            "scaling": "none",  # aún no escalado
            "merge_method": "left",
        },
        inputs={
            "events_yml": "config/exogenous_events.yml",
            "macro_files": json.dumps(
                {
                    "ECB_Rate": str(EXO_DIR / "MACRO_ECB_Deposit_Rate.csv"),
                    "Inflation": str(EXO_DIR / "MACRO_Inflation_HICP_EA.csv"),
                    "IBEX_Close": str(EXO_DIR / "MACRO_IBEX_Close.csv"),
                },
                ensure_ascii=False,
            ),
            "raw_bbva": str(cfg.io.raw_dir / "BBVA.MC.csv"),
            "raw_san": str(cfg.io.raw_dir / "SAN.MC.csv"),
        },
        outputs={
            "bbva_exogenous": str(out_bbva),
            "san_exogenous": str(out_san),
        },
    )

    print(f"[COMPLETE] Exogenous datasets saved to {EXO_DIR}")


if __name__ == "__main__":
    run()
