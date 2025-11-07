# Notes:
#   - It infers the symbol name directly from the filename.
#   - It creates a validation summary CSV under `.cache/validation/`.
#   - Results are also logged for pipeline traceability.
#
# Purpose:
#   This module validates the integrity of all raw CSV files stored under
#   `.cache/raw/`. It checks for missing values, duplicated dates, and
#   temporal consistency (min/max dates, monotonic order).

from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io_utils import read_csv
from src.utils.logging_utils import log_lineage


def _infer_symbol_from_path(path: Path) -> str:
    """
    Infers the ticker symbol from the given file path. Assumes filenames are like:
        - "BBVA.MC.csv" for regular assets
        - "IDX_IBEX35.csv" for indices, which maps to "^IBEX35"
    Args:
        path: The file path to infer the symbol from.
    Returns:
        str: The inferred ticker symbol.
    """
    stem = path.stem
    return stem.replace("IDX_", "^")


def _report_for(df: pd.DataFrame, symbol: str) -> dict:
    """
    Generates a small integrity report for the given DataFrame.
    Checks for NaNs, duplicates (by Date), and date consistency.
    Args:
        df: DataFrame to analyze.
        symbol: Ticker symbol for reporting.
    Returns:
        dict: Integrity report with various metrics.
    """
    if "Date" not in df.columns:
        raise ValueError(f"[ERROR] File for {symbol} has no 'Date' column.")

    # Clean and sort dates
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"]).sort_values("Date")

    report = {
        "Symbol": symbol,
        "Rows": int(len(df)),
        "NaN Total": int(df.isna().sum().sum()),
        "Duplicates": int(df.duplicated(subset=["Date"]).sum()),
        "Date Min": str(df["Date"].min()) if not df.empty else None,
        "Date Max": str(df["Date"].max()) if not df.empty else None,
        "Monotonic Increasing": df["Date"].is_monotonic_increasing,
    }
    return report


def run():
    cfg = load_config("config/data.yml")
    inputs = {}
    results = []

    for p in sorted(cfg.io.raw_dir.glob("*.csv")):
        df = read_csv(p)
        sym = _infer_symbol_from_path(p)
        rep = _report_for(df, sym)
        results.append(rep)
        inputs[sym] = str(p)

    summary = pd.DataFrame(results)
    out_path = cfg.io.cache_dir / "validation" / "Raw_Integrity_Summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    log_lineage(
        step="data.validate_integrity",
        params={"checks": ["NaN", "Duplicates", "Monotonic", "Date Range"]},
        inputs=inputs,
        outputs={"summary_csv": str(out_path)},
    )


if __name__ == "__main__":
    run()
