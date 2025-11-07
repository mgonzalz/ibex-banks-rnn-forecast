# Notes:
#   Safe I/O utilities to read and write CSV files strictly within the .cache directory.
#   This module adds an integrity check that prevents writing any data outside of .cache,
#   and uses a temporary file (.tmp) to ensure atomic writes (avoiding partial saves).
#
# Purpose:
#   To guarantee data safety and reproducibility by enforcing that all cached data
#   (raw, features, exogenous, etc.) is stored only under the .cache/ hierarchy.
#   This also helps automated tests verify that no script writes outside the cache.

from pathlib import Path

import pandas as pd

# Absolute path to the cache root
CACHE_ROOT = Path(".cache").resolve()


def _is_under_cache(path: Path) -> bool:
    """
    Checks whether a given path is inside the .cache directory.
    Returns True if the path (or its parent) is under CACHE_ROOT.
    Handles non-existent paths safely.
    Args:
        path: The file path to check.
    Returns:
        bool: True if path is under .cache, False otherwise.
    """
    try:
        return CACHE_ROOT in path.resolve().parents or path.resolve() == CACHE_ROOT
    except FileNotFoundError:
        # Path may not exist yet; still resolve parent.
        return CACHE_ROOT in path.resolve().parents


def write_csv_cache(df: pd.DataFrame, out_path: Path):
    """
    Safely writes a DataFrame to a CSV file inside .cache/.
        - Verifies that the path is under .cache (security check).
        - Creates directories as needed.
        - Writes to a temporary .tmp file first, then renames atomically.
        (Prevents corruption if process stops mid-write.)
    Args:
        df: DataFrame to be saved.
        out_path: Destination path (must be under .cache/).
    Returns:
        None
    """
    out_path = out_path.resolve()
    if not _is_under_cache(out_path):
        raise ValueError(f"[ERROR] CSV file outside of .cache not allowed: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def read_csv(path: Path) -> pd.DataFrame:
    """
    Simple wrapper around pandas.read_csv for consistency.
    Reads any CSV file from a given path.
    Args:
        path: The file path to read the CSV from.
    Returns:
        pd.DataFrame: The contents of the CSV file as a DataFrame.
    """
    return pd.read_csv(path)
