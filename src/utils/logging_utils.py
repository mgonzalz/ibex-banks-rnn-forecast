# Notes:
#   This module implements simple but structured logging utilities for tracking
#   data lineage — that is, which inputs, parameters, and outputs were used in each
#   processing step. Each record is written as a JSON line inside logs/data_lineage.jsonl.
#
# Purpose:
#   To provide transparency, reproducibility, and auditability throughout the pipeline.
#   Every data transformation (raw → exogenous → features → model) can be traced back
#   to its parameters and input files. This helps debug, document experiments,
#   and comply with good data engineering practices.

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

# Global constant: Path to the data lineage log file
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_LINEAGE = LOGS_DIR / "data_lineage.jsonl"


# Helper functions
def _hash_dict(d: Dict[str, Any]) -> str:
    """
    Creates a short SHA-256 hash from a dictionary.
    Used to uniquely identify parameter sets.
    Args:
        d: Dictionary to hash.
    Returns:
        str: First 12 characters of the SHA-256 hash.
    """
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# Main logging function
def log_lineage(
    step: str, params: Dict[str, Any], inputs: Dict[str, str], outputs: Dict[str, str]
):
    """
    Appends a new record describing one processing step to data_lineage.jsonl.
    Args:
        step: Name of the processing step (e.g., 'load_raw', 'build_features').
        params: Dictionary of parameters used in this step.
        inputs: Mapping of input file names or sources.
        outputs: Mapping of output files or destinations.
    Returns:
        None
    """
    rec = {
        "ts": int(time.time()),
        "step": step,
        "params": params,
        "params_hash": _hash_dict(params),
        "inputs": inputs,
        "outputs": outputs,
    }
    with open(DATA_LINEAGE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
