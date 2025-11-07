# Notes:
#   This module handles the structured loading and validation of YAML configuration files
#   used by the IBEX-Banks-RNN project. It converts each section of config/data.yml
#   into strongly typed Python dataclasses, ensuring consistency and type safety
#   across the codebase.
#
# Purpose:
#   To centralize configuration management (timezone, tickers, dates, I/O directories),
#   automatically create cache directories, and provide a single entry point
#   for all other scripts (data loading, feature building, training, etc.).

from dataclasses import dataclass
from pathlib import Path
import yaml

# Data classes for each configuration section


@dataclass
class ProjectConfig:
    timezone: str  # e.g., "Europe/Madrid"


@dataclass
class Asset:
    symbol: str  # Ticker symbol (e.g., "BBVA.MC")
    name: str  # Human-readable name (e.g., "BBVA")


@dataclass
class UniverseConfig:
    targets: list  # list[Asset]: Main assets to predict
    references: list  # list[Asset]: Benchmark indices for comparison


@dataclass
class DatesConfig:
    start: str  # Start of historical data (e.g., "2000-01-01")
    train_end: str  # Last date for training period
    forecast_days: list  # List of forecast target dates


@dataclass
class IOConfig:
    cache_dir: Path  # Root .cache directory
    raw_dir: Path  # Subdirectory for raw downloaded data
    exo_dir: Path  # Subdirectory for exogenous variables
    features_dir: Path  # Subdirectory for processed feature datasets


@dataclass
class DataConfig:
    project: ProjectConfig
    universe: UniverseConfig
    columns: list  # List of data columns to load (e.g., ["Open", "Close", "Volume"])
    dates: DatesConfig
    io: IOConfig


# Helper functions


def _ensure_dirs(io_cfg: IOConfig):
    """
    Ensures that all required cache directories exist.
    Creates missing ones safely (idempotent).
    Args:
        io_cfg: IOConfig object with directory paths.
    Returns:
        None
    """
    for p in [io_cfg.cache_dir, io_cfg.raw_dir, io_cfg.exo_dir, io_cfg.features_dir]:
        p.mkdir(parents=True, exist_ok=True)


def load_config(path: str = "config/data.yml") -> DataConfig:
    """
    Loads the main data configuration from a YAML file and builds a structured DataConfig.
    Steps:
      1. Reads YAML safely.
      2. Creates typed dataclasses for each section.
      3. Ensures all I/O directories exist under .cache/.
      4. Returns a fully initialized DataConfig instance.
    Args:
        path: Path to the YAML configuration file.
    Returns:
        DataConfig: Fully populated configuration object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project = ProjectConfig(timezone=raw["project"]["timezone"])
    universe = UniverseConfig(
        targets=[Asset(**a) for a in raw["universe"]["targets"]],
        references=[Asset(**a) for a in raw["universe"]["references"]],
    )
    dates = DatesConfig(
        start=raw["dates"]["start"],
        train_end=raw["dates"]["train_end"],
        forecast_days=list(raw["dates"]["forecast_days"]),
    )
    io_cfg = IOConfig(
        cache_dir=Path(raw["io"]["cache_dir"]),
        raw_dir=Path(raw["io"]["raw_dir"]),
        exo_dir=Path(raw["io"]["exo_dir"]),
        features_dir=Path(raw["io"]["features_dir"]),
    )
    _ensure_dirs(io_cfg)

    return DataConfig(
        project=project,
        universe=universe,
        columns=list(raw["columns"]),
        dates=dates,
        io=io_cfg,
    )
