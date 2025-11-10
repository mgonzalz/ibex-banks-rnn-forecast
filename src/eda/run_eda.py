# Notes:
#   Run EDA pipeline for IBEX-Banks (BBVA & SAN) for figures only + data lineage logging.
#   Saves figures in reports/figures and logs lineage to data_lineage.jsonl.
#
# Purpose:
#   This script performs exploratory data analysis (EDA) on stock data for BBVA and SAN.
#   It generates various plots (seasonal decomposition, weekday boxplots, monthly means,
#   volatility regimes, rolling correlation) and saves them to the reports/figures directory.
#   Additionally, it enriches the datasets with new features and logs the data lineage
#   including inputs, outputs, and parameters used in the analysis.

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data.normalize_csv import normalize_and_save
from src.utils.logging_utils import log_lineage

plt.rcParams["figure.dpi"] = 110

# Directories & constants
RAW_DIR = Path(".cache/raw")
PROC_DIR = Path(".cache/processed")
FIG_DIR = Path("reports/figures")
for d in (PROC_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BBVA.MC", "SAN.MC"]

# Helper for year ticks across all time-series plots
YEARS_TICKS = pd.date_range(start="2000-01-01", end="2025-01-01", freq="YS")
XMIN, XMAX = pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31")


# Temporal integrity
def add_temporal_checks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    dupes = df.index.duplicated().sum()
    if dupes:
        logging.warning(
            f"[ERROR] [{name}] {dupes} duplicated dates. Keeping first occurrence."
        )
        df = df[~df.index.duplicated(keep="first")].copy()
    diff_days = df.index.to_series().diff().dt.days
    df["DiffDays"] = diff_days
    df["IsGap"] = (diff_days > 1).astype(int)
    return df


# -------------------------
# Decomposition
# -------------------------
def decompose_and_save(df: pd.DataFrame, symbol: str, period: int = 252) -> str:
    series = df["Close"].astype(float).dropna()
    min_len = max(2 * period, period + 10)
    if len(series) < min_len:
        logging.warning(
            f"[ERROR] [{symbol}] Not enough data for seasonal_decompose(period={period}). Skipping."
        )
        return ""

    result = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(result.observed, label="Observed")
    axes[1].plot(result.trend, label="Trend")
    axes[2].plot(result.seasonal, label="Seasonal")
    axes[3].plot(result.resid, label="Residual")

    axes[0].set_title("Observed")
    axes[1].set_title("Trend")
    axes[2].set_title("Seasonal")
    axes[3].set_title("Residual")

    for ax in axes:
        ax.set_xticks(YEARS_TICKS)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlim(XMIN, XMAX)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"{symbol} — Seasonal Decomposition", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = FIG_DIR / f"Decomposition_{period}_{symbol}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[OUTPUT][{symbol}] Generated seasonal decomposition plot: {out}")
    return str(out)


# Calendar features & plots
def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["Weekday"] = df.index.weekday
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["ReturnPCT"] = df["Close"].pct_change() * 100.0
    return df


# Plotting functions
def plot_weekday_boxplots(bbva: pd.DataFrame, san: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    bbva.boxplot(column="ReturnPCT", by="Weekday", ax=axes[0])
    axes[0].set_title("BBVA.MC — Returns by Weekday")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Daily Return (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    san.boxplot(column="ReturnPCT", by="Weekday", ax=axes[1])
    axes[1].set_title("SAN.MC — Returns by Weekday")
    axes[1].set_xlabel("Day")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Weekly Return Patterns", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "Weekly_Return_Patterns.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logging.info(f"[OUTPUT] Generated: {out}")
    return str(out)


def plot_monthly_means(bbva: pd.DataFrame, san: pd.DataFrame) -> str:
    mean_month_bbva = bbva.groupby("Month")["ReturnPCT"].mean()
    mean_month_san = san.groupby("Month")["ReturnPCT"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].bar(mean_month_bbva.index, mean_month_bbva.values)
    axes[0].set_title("BBVA.MC — Average Monthly Return")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average Return (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].bar(mean_month_san.index, mean_month_san.values)
    axes[1].set_title("SAN.MC — Average Monthly Return")
    axes[1].set_xlabel("Month")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Mean Monthly Return Analysis", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "Monthly_Mean_Returns.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logging.info(f"[OUTPUT] Generated: {out}")
    return str(out)


# Volatility & Regimes
def add_volatility_and_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ReturnPCT"] = df["Close"].pct_change() * 100.0
    df["Vol30"] = df["ReturnPCT"].rolling(30).std()
    vol_z = (df["Vol30"] - df["Vol30"].mean()) / df["Vol30"].std()
    df["RegimeFlag"] = (vol_z > 1.5).astype(int)
    return df


def plot_volatility(df: pd.DataFrame, symbol: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Vol30"], label="Vol30 (30-day rolling std)")
    ax.fill_between(
        df.index,
        0,
        df["Vol30"],
        where=df["RegimeFlag"] == 1,
        alpha=0.2,
        label="High volatility",
    )

    ax.set_xticks(YEARS_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlim(XMIN, XMAX)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.suptitle(
        "Temporal evolution of volatility (Rolling 30 days)\nVolatility Regime Detection",
        fontsize=13,
    )
    out = FIG_DIR / f"Volatility_{symbol}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logging.info(f"[OUTPUT] Generated: {out}")
    return str(out)


# Correlation rolling plot
def correlation_rolling_plot(bbva: pd.DataFrame, san: pd.DataFrame) -> str:
    if "ReturnPCT" not in bbva.columns:
        bbva["ReturnPCT"] = bbva["Close"].pct_change() * 100.0
    if "ReturnPCT" not in san.columns:
        san["ReturnPCT"] = san["Close"].pct_change() * 100.0

    corr_global = bbva["ReturnPCT"].corr(san["ReturnPCT"])
    logging.info(
        f"[DATA] Global correlation BBVA–SAN (Daily returns, all years): {corr_global:.3f}"
    )

    rolling_corr = bbva["ReturnPCT"].rolling(60).corr(san["ReturnPCT"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_corr, label="Rolling correlation (60d)")
    ax.axhline(rolling_corr.mean(), ls="--", color="black", label="Mean")

    ax.set_xticks(YEARS_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlim(XMIN, XMAX)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax.set_title("Dynamic Correlation BBVA vs SAN (60-day window)")
    out = FIG_DIR / "Rolling_Correlation_BBVA_SAN.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logging.info(f"[OUTPUT] Generated: {out}")
    return str(out)


# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Run EDA for IBEX-Banks (BBVA & SAN) — Figures + Lineage"
    )
    parser.add_argument(
        "--period",
        type=int,
        default=252,
        help="Period for Seasonal Decomposition (default: 252)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1) Normalize inputs using our module
    bbva = normalize_and_save(
        RAW_DIR / "BBVA.MC.csv", symbol=SYMBOLS[0], out_dir=PROC_DIR
    )
    san = normalize_and_save(
        RAW_DIR / "SAN.MC.csv", symbol=SYMBOLS[1], out_dir=PROC_DIR
    )
    if "Date" in bbva.columns:
        bbva = bbva.set_index("Date")
    if "Date" in san.columns:
        san = san.set_index("Date")

    # Temporal checks
    bbva = add_temporal_checks(bbva, "BBVA.MC")
    san = add_temporal_checks(san, "SAN.MC")

    # Decomposition
    figs = []
    figs.append(decompose_and_save(bbva, "BBVA.MC", period=args.period))
    figs.append(decompose_and_save(san, "SAN.MC", period=args.period))

    # Weekly / Monthly patterns
    bbva = calendar_features(bbva)
    san = calendar_features(san)
    figs.append(plot_weekday_boxplots(bbva, san))
    figs.append(plot_monthly_means(bbva, san))

    # Volatility + Regimes
    bbva = add_volatility_and_regime(bbva)
    san = add_volatility_and_regime(san)
    figs.append(plot_volatility(bbva, "BBVA.MC"))
    figs.append(plot_volatility(san, "SAN.MC"))

    # Rolling correlation
    figs.append(correlation_rolling_plot(bbva, san))

    # Save enriched datasets
    enriched_paths = {}
    for df, name in [(bbva, "BBVA.MC"), (san, "SAN.MC")]:
        out_path = PROC_DIR / f"{name}_enriched.csv"
        df.to_csv(out_path, index=True)
        enriched_paths[name] = str(out_path)
        logging.info(f"[OUTPUT] Generated enriched dataset: {out_path}")

    # Log lineage
    inputs = {
        "bbva_raw": str(RAW_DIR / "BBVA.MC.csv"),
        "san_raw": str(RAW_DIR / "SAN.MC.csv"),
    }
    outputs = {
        "enriched": enriched_paths,
        "figures": {f"fig_{i}": p for i, p in enumerate(figs) if p},
    }
    params = {
        "period": args.period,
        "symbols": SYMBOLS,
        "volatility_window": 30,
        "rolling_corr_window": 60,
        "year_ticks_start": "2000-01-01",
        "year_ticks_end": "2025-12-31",
    }
    log_lineage(step="eda.run_eda", params=params, inputs=inputs, outputs=outputs)
    logging.info("[LOGS] Logged EDA lineage record")

    logging.info("[COMPLETED] EDA completed (FIGURES ONLY + LINEAGE).")


if __name__ == "__main__":
    main()
