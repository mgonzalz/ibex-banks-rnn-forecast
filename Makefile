# =========================
# Project-wide Makefile
# =========================

# --- General settings ---
SHELL := /bin/sh
.ONESHELL:
.DEFAULT_GOAL := help

# Main directories
SRC_DIRS := src tests
NB_DIR  := notebooks

# Export PYTHONPATH to the current project directory
export PYTHONPATH := $(CURDIR)

# Shortcut to poetry
POETRY := poetry


# --- Makefile targets ---
## Show this help
help:
	@awk 'BEGIN {FS":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
	' $(MAKEFILE_LIST)

## Create/update the virtualenv and install deps (incl. dev group)
env:
	$(POETRY) install --with dev --no-root

## Download/refresh raw OHLCV under .cache/raw/*
data-raw:
	$(POETRY) run python -m src.data.load_raw

## Validate raw CSV integrity => .cache/validation/raw_integrity_summary.csv
data-validate:
	$(POETRY) run python -m src.data.validate_integrity

## Run exploratory data analysis (EDA) for given period (default: 252 days)
run-eda:
	$(POETRY) run python -m src.eda.run_eda --period 252

## Build exogenous datasets (events + macro) and store under .cache/exogenous/
build-exogenous:
	$(POETRY) run python -m src.data.build_exogenous

## Auto-fix imports, formatting and lint issues (isort -> black -> ruff --fix)
fix:
	$(POETRY) run isort $(SRC_DIRS)
	$(POETRY) run black $(SRC_DIRS)
	$(POETRY) run ruff check $(SRC_DIRS) --fix

## Run static checks without modifying files (ruff + flake8)
lint:
	$(POETRY) run ruff check $(SRC_DIRS)
	$(POETRY) run flake8 $(SRC_DIRS)

## Format notebooks with nbqa (isort + black + ruff)
nb-fix:
	$(POETRY) run nbqa isort $(NB_DIR)
	$(POETRY) run nbqa black $(NB_DIR)
	$(POETRY) run nbqa ruff $(NB_DIR) -- --fix
