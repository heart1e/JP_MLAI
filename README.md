# JPM MLCOE TSRL 2026 Q1 (Notebook Runner)

This repository contains the TensorFlow/TensorFlow Probability implementation for Question 1. The primary workflow is the notebook at `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb`, which loads Yahoo Finance statements, engineers features, trains the constrained model, and reports results.

## Repo Layout

- `Question_1/notebooks/` : main notebook
- `Question_1/config/model_config.json` : tickers, frequency, seed, column aliases
- `Question_1/data/yfinance/` : cached statements (generated)
- `Question_1/scripts/fetch_yf_statements.py` : optional prefetch script

## Requirements

- Python 3.10+ (recommended)
- TensorFlow, TensorFlow Probability
- See `requirements.txt` or `environment.yml`

## Install (pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Install (conda)

```bash
conda env create -f environment.yml
conda activate jp_mlai_q1
```

## Data Download (optional but faster)

The notebook will fetch data on demand if cache files are missing. You can prefetch all tickers to the cache directory:

```bash
python Question_1/scripts/fetch_yf_statements.py
```

By default, the script writes to `Question_1/data/yfinance/{yearly,quarterly}/`.

## Run the Notebook

```bash
jupyter lab
```

Open `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb` and run all cells.

## Outputs

- Cached statement CSVs: `Question_1/data/yfinance/{yearly,quarterly}/*.csv`
- Results and plots: rendered inside the notebook (no files are saved by default)

## Configuration (model_config.json)

Key settings used by the notebook:

- `seed`: 3
- `target_freq`: `yearly`
- `tickers`:
  - AAPL, GOOG, 700, 1810, IBM, TSLA, 9633, 9987, 9988, IBKR, KO, MCD, EL, BRK B, NESN
- `special_tickers` mapping:
  - 700 -> 0700.HK
  - 1810 -> 1810.HK
  - 9633 -> 9633.HK
  - 9987 -> 9987.HK
  - 9988 -> 9988.HK
  - BRK B -> BRK-B
  - NESN -> NESN.SW
- `data_dir`: `data/yfinance`

Adjust `Question_1/config/model_config.json` to change frequency, tickers, seed, or data paths.

## Notes

- Internet access is required to download Yahoo Finance statements.
- Deterministic TensorFlow ops are enabled when available; some platforms may still show minor nondeterminism.
