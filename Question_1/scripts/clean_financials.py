"""Clean and standardize financial statement CSVs across sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = ROOT_DIR / "config" / "model_config.json"
DATA_CONFIG_PATH = ROOT_DIR / "config" / "data_config.json"

with open(MODEL_CONFIG_PATH, "r", encoding = "utf-8-sig") as f:
    MODEL_CONFIG = json.load(f)

with open(DATA_CONFIG_PATH, "r", encoding = "utf-8-sig") as f:
    DATA_CONFIG = json.load(f)

BS_ALIASES = MODEL_CONFIG["bs_aliases"]
IS_ALIASES = {
    "Total Revenue": [
        "Total Revenue",
        "Operating Revenue",
        "TotalRevenue",
        "OperatingRevenue",
        "Revenues",
        "total_revenue",
    ],
    "Cost Of Revenue": [
        "Cost Of Revenue",
        "Cost of Revenue",
        "CostOfRevenue",
        "ReconciledCostOfRevenue",
        "cost_of_revenue",
    ],
    "Operating Income": [
        "Operating Income",
        "OperatingIncome",
        "OperatingIncomeLoss",
    ],
    "Net Income": [
        "Net Income",
        "NetIncome",
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "net_income",
    ],
}
SEC_CANON_MAP = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities",
    "total_equity": "Total Equity",
    "total_revenue": "Total Revenue",
    "net_income": "Net Income",
}
SEC_BS_COLS = ["Total Assets", "Total Liabilities", "Total Equity"]
SEC_IS_COLS = ["Total Revenue", "Net Income"]


def load_statement(path: Path) -> pd.DataFrame:
    """Load a statement CSV and normalize the index to datetime."""
    df = pd.read_csv(path, index_col = 0)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, errors = "coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def select_columns(df: pd.DataFrame, alias_map: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """Select the first available column for each canonical field."""
    if df.empty:
        return df
    out = pd.DataFrame(index = df.index)
    for canon, options in alias_map.items():
        for name in options:
            if name in df.columns:
                out[canon] = pd.to_numeric(df[name], errors = "coerce")
                break
    return out


def write_clean(df: pd.DataFrame, out_path: Path) -> None:
    """Write cleaned data to disk."""
    out_path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(out_path)


def clean_yfinance(freq: str) -> None:
    """Clean YFinance statements for a given frequency."""
    src_dir = ROOT_DIR / "data" / "yfinance" / freq
    out_dir = ROOT_DIR / "data" / "clean" / "yfinance" / freq

    for path in sorted(src_dir.glob("*.csv")):
        name = path.name
        if "balance_sheet" in name:
            df = load_statement(path)
            cleaned = select_columns(df, BS_ALIASES)
            write_clean(cleaned, out_dir / name)
        elif "income_statement" in name:
            df = load_statement(path)
            cleaned = select_columns(df, IS_ALIASES)
            write_clean(cleaned, out_dir / name)


def clean_sec(freq: str) -> None:
    """Clean SEC statements for a given frequency."""
    src_dir = ROOT_DIR / "data" / "sec" / freq
    out_dir = ROOT_DIR / "data" / "clean" / "sec" / freq

    for path in sorted(src_dir.glob("sec_*_*.csv")):
        df = load_statement(path)
        if df.empty:
            continue
        slug = path.stem.replace("sec_", "")

        df = df.rename(columns = SEC_CANON_MAP)

        bs = df.reindex(columns = [c for c in SEC_BS_COLS if c in df.columns])
        is_df = df.reindex(columns = [c for c in SEC_IS_COLS if c in df.columns])

        if not bs.empty:
            write_clean(bs, out_dir / f"{slug}_balance_sheet_{freq}.csv")
        if not is_df.empty:
            write_clean(is_df, out_dir / f"{slug}_income_statement_{freq}.csv")


def main() -> int:
    """Run cleaning for all sources and frequencies."""
    for freq in ("yearly", "quarterly"):
        clean_yfinance(freq)
        clean_sec(freq)
    print("Cleaned data written to data/clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
