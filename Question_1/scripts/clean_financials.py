"""Clean and standardize financial statement CSVs across sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG_PATH = ROOT_DIR / "config" / "pipeline_config.json"

with open(PIPELINE_CONFIG_PATH, "r", encoding = "utf-8") as f:
    PIPELINE_CONFIG = json.load(f)

BS_ALIASES = PIPELINE_CONFIG["training"]["aliases"]["balance_sheet"]
IS_ALIASES = PIPELINE_CONFIG["training"]["aliases"]["income_statement"]
YFINANCE_RAW_DIR = ROOT_DIR / PIPELINE_CONFIG["paths"].get("yfinance_raw_dir", "data/yfinance")
YFINANCE_CLEAN_DIR = ROOT_DIR / PIPELINE_CONFIG["paths"].get("yfinance_clean_dir", "data/clean/yfinance")
SEC_RAW_DIR = ROOT_DIR / PIPELINE_CONFIG["paths"].get("sec_raw_dir", "data/sec")
SEC_CLEAN_DIR = ROOT_DIR / PIPELINE_CONFIG["paths"].get("sec_clean_dir", "data/clean/sec")

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
    valid_index_mask = df.index.notna()
    df = df.loc[valid_index_mask].sort_index()


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


def clean_yfinance_slug(slug: str, freq: str) -> list[Path]:
    """Clean one Yahoo Finance ticker slug for a given frequency."""

    src_dir = YFINANCE_RAW_DIR / freq
    out_dir = YFINANCE_CLEAN_DIR / freq
    written: list[Path] = []

    for stem in ("balance_sheet", "income_statement"):
        path = src_dir / f"{slug}_{stem}_{freq}.csv"

        if not path.exists():

            continue

        if stem == "balance_sheet":
            cleaned = select_columns(load_statement(path), BS_ALIASES)
        else:
            cleaned = select_columns(load_statement(path), IS_ALIASES)

        out_path = out_dir / path.name
        write_clean(cleaned, out_path)
        written.append(out_path)

    return written


def clean_sec_slug(slug: str, freq: str) -> list[Path]:
    """Clean one SEC ticker slug for a given frequency."""

    src_path = SEC_RAW_DIR / freq / f"sec_{slug}_{freq}.csv"
    out_dir = SEC_CLEAN_DIR / freq

    if not src_path.exists():

        return []

    df = load_statement(src_path)

    if df.empty:

        return []

    df = df.rename(columns = SEC_CANON_MAP)
    written: list[Path] = []
    bs = df.reindex(columns = [c for c in SEC_BS_COLS if c in df.columns])
    is_df = df.reindex(columns = [c for c in SEC_IS_COLS if c in df.columns])

    if not bs.empty:
        bs_path = out_dir / f"{slug}_balance_sheet_{freq}.csv"
        write_clean(bs, bs_path)
        written.append(bs_path)

    if not is_df.empty:
        is_path = out_dir / f"{slug}_income_statement_{freq}.csv"
        write_clean(is_df, is_path)
        written.append(is_path)

    return written


def clean_yfinance(freq: str) -> None:
    """Clean YFinance statements for a given frequency."""

    src_dir = YFINANCE_RAW_DIR / freq
    slugs = set()

    for path in sorted(src_dir.glob("*_balance_sheet_*.csv")):
        slugs.add(path.name.replace(f"_balance_sheet_{freq}.csv", ""))

    for path in sorted(src_dir.glob("*_income_statement_*.csv")):
        slugs.add(path.name.replace(f"_income_statement_{freq}.csv", ""))

    for slug in sorted(slugs):
        clean_yfinance_slug(slug, freq)


def clean_sec(freq: str) -> None:
    """Clean SEC statements for a given frequency."""

    src_dir = SEC_RAW_DIR / freq

    for path in sorted(src_dir.glob("sec_*_*.csv")):
        slug = path.stem.replace("sec_", "")
        clean_sec_slug(slug.replace(f"_{freq}", ""), freq)


def main() -> int:
    """Run cleaning for all sources and frequencies."""

    for freq in ("yearly", "quarterly"):
        clean_yfinance(freq)
        clean_sec(freq)

    print("Cleaned data written to data/clean.")


    return 0


if __name__ == "__main__":

    raise SystemExit(main())
