"""Fetch balance sheet and income statement data from Yahoo Finance."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time
from typing import Iterable, Tuple

import pandas as pd
import yfinance as yf

SPECIAL_TICKERS = {
    "700": "0700.HK",
    "1810": "1810.HK",
    "9633": "9633.HK",
    "9987": "9987.HK",
    "9988": "9988.HK",
    "BRK B": "BRK-B",
    "NESN": "NESN.SW",
}

DEFAULT_TICKERS = [
    "AAPL",
    "GOOG",
    "700",
    "1810",
    "IBM",
    "TSLA",
    "9633",
    "9987",
    "9988",
    "IBKR",
    "KO",
    "MCD",
    "EL",
    "BRK B",
    "NESN",
]


def normalize_ticker(raw: str) -> str:
    """Normalize tickers and apply project-specific mappings."""
    value = raw.strip()
    upper = value.upper()

    if upper in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[upper]

    if re.fullmatch(r"\d+", value):
        return value.zfill(4) + ".HK"

    return upper


def slugify(ticker: str) -> str:
    """Create a filesystem-friendly slug from a ticker."""
    return re.sub(r"[^a-z0-9]+", "_", ticker.lower()).strip("_")


def fetch_statements(ticker: str, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch balance sheet and income statement for a ticker."""
    tkr = yf.Ticker(ticker)
    bs = tkr.get_balance_sheet(freq = freq)
    is_df = tkr.get_financials(freq = freq)

    if bs is None or is_df is None:
        return pd.DataFrame(), pd.DataFrame()

    bs = bs.T.sort_index()
    is_df = is_df.T.sort_index()

    return bs, is_df


def save_statements(
    bs: pd.DataFrame,
    is_df: pd.DataFrame,
    data_dir: Path,
    slug: str,
    freq: str
) -> None:
    """Save balance sheet and income statement to CSV files."""
    freq_dir = data_dir / freq
    freq_dir.mkdir(parents = True, exist_ok = True)
    bs_path = freq_dir / f"{slug}_balance_sheet_{freq}.csv"
    is_path = freq_dir / f"{slug}_income_statement_{freq}.csv"

    bs.to_csv(bs_path)
    is_df.to_csv(is_path)


def needs_fetch(data_dir: Path, slug: str, freq: str) -> bool:
    """Return True when cached files are missing for a ticker/frequency."""
    freq_dir = data_dir / freq
    bs_path = freq_dir / f"{slug}_balance_sheet_{freq}.csv"
    is_path = freq_dir / f"{slug}_income_statement_{freq}.csv"

    return not (bs_path.exists() and is_path.exists())


def iter_tickers(values: Iterable[str]) -> list[str]:
    """Normalize all tickers from an iterable."""
    return [normalize_ticker(v) for v in values]


def main() -> int:
    """Run the CLI workflow for fetching statements."""
    parser = argparse.ArgumentParser(description = "Fetch statements from yfinance.")
    parser.add_argument(
        "--tickers",
        nargs = "*",
        default = DEFAULT_TICKERS,
        help = "Ticker list (default is the project list)."
    )
    parser.add_argument(
        "--data-dir",
        default = str(Path(__file__).resolve().parents[1] / "data" / "yfinance"),
        help = "Output data directory (yearly/quarterly subfolders will be created)."
    )
    parser.add_argument(
        "--force",
        action = "store_true",
        help = "Refetch even if cached files exist."
    )
    parser.add_argument(
        "--sleep",
        type = float,
        default = 0.5,
        help = "Sleep seconds between tickers."
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents = True, exist_ok = True)

    tickers = iter_tickers(args.tickers)
    results = []

    for ticker in tickers:
        slug = slugify(ticker)
        yearly_needed = args.force or needs_fetch(data_dir, slug, "yearly")
        quarterly_needed = args.force or needs_fetch(data_dir, slug, "quarterly")

        if not yearly_needed and not quarterly_needed:
            print(f"{ticker}: cached")
            results.append((ticker, "cached", 0, 0))
            continue

        try:
            # Fetch data only for missing frequencies to avoid redundant calls.
            bs_y, is_y = fetch_statements(ticker, "yearly") if yearly_needed else (pd.DataFrame(), pd.DataFrame())
            bs_q, is_q = fetch_statements(ticker, "quarterly") if quarterly_needed else (pd.DataFrame(), pd.DataFrame())

            if yearly_needed and not bs_y.empty and not is_y.empty:
                save_statements(bs_y, is_y, data_dir, slug, "yearly")

            if quarterly_needed and not bs_q.empty and not is_q.empty:
                save_statements(bs_q, is_q, data_dir, slug, "quarterly")

            print(
                f"{ticker}: yearly={bs_y.shape[0]} rows, quarterly={bs_q.shape[0]} rows"
            )
            results.append((ticker, "ok", bs_y.shape[0], bs_q.shape[0]))

        except Exception as exc:
            print(f"{ticker}: failed ({exc})")
            results.append((ticker, "failed", 0, 0))

        # Gentle throttle to respect data source rate limits.
        time.sleep(max(args.sleep, 0.0))

    ok = sum(1 for _, status, _, _ in results if status == "ok")
    cached = sum(1 for _, status, _, _ in results if status == "cached")
    failed = sum(1 for _, status, _, _ in results if status == "failed")
    print(f"Done. ok={ok}, cached={cached}, failed={failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
