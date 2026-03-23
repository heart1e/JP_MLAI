"""Fetch SEC XBRL facts (yearly or quarterly) for one or more tickers."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import os
import re
import urllib.request
from typing import Any, Dict, Iterable
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline_config.json"
with open(CONFIG_PATH, "r", encoding = "utf-8") as f:
    PIPELINE_CONFIG = json.load(f)

DEFAULT_TICKERS = PIPELINE_CONFIG["tickers"].get("universe", ["AAPL"])
DEFAULT_TICKER_CIK = PIPELINE_CONFIG["tickers"].get("sec_cik_overrides", {})
FIELDS = PIPELINE_CONFIG["fetch"]["sec"]["fields"]
ALLOWED_FORMS_YEARLY = set(PIPELINE_CONFIG["fetch"]["sec"]["allowed_forms"]["yearly"])
ALLOWED_FORMS_QUARTERLY = set(PIPELINE_CONFIG["fetch"]["sec"]["allowed_forms"].get("quarterly", []))
DEFAULT_YEARS = int(PIPELINE_CONFIG["fetch"]["sec"].get("years_default", 10))
DEFAULT_SEC_DIR_YEARLY = f"{PIPELINE_CONFIG['paths'].get('sec_raw_dir', 'data/sec')}/yearly"
DEFAULT_SEC_DIR_QUARTERLY = f"{PIPELINE_CONFIG['paths'].get('sec_raw_dir', 'data/sec')}/quarterly"

FULL_YEAR_FRAME = re.compile(r"^(?:CY|FY)(\d{4})$")
YEAR_END_FRAME = re.compile(r"^(?:CY|FY)(\d{4})Q4$")
QUARTER_FRAME = re.compile(r"^(?:CY|FY)(\d{4})Q([1-4])$")

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def normalize_cik(value: str) -> str:
    """Normalize CIK to a zero-padded 10-digit string."""

    digits = re.sub(r"\D+", "", value)


    return digits.zfill(10)


def normalize_sec_ticker(raw: str) -> str:
    """Normalize ticker symbols for SEC mappings."""

    value = raw.strip().upper()
    value = value.replace("-", ".")
    value = re.sub(r"\s+", ".", value)


    return value


def get_user_agent(cli_value: str | None) -> str:
    """Resolve the SEC User-Agent from CLI or environment defaults."""

    if cli_value:

        return cli_value

    env_value = os.environ.get("SEC_USER_AGENT")

    if env_value:

        return env_value


    return "ORION (contact: unavailable)"


def fetch_json(url: str, user_agent: str) -> Dict[str, Any]:
    """Fetch JSON from a URL with SEC-friendly headers."""

    req = urllib.request.Request(
        url,
        headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
    )

    with urllib.request.urlopen(req, timeout = 30) as resp:
        raw = resp.read()
        encoding = resp.headers.get("Content-Encoding", "")

        if encoding.lower() == "gzip":
            raw = gzip.decompress(raw)


        return json.loads(raw.decode("utf-8"))


def fetch_company_facts(cik: str, user_agent: str) -> Dict[str, Any]:
    """Fetch SEC XBRL company facts for a given CIK."""

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    return fetch_json(url, user_agent)


def fetch_ticker_cik_map(user_agent: str) -> Dict[str, str]:
    """Fetch the SEC ticker-to-CIK mapping."""

    data = fetch_json(SEC_TICKER_URL, user_agent)
    mapping = {}

    for _, entry in data.items():
        ticker = str(entry.get("ticker", "")).upper()
        cik = normalize_cik(str(entry.get("cik_str", "")))

        if ticker and cik:
            mapping[ticker] = cik


    return mapping


def pick_tag(facts: Dict[str, Any], tags: list[str]) -> str | None:
    """Return the first available tag from a list of candidates."""

    for tag in tags:

        if tag in facts:

            return tag


    return None


def quarter_end_date(year: int, quarter: int) -> datetime:
    """Return the calendar quarter end date."""
    month = quarter * 3
    day = 31 if month in (3, 12) else 30

    return datetime(year, month, day)


def parse_end_date(entry: Dict[str, Any], year: int | None, quarter: int | None) -> datetime | None:
    """Return a period end date from entry metadata or fallbacks."""

    end = entry.get("end")

    if end:

        try:

            return pd.to_datetime(end).to_pydatetime()

        except Exception:
            pass

    if year is None:

        return None

    if quarter is None:

        return datetime(year, 12, 31)

    return quarter_end_date(year, quarter)


def extract_yearly_series(fact: Dict[str, Any]) -> Dict[datetime, float]:
    """Extract a yearly series from XBRL fact units."""

    units = fact.get("units", {})

    if not units:

        return {}

    unit_key = "USD" if "USD" in units else next(iter(units))
    entries = units.get(unit_key, [])

    by_year: Dict[int, Dict[str, Any]] = {}

    for entry in entries:
        form = entry.get("form")
        fy = entry.get("fy")
        val = entry.get("val")
        frame = entry.get("frame") or ""

        if form not in ALLOWED_FORMS_YEARLY:

            continue

        if val is None:

            continue

        year = None
        priority = 0

        if frame:
            match = FULL_YEAR_FRAME.match(frame)

            if match:
                year = int(match.group(1))
                priority = 2

            else:
                match_q4 = YEAR_END_FRAME.match(frame)

                if match_q4:
                    year = int(match_q4.group(1))
                    priority = 1

                elif "Q" in frame:

                    continue

        if year is None:

            if fy is None:

                continue

            year = int(fy)

        end_dt = parse_end_date(entry, year, None)
        filed = entry.get("filed") or ""
        key = (priority, filed, end_dt or datetime(year, 12, 31))

        prev = by_year.get(year)

        if prev is None or key > prev["key"]:
            by_year[year] = {"val": val, "key": key, "end": end_dt}

    series: Dict[datetime, float] = {}

    for year, info in by_year.items():
        end_dt = info["end"] or datetime(year, 12, 31)
        series[end_dt] = info["val"]


    return series


def extract_quarterly_series(fact: Dict[str, Any]) -> Dict[datetime, float]:
    """Extract a quarterly series from XBRL fact units."""

    units = fact.get("units", {})

    if not units:

        return {}

    unit_key = "USD" if "USD" in units else next(iter(units))
    entries = units.get(unit_key, [])

    by_period: Dict[tuple[int, int], Dict[str, Any]] = {}

    for entry in entries:
        form = entry.get("form")
        fy = entry.get("fy")
        fp = entry.get("fp") or ""
        val = entry.get("val")
        frame = entry.get("frame") or ""

        if form not in ALLOWED_FORMS_QUARTERLY:

            continue

        if val is None:

            continue

        year = None
        quarter = None

        if frame:
            match = QUARTER_FRAME.match(frame)

            if match:
                year = int(match.group(1))
                quarter = int(match.group(2))

        if year is None and fy is not None and fp.startswith("Q"):

            try:
                year = int(fy)
                quarter = int(fp.replace("Q", ""))

            except ValueError:

                continue

        if year is None or quarter is None:

            continue

        end_dt = parse_end_date(entry, year, quarter)
        filed = entry.get("filed") or ""
        key = (filed, end_dt or quarter_end_date(year, quarter))

        period_key = (year, quarter)
        prev = by_period.get(period_key)

        if prev is None or key > prev["key"]:
            by_period[period_key] = {"val": val, "key": key, "end": end_dt}

    series: Dict[datetime, float] = {}

    for (year, quarter), info in by_period.items():
        end_dt = info["end"] or quarter_end_date(year, quarter)
        series[end_dt] = info["val"]


    return series


def build_dataframe_yearly(facts: Dict[str, Any], years: int) -> pd.DataFrame:
    """Build a wide DataFrame with annual XBRL fields."""

    data: Dict[str, Dict[datetime, float]] = {}
    all_dates: set[datetime] = set()

    for field, tags in FIELDS.items():
        tag = pick_tag(facts, tags)

        if tag is None:

            continue

        series = extract_yearly_series(facts[tag])

        if not series:

            continue

        data[field] = series
        all_dates.update(series.keys())

    if not all_dates:

        raise RuntimeError("No annual data found in company facts.")

    dates = sorted(all_dates)

    if years > 0:
        years_keep = sorted({d.year for d in dates})[-years:]
        dates = [d for d in dates if d.year in years_keep]

    df = pd.DataFrame(index = pd.to_datetime(dates))

    for field, series in data.items():
        df[field] = [series.get(d) for d in df.index.to_pydatetime()]

    df.index.name = "period_end"


    return df.sort_index()


def build_dataframe_quarterly(facts: Dict[str, Any], quarters: int) -> pd.DataFrame:
    """Build a wide DataFrame with quarterly XBRL fields."""
    data: Dict[str, Dict[datetime, float]] = {}
    all_dates: set[datetime] = set()

    for field, tags in FIELDS.items():
        tag = pick_tag(facts, tags)

        if tag is None:

            continue

        series = extract_quarterly_series(facts[tag])

        if not series:

            continue

        data[field] = series
        all_dates.update(series.keys())

    if not all_dates:

        raise RuntimeError("No quarterly data found in company facts.")

    dates = sorted(all_dates)

    if quarters > 0:
        dates = dates[-quarters:]

    df = pd.DataFrame(index = pd.to_datetime(dates))

    for field, series in data.items():
        df[field] = [series.get(d) for d in df.index.to_pydatetime()]

    df.index.name = "period_end"


    return df.sort_index()


def resolve_cik(
    ticker: str,
    user_agent: str,
    cik_map: Dict[str, str] | None
) -> str:
    """Resolve a CIK for a ticker using config and SEC mappings."""

    if ticker in DEFAULT_TICKER_CIK:

        return normalize_cik(DEFAULT_TICKER_CIK[ticker])

    if cik_map is not None:

        if ticker in cik_map:

            return cik_map[ticker]

        alt = ticker.replace(".", "-")

        if alt in cik_map:

            return cik_map[alt]

        alt_flat = ticker.replace(".", "")

        if alt_flat in cik_map:

            return cik_map[alt_flat]


    return ""


def iter_tickers(values: Iterable[str]) -> list[str]:
    """Normalize tickers for SEC lookup."""

    return [normalize_sec_ticker(v) for v in values]


def refresh_ticker_facts(
    ticker: str,
    freq: str = "both",
    years: int = DEFAULT_YEARS,
    quarters: int = 0,
    user_agent: str | None = None,
    data_dir_yearly: Path | None = None,
    data_dir_quarterly: Path | None = None,
    cik_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Fetch and persist SEC XBRL facts for a single ticker."""

    normalized = normalize_sec_ticker(ticker)
    target_yearly = data_dir_yearly or (Path(__file__).resolve().parents[1] / DEFAULT_SEC_DIR_YEARLY)
    target_quarterly = data_dir_quarterly or (Path(__file__).resolve().parents[1] / DEFAULT_SEC_DIR_QUARTERLY)
    target_yearly.mkdir(parents = True, exist_ok = True)
    target_quarterly.mkdir(parents = True, exist_ok = True)

    resolved_agent = get_user_agent(user_agent)
    cik = resolve_cik(normalized, resolved_agent, cik_map)

    if not cik:
        raise RuntimeError(f"Missing CIK for {normalized}.")

    company = fetch_company_facts(cik, resolved_agent)
    facts = company.get("facts", {}).get("us-gaap", {})
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    result: Dict[str, Any] = {
        "ticker": normalized,
        "slug": slug,
        "cik": cik,
        "status": "ok",
        "yearly_rows": 0,
        "quarterly_rows": 0,
        "paths": [],
    }

    if freq in ("yearly", "both"):
        df_yearly = build_dataframe_yearly(facts, years)
        out_yearly = target_yearly / f"sec_{slug}_yearly.csv"
        df_yearly.to_csv(out_yearly)
        result["yearly_rows"] = int(df_yearly.shape[0])
        result["paths"].append(str(out_yearly))

    if freq in ("quarterly", "both"):

        if not ALLOWED_FORMS_QUARTERLY:
            raise RuntimeError("Quarterly forms list is empty. Update pipeline_config.json.")

        df_quarterly = build_dataframe_quarterly(facts, quarters)
        out_quarterly = target_quarterly / f"sec_{slug}_quarterly.csv"
        df_quarterly.to_csv(out_quarterly)
        result["quarterly_rows"] = int(df_quarterly.shape[0])
        result["paths"].append(str(out_quarterly))

    return result


def main() -> int:
    """Run the CLI workflow for downloading SEC facts."""
    parser = argparse.ArgumentParser(description = "Fetch SEC XBRL company facts.")
    parser.add_argument("--tickers", nargs = "*", default = DEFAULT_TICKERS, help = "Ticker list.")
    parser.add_argument(
        "--freq",
        choices = ["yearly", "quarterly", "both"],
        default = "both",
        help = "Frequency to fetch."
    )
    parser.add_argument("--years", type = int, default = DEFAULT_YEARS, help = "Years to keep (0 = all).")
    parser.add_argument("--quarters", type = int, default = 0, help = "Quarters to keep (0 = all).")
    parser.add_argument("--user-agent", default = "", help = "SEC User-Agent header.")
    parser.add_argument(
        "--data-dir-yearly",
        default = str(Path(__file__).resolve().parents[1] / DEFAULT_SEC_DIR_YEARLY),
        help = "Output directory for yearly data."
    )
    parser.add_argument(
        "--data-dir-quarterly",
        default = str(Path(__file__).resolve().parents[1] / DEFAULT_SEC_DIR_QUARTERLY),
        help = "Output directory for quarterly data."
    )

    args = parser.parse_args()
    tickers = iter_tickers(args.tickers)
    user_agent = get_user_agent(args.user_agent or None)

    cik_map = None

    if any(t not in DEFAULT_TICKER_CIK for t in tickers):
        cik_map = fetch_ticker_cik_map(user_agent)

    yearly_dir = Path(args.data_dir_yearly)
    quarterly_dir = Path(args.data_dir_quarterly)
    yearly_dir.mkdir(parents = True, exist_ok = True)
    quarterly_dir.mkdir(parents = True, exist_ok = True)

    for ticker in tickers:
        cik = resolve_cik(ticker, user_agent, cik_map)

        if not cik:
            print(f"{ticker}: missing CIK, skipped")

            continue

        company = fetch_company_facts(cik, user_agent)
        facts = company.get("facts", {}).get("us-gaap", {})

        slug = re.sub(r"[^a-z0-9]+", "_", ticker.lower()).strip("_")

        if args.freq in ("yearly", "both"):
            df_yearly = build_dataframe_yearly(facts, args.years)
            out_yearly = yearly_dir / f"sec_{slug}_yearly.csv"
            df_yearly.to_csv(out_yearly)
            print(f"{ticker} yearly: {df_yearly.shape[0]} rows -> {out_yearly}")

        if args.freq in ("quarterly", "both"):

            if not ALLOWED_FORMS_QUARTERLY:

                raise RuntimeError("Quarterly forms list is empty. Update pipeline_config.json.")
            df_quarterly = build_dataframe_quarterly(facts, args.quarters)
            out_quarterly = quarterly_dir / f"sec_{slug}_quarterly.csv"
            df_quarterly.to_csv(out_quarterly)
            print(f"{ticker} quarterly: {df_quarterly.shape[0]} rows -> {out_quarterly}")


    return 0


if __name__ == "__main__":

    raise SystemExit(main())
