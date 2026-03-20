"""Part 2 annual-report extraction helpers for Question 1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd
import requests

try:
    from pypdf import PdfReader

except Exception as exc:  # pragma: no cover - import guard for notebook envs

    raise ImportError("Please install pypdf first: pip install pypdf") from exc


def resolve_question_root(question_root: Path | None = None) -> Path:
    """Resolve the Question_1 workspace root."""

    if question_root is not None:

        return question_root.resolve()

    return Path(__file__).resolve().parents[1]


NUMBER_TOKEN_RE = re.compile(r'\(?\d[\d,]*\)?')


def _download_if_missing(url: str, pdf_path: Path) -> None:
    """Download source PDF when not available locally.

    Try urllib first, then fallback to requests with browser headers."""

    pdf_path.parent.mkdir(parents = True, exist_ok = True)

    if pdf_path.exists():

        return

    try:
        urlretrieve(url, pdf_path)

        return

    except Exception:
        pass

    import requests

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers = headers, timeout = 90, verify = False)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)


def _parse_number_token(token: str) -> float:
    """Parse tokens like '12,345' or '(147)' into float."""

    negative = token.startswith('(') and token.endswith(')')
    core = token.strip('()').replace(',', '')
    value = float(core)

    return -value if negative else value


def _extract_amount(line: str, trailing_columns: int) -> float:
    """Extract first value from trailing report year columns."""

    cleaned = re.sub(r'\(Note[^)]*\)', '', line, flags = re.IGNORECASE)
    cleaned = cleaned.replace('$', ' ')
    tokens = NUMBER_TOKEN_RE.findall(cleaned)
    values = [_parse_number_token(tok) for tok in tokens]

    if len(values) < trailing_columns:

        raise ValueError(f'Cannot parse amount from line: {line}')

    return values[-trailing_columns]


def _normalized_lines(text: str) -> list[str]:
    """Return non-empty stripped lines."""

    return [line.strip() for line in text.splitlines() if line.strip()]


def _find_line(lines: list[str], phrase: str) -> str:
    """Find first line containing phrase (case-insensitive)."""

    target = phrase.lower()

    for line in lines:

        if target in line.lower():

            return line

    raise KeyError(f'Line not found for phrase: {phrase}')


def _find_statement_page(reader: PdfReader, predicate, page_name: str) -> tuple[int, str]:
    """Find first PDF page matching predicate(text)."""

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ''

        if predicate(text):

            return idx, text

    raise RuntimeError(f'Could not locate {page_name} page in PDF.')


def _parse_income_statement(page_text: str) -> dict[str, float]:
    """Parse selected GM income statement fields (2023 column)."""

    lines = _normalized_lines(page_text)

    return {

        'total_net_sales_revenue': _extract_amount(_find_line(lines, 'Total net sales and revenue'), 3),
        'total_costs_expenses': _extract_amount(_find_line(lines, 'Total costs and expenses'), 3),
        'operating_income': _extract_amount(_find_line(lines, 'Operating income (loss)'), 3),
        'interest_expense_automotive': _extract_amount(_find_line(lines, 'Automotive interest expense'), 3),
        'income_tax_expense': _extract_amount(_find_line(lines, 'Income tax expense (benefit)'), 3),
        'net_income': _extract_amount(_find_line(lines, 'Net income (loss) $'), 3),
        'net_income_attributable_to_stockholders': _extract_amount(
            _find_line(lines, 'Net income (loss) attributable to stockholders'),
            3,
        ),
    }


def _parse_balance_sheet(page_text: str) -> dict[str, float]:
    """Parse selected GM balance sheet fields (2023 column)."""

    lines = _normalized_lines(page_text)
    result = {
        'cash_and_cash_equivalents': _extract_amount(_find_line(lines, 'Cash and cash equivalents'), 2),
        'marketable_debt_securities': _extract_amount(_find_line(lines, 'Marketable debt securities'), 2),
        'accounts_notes_receivable': _extract_amount(_find_line(lines, 'Accounts and notes receivable'), 2),
        'inventories': _extract_amount(_find_line(lines, 'Inventories'), 2),
        'total_current_liabilities': _extract_amount(_find_line(lines, 'Total current liabilities'), 2),
        'total_assets': _extract_amount(_find_line(lines, 'Total Assets'), 2),
        'total_liabilities': _extract_amount(_find_line(lines, 'Total Liabilities'), 2),
        'total_equity': _extract_amount(_find_line(lines, 'Total Equity'), 2),
    }

    context = None

    for line in lines:
        lower = line.lower()

        if 'short-term debt and current portion of long-term debt' in lower:
            context = 'short_term'

            continue

        if 'long-term debt' in lower and 'short-term' not in lower:
            context = 'long_term'

            continue

        if context and lower.startswith('automotive'):
            result[f'{context}_debt_automotive'] = _extract_amount(line, 2)

            continue

        if context and lower.startswith('gm financial'):
            result[f'{context}_debt_gm_financial'] = _extract_amount(line, 2)

            if context == 'short_term':
                context = None

    required_debt = [
        'short_term_debt_automotive',
        'short_term_debt_gm_financial',
        'long_term_debt_automotive',
        'long_term_debt_gm_financial',
    ]

    missing = [k for k in required_debt if k not in result]

    if missing:

        raise RuntimeError(f'Missing debt components: {missing}')

    return result


def _parse_cash_flow(page_text: str) -> dict[str, float]:
    """Parse depreciation components for EBITDA proxy."""

    lines = _normalized_lines(page_text)

    return {

        'depr_impairment_operating_leases': _extract_amount(
            _find_line(lines, 'Depreciation and impairment of Equipment on operating leases'),
            3,
        ),
        'depr_amort_impairment_property': _extract_amount(
            _find_line(lines, 'Depreciation, amortization and impairment charges on Property'),
            3,
        ),
    }


def _compute_metrics(income: dict[str, float], balance: dict[str, float], cash_flow: dict[str, float]) -> dict[str, float]:
    """Compute core Part 2 ratios with explicit assumptions."""

    total_debt = (
        balance['short_term_debt_automotive']
        + balance['short_term_debt_gm_financial']
        + balance['long_term_debt_automotive']
        + balance['long_term_debt_gm_financial']
    )

    ebitda_proxy = (
        income['operating_income']
        + cash_flow['depr_impairment_operating_leases']
        + cash_flow['depr_amort_impairment_property']
    )

    quick_assets = (
        balance['cash_and_cash_equivalents']
        + balance['marketable_debt_securities']
        + balance['accounts_notes_receivable']
    )

    return {

        'net_income_musd': income['net_income'],
        'cost_to_income_ratio': income['total_costs_expenses'] / income['total_net_sales_revenue'],
        'quick_ratio': quick_assets / balance['total_current_liabilities'],
        'debt_to_equity_ratio': total_debt / balance['total_equity'],
        'debt_to_assets_ratio': total_debt / balance['total_assets'],
        'debt_to_capital_ratio': total_debt / (total_debt + balance['total_equity']),
        'debt_to_ebitda_ratio': total_debt / ebitda_proxy,
        'interest_coverage_ratio': income['operating_income'] / income['interest_expense_automotive'],
        'total_debt_musd': total_debt,
        'ebitda_proxy_musd': ebitda_proxy,
    }


def _normalize_ocr_numeric_spacing(text: str) -> str:
    """Fix common PDF OCR number splits such as '67, 517' or '7,7 74'."""

    fixed = re.sub(r'(?<=\d,)\s+(?=\d{3}\b)', '', text)
    fixed = re.sub(r'(?<=\d,\d)\s+(?=\d{2,3}\b)', '', fixed)
    fixed = re.sub(r'\s+', ' ', fixed)

    return fixed.strip()


def _normalized_lines_ocr(text: str) -> list[str]:
    """Return non-empty lines with OCR spacing normalized."""

    return [_normalize_ocr_numeric_spacing(line) for line in text.splitlines() if line.strip()]


def _extract_amount_ocr(line: str, trailing_columns: int) -> float:
    """Extract current-year value from line with OCR-cleaning pre-step."""

    cleaned = _normalize_ocr_numeric_spacing(line)
    cleaned = cleaned.replace('$', ' ')
    tokens = NUMBER_TOKEN_RE.findall(cleaned)
    values = [_parse_number_token(tok) for tok in tokens]

    if len(values) < trailing_columns:

        raise ValueError(f'Cannot parse amount from line: {line}')

    return values[-trailing_columns]


def _find_line_ocr(lines: list[str], phrase: str) -> str:
    """Find first line containing phrase (case-insensitive)."""

    target = phrase.lower()

    for line in lines:

        if target in line.lower():

            return line

    raise KeyError(f'Line not found for phrase: {phrase}')


def _parse_lvmh_income_statement(page_text: str) -> dict[str, float]:
    """Parse selected 2024 values from LVMH consolidated income statement."""

    lines = _normalized_lines_ocr(page_text)

    return {

        'revenue': _extract_amount_ocr(_find_line_ocr(lines, 'Revenue'), 3),
        'cost_of_sales': _extract_amount_ocr(_find_line_ocr(lines, 'Cost of sales'), 3),
        'marketing_selling_expenses': _extract_amount_ocr(_find_line_ocr(lines, 'Marketing and selling expenses'), 3),
        'general_admin_expenses': _extract_amount_ocr(_find_line_ocr(lines, 'General and administrative expenses'), 3),
        'operating_profit': _extract_amount_ocr(_find_line_ocr(lines, 'Operating profit'), 3),
        'cost_of_net_financial_debt': _extract_amount_ocr(_find_line_ocr(lines, 'Cost of net financial debt'), 3),
        'interest_on_lease_liabilities': _extract_amount_ocr(_find_line_ocr(lines, 'Interest on lease liabilities'), 3),
        'net_profit_before_minority_interests': _extract_amount_ocr(_find_line_ocr(lines, 'Net profit before minority interests'), 3),
        'net_profit_group_share': _extract_amount_ocr(_find_line_ocr(lines, 'Net profit, Group share'), 3),
    }


def _parse_lvmh_balance_sheet(page_text: str) -> dict[str, float]:
    """Parse selected 2024 values from LVMH consolidated balance sheet."""

    lines = _normalized_lines_ocr(page_text)

    return {

        'cash_and_cash_equivalents': _extract_amount_ocr(_find_line_ocr(lines, 'Cash and cash equivalents'), 3),
        'trade_accounts_receivable': _extract_amount_ocr(_find_line_ocr(lines, 'Trade accounts receivable'), 3),
        'current_liabilities': _extract_amount_ocr(_find_line_ocr(lines, 'Current liabilities'), 3),
        'total_assets': _extract_amount_ocr(_find_line_ocr(lines, 'Total assets'), 3),
        'equity_total': _extract_amount_ocr(_find_line_ocr(lines, 'Equity '), 3),
        'short_term_borrowings': _extract_amount_ocr(_find_line_ocr(lines, 'Short - term borrowings'), 3),
        'long_term_borrowings': _extract_amount_ocr(_find_line_ocr(lines, 'Long - term borrowings'), 3),
    }


def _parse_lvmh_cash_flow(page_text: str) -> dict[str, float]:
    """Parse selected 2024 values from LVMH consolidated cash flow statement."""

    lines = _normalized_lines_ocr(page_text)

    return {

        'dep_amort_provisions_increase': _extract_amount_ocr(
            _find_line_ocr(lines, 'Net increase in depreciation, amortization and provisions'),
            3,
        ),
        'depr_right_of_use_assets': _extract_amount_ocr(
            _find_line_ocr(lines, 'Depreciation of right - of - use assets'),
            3,
        ),
    }


def _compute_lvmh_metrics(income: dict[str, float], balance: dict[str, float], cash_flow: dict[str, float]) -> dict[str, float]:
    """Compute requested ratios for LVMH with explicit assumptions."""

    total_debt = balance['short_term_borrowings'] + balance['long_term_borrowings']
    quick_assets = balance['cash_and_cash_equivalents'] + balance['trade_accounts_receivable']
    cost_to_income = (
        abs(income['cost_of_sales'])
        + abs(income['marketing_selling_expenses'])
        + abs(income['general_admin_expenses'])
    ) / income['revenue']
    ebitda_proxy = (
        income['operating_profit']
        + cash_flow['dep_amort_provisions_increase']
        + cash_flow['depr_right_of_use_assets']
    )
    interest_expense_proxy = abs(income['cost_of_net_financial_debt']) + abs(income['interest_on_lease_liabilities'])

    return {

        'net_income_musd': income['net_profit_before_minority_interests'],
        'net_income_group_share_musd': income['net_profit_group_share'],
        'cost_to_income_ratio': cost_to_income,
        'quick_ratio': quick_assets / balance['current_liabilities'],
        'debt_to_equity_ratio': total_debt / balance['equity_total'],
        'debt_to_assets_ratio': total_debt / balance['total_assets'],
        'debt_to_capital_ratio': total_debt / (total_debt + balance['equity_total']),
        'debt_to_ebitda_ratio': total_debt / ebitda_proxy,
        'interest_coverage_ratio': income['operating_profit'] / interest_expense_proxy,
        'total_debt_musd': total_debt,
        'ebitda_proxy_musd': ebitda_proxy,
    }


SEC_HEADERS = {'User-Agent':'Mozilla/5.0 (compatible; ORION/1.0; +test@example.com)','Accept':'application/pdf,text/html,*/*'}

def _ensure_file(url: str, path: Path, headers = None, verify = True):

    path.parent.mkdir(parents = True, exist_ok = True)

    if path.exists():

        return

    r = requests.get(url, headers = headers or {'User-Agent':'Mozilla/5.0'}, timeout = 120, verify = verify)
    r.raise_for_status(); path.write_bytes(r.content)


def _first_line(lines, token: str) -> str:

    for line in lines:

        if token in line:

            return line

    raise KeyError(token)


def _extract_lookahead(lines, token: str, trailing_cols: int, lookahead: int = 4) -> float:

    t = token.lower()
    for i,line in enumerate(lines):

        if t in line.lower():

            try:

                return _extract_amount(' '.join(lines[i:i+lookahead]), trailing_cols)

            except Exception:
                pass

    raise KeyError(token)


def _sec_facts(cik: str) -> dict:

    u = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'

    return requests.get(u, headers = SEC_HEADERS, timeout = 120).json()['facts']['us-gaap']


def _pick_sec(facts: dict, tag: str, fy: int, end_date: str) -> float:

    arr = facts.get(tag,{}).get('units',{}).get('USD',[])
    pts = [p for p in arr if p.get('form')=='10-K' and p.get('fy')==fy]
    pts = [p for p in pts if p.get('end')==end_date] or pts

    if not pts:

        raise KeyError(f'{tag} FY{fy} {end_date}')
    pts.sort(key = lambda p:(p.get('start') or '', p.get('end') or '', p.get('filed') or ''))

    return float(pts[-1]['val'])


def extract_gm_pilot(question_root: Path | None = None) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    GM_PDF_URL = 'https://investor.gm.com/static-files/1fff6f59-551f-4fe0-bca9-74bfc9a56aeb'
    GM_PDF_PATH = PROJECT_ROOT / 'Question_1' / 'data' / 'raw' / 'gm_annual_report_2023.pdf'
    GM_OUTPUT_PATH = PROJECT_ROOT / 'Question_1' / 'output' / 'gm_2023_mini_metrics.json'
    # Execute extraction and compute metrics.
    _download_if_missing(GM_PDF_URL, GM_PDF_PATH)
    reader = PdfReader(str(GM_PDF_PATH))

    income_page_idx, income_text = _find_statement_page(
        reader,
        lambda t: (
            'CONSOLIDATED INCOME STATEMENTS' in t
            and 'Total net sales and revenue' in t
            and 'GENERAL MOTORS COMPANY AND SUBSIDIARIES' in t
        ),
        'income statement',
    )

    balance_page_idx, balance_text = _find_statement_page(
        reader,
        lambda t: (
            'CONSOLIDATED BALANCE SHEETS' in t
            and 'LIABILITIES AND EQUITY' in t
            and 'Total Assets' in t
        ),
        'balance sheet',
    )

    cash_flow_page_idx, cash_flow_text = _find_statement_page(
        reader,
        lambda t: (
            'CONSOLIDATED STATEMENTS OF CASH FLOWS' in t
            and 'Depreciation and impairment of Equipment on operating leases' in t
        ),
        'cash flow statement',
    )

    income_vals = _parse_income_statement(income_text)
    balance_vals = _parse_balance_sheet(balance_text)
    cash_flow_vals = _parse_cash_flow(cash_flow_text)
    metrics = _compute_metrics(income_vals, balance_vals, cash_flow_vals)

    part2_output = {
        'company': 'General Motors Company',
        'statement_year': 2023,
        'units': 'USD millions unless ratio',
        'source_pdf_url': GM_PDF_URL,
        'source_pdf_path': str(GM_PDF_PATH.as_posix()),
        'statement_pages_pdf_1_based': {
            'income_statement': income_page_idx + 1,
            'balance_sheet': balance_page_idx + 1,
            'cash_flow_statement': cash_flow_page_idx + 1,
        },
        'extracted_values': {
            'income_statement': income_vals,
            'balance_sheet': balance_vals,
            'cash_flow_statement': cash_flow_vals,
        },
        'metrics': {k: round(v, 6) for k, v in metrics.items()},
        'assumptions': [
            'Cost-to-income = Total costs and expenses / Total net sales and revenue.',
            'Quick assets = cash + marketable debt securities + accounts and notes receivable.',
            'Total debt = short-term debt + long-term debt (Automotive + GM Financial).',
            'EBITDA proxy = operating income + depreciation/impairment of operating leases + depreciation/amortization/impairment of property.',
            'Interest coverage = operating income / automotive interest expense.',
        ],
    }

    GM_OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    GM_OUTPUT_PATH.write_text(json.dumps(part2_output, indent = 2), encoding = 'utf-8')

    print('Saved:', GM_OUTPUT_PATH)
    print('Income / Balance / Cash Flow pages:', income_page_idx + 1, balance_page_idx + 1, cash_flow_page_idx + 1)

    pd.DataFrame([
        {
            'metric': k,
            'value': v,
        }
        for k, v in part2_output['metrics'].items()
    ])

    return part2_output


def extract_lvmh_extension(question_root: Path | None = None) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    LVMH_PDF_URL = 'https://lvmh-com.cdn.prismic.io/lvmh-com/Z5kVBpbqstJ999KR_Financialdocuments-December31%2C2024.pdf'
    LVMH_PDF_PATH = PROJECT_ROOT / 'Question_1' / 'data' / 'raw' / 'lvmh_financial_documents_2024.pdf'
    LVMH_OUTPUT_PATH = PROJECT_ROOT / 'Question_1' / 'output' / 'lvmh_2024_mini_metrics.json'
    _download_if_missing(LVMH_PDF_URL, LVMH_PDF_PATH)
    lvmh_reader = PdfReader(str(LVMH_PDF_PATH))

    lvmh_income_page_idx, lvmh_income_text = _find_statement_page(
        lvmh_reader,
        lambda t: (
            'CONSOLIDATED INCOME STATEMENT' in t
            and 'Net profit before minority interests' in t
            and 'Financial Documents - December 31, 2024' in t
        ),
        'LVMH income statement',
    )

    lvmh_balance_page_idx, lvmh_balance_text = _find_statement_page(
        lvmh_reader,
        lambda t: (
            'CONSOLIDATED BALANCE SHEET' in t
            and 'Total assets' in t
            and 'Total liabilities and equity' in t
        ),
        'LVMH balance sheet',
    )

    lvmh_cash_flow_page_idx, lvmh_cash_flow_text = _find_statement_page(
        lvmh_reader,
        lambda t: (
            'CONSOLIDATED CASH FLOW STATEMENT' in t
            and 'Net increase in depreciation, amortization and provisions' in t
        ),
        'LVMH cash flow statement',
    )

    lvmh_income_vals = _parse_lvmh_income_statement(lvmh_income_text)
    lvmh_balance_vals = _parse_lvmh_balance_sheet(lvmh_balance_text)
    lvmh_cash_flow_vals = _parse_lvmh_cash_flow(lvmh_cash_flow_text)
    lvmh_metrics = _compute_lvmh_metrics(lvmh_income_vals, lvmh_balance_vals, lvmh_cash_flow_vals)

    lvmh_output = {
        'company': 'LVMH Moet Hennessy Louis Vuitton SE',
        'statement_year': 2024,
        'units': 'EUR millions unless ratio',
        'source_pdf_url': LVMH_PDF_URL,
        'source_pdf_path': str(LVMH_PDF_PATH.as_posix()),
        'statement_pages_pdf_1_based': {
            'income_statement': lvmh_income_page_idx + 1,
            'balance_sheet': lvmh_balance_page_idx + 1,
            'cash_flow_statement': lvmh_cash_flow_page_idx + 1,
        },
        'extracted_values': {
            'income_statement': lvmh_income_vals,
            'balance_sheet': lvmh_balance_vals,
            'cash_flow_statement': lvmh_cash_flow_vals,
        },
        'metrics': {k: round(v, 6) for k, v in lvmh_metrics.items()},
        'assumptions': [
            'Cost-to-income = (Cost of sales + Marketing and selling expenses + General and administrative expenses) / Revenue.',
            'Quick assets = cash and cash equivalents + trade accounts receivable.',
            'Total debt = short-term borrowings + long-term borrowings (borrowings only, lease liabilities excluded).',
            'EBITDA proxy = operating profit + net increase in depreciation/amortization/provisions + depreciation of right-of-use assets.',
            'Interest coverage = operating profit / (abs(cost of net financial debt) + abs(interest on lease liabilities)).',
        ],
    }

    LVMH_OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    LVMH_OUTPUT_PATH.write_text(json.dumps(lvmh_output, indent = 2), encoding = 'utf-8')

    print('Saved:', LVMH_OUTPUT_PATH)
    print('LVMH Income / Balance / Cash Flow pages:', lvmh_income_page_idx + 1, lvmh_balance_page_idx + 1, lvmh_cash_flow_page_idx + 1)

    pd.DataFrame([
        {
            'metric': k,
            'value': v,
        }
        for k, v in lvmh_output['metrics'].items()
    ])

    return lvmh_output


def build_cross_summary(
    question_root: Path | None = None,
    payload_overrides: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    CROSS_OUTPUT_PATH = PROJECT_ROOT / 'Question_1' / 'output' / 'mini_cross_company_metrics.json'

    overrides = payload_overrides or {}
    output_map = {
        'GM_2023': PROJECT_ROOT / 'Question_1' / 'output' / 'gm_2023_mini_metrics.json',
        'LVMH_2024': PROJECT_ROOT / 'Question_1' / 'output' / 'lvmh_2024_mini_metrics.json',
        'JPM_2024': PROJECT_ROOT / 'Question_1' / 'output' / 'jpm_2024_mini_metrics.json',
    }

    payload_map = {}
    for label, path in output_map.items():

        if label in overrides:
            payload_map[label] = overrides[label]

        elif path.exists():
            payload_map[label] = json.loads(path.read_text(encoding = 'utf-8'))

        else:
            payload_map[label] = None

    cross_payload = {
        'gm': payload_map['GM_2023'],
        'lvmh': payload_map['LVMH_2024'],
        'jpm': payload_map['JPM_2024'],
    }
    CROSS_OUTPUT_PATH.write_text(json.dumps(cross_payload, indent = 2), encoding = 'utf-8')

    rows = []
    for label, payload in payload_map.items():

        if payload is None:

            continue

        metrics = payload.get('metrics', {})
        rows.append({
            'company': label,
            'units': payload.get('units'),
            'net_income': metrics.get('net_income_musd'),
            'cost_to_income': metrics.get('cost_to_income_ratio'),
            'quick_ratio': metrics.get('quick_ratio'),
            'debt_to_equity': metrics.get('debt_to_equity_ratio'),
            'debt_to_assets': metrics.get('debt_to_assets_ratio'),
            'debt_to_capital': metrics.get('debt_to_capital_ratio'),
            'debt_to_ebitda': metrics.get('debt_to_ebitda_ratio'),
            'interest_coverage': metrics.get('interest_coverage_ratio'),
        })

    return {

        'cross_payload': cross_payload,
        'summary': pd.DataFrame(rows),
        'output_path': CROSS_OUTPUT_PATH,
    }


def validate_manifest_outputs(question_root: Path | None = None) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    MANIFEST_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'annual_report_manifest_2026q1.json'
    CONTRACT_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'extraction_output_contract_2026q1.json'
    # Validate extraction outputs against manifest + contract.
    MANIFEST_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'annual_report_manifest_2026q1.json'
    CONTRACT_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'extraction_output_contract_2026q1.json'

    manifest_cfg = json.loads(MANIFEST_PATH.read_text(encoding = 'utf-8'))
    contract_cfg = json.loads(CONTRACT_PATH.read_text(encoding = 'utf-8'))


    def _type_ok(value, kind: str) -> bool:
        """Check simplified type labels used by contract."""

        if kind == 'string':

            return isinstance(value, str)

        if kind == 'number':

            return isinstance(value, (int, float))

        if kind == 'object':

            return isinstance(value, dict)

        if kind == 'array':

            return isinstance(value, list)

        return False


    def _validate_payload(payload: dict, contract: dict, output_path: Path) -> list[str]:
        """Return validation errors for one output JSON payload."""

        errors = []

        for key in contract['required_top_level_keys']:

            if key not in payload:
                errors.append(f"missing top-level key: {key}")

        for key, expected in contract['types'].items():

            if key in payload and not _type_ok(payload[key], expected):
                errors.append(f"wrong type for '{key}': expected {expected}, got {type(payload[key]).__name__}")

        pages = payload.get('statement_pages_pdf_1_based', {})

        if isinstance(pages, dict):

            for key in contract['required_statement_page_keys']:

                if key not in pages:
                    errors.append(f"missing page key: {key}")

                elif not isinstance(pages[key], (int, float)):
                    errors.append(f"page key not numeric: {key}")

        extracted = payload.get('extracted_values', {})

        if isinstance(extracted, dict):

            for key in contract['required_extracted_sections']:

                if key not in extracted:
                    errors.append(f"missing extracted section: {key}")

                elif not isinstance(extracted[key], dict):
                    errors.append(f"extracted section is not object: {key}")

        metrics = payload.get('metrics', {})

        if isinstance(metrics, dict):

            for key in contract['required_metric_keys']:

                if key not in metrics:
                    errors.append(f"missing metric: {key}")

                elif not isinstance(metrics[key], (int, float)):
                    errors.append(f"metric not numeric: {key}")

        pdf_path_raw = payload.get('source_pdf_path')

        if isinstance(pdf_path_raw, str) and pdf_path_raw:
            pdf_path = Path(pdf_path_raw)

            if not pdf_path.is_absolute():
                pdf_path = PROJECT_ROOT / pdf_path_raw

            if not pdf_path.exists():
                errors.append(f"source_pdf_path missing: {pdf_path}")

        if not output_path.exists():
            errors.append(f"output missing: {output_path}")

        return errors


    validation_rows = []
    all_errors = []

    for item in manifest_cfg['companies']:
        output_path = PROJECT_ROOT / item['output_path']

        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding = 'utf-8'))
            errs = _validate_payload(payload, contract_cfg, output_path)

        else:
            errs = [f"output missing: {output_path}"]

        validation_rows.append(
            {
                'id': item['id'],
                'company': item['company_name'],
                'manifest_status': item['status'],
                'output_exists': output_path.exists(),
                'error_count': len(errs),
                'validation_passed': len(errs) == 0,
            }
        )

        for err in errs:
            all_errors.append({'id': item['id'], 'error': err})

    validation_df = pd.DataFrame(validation_rows).sort_values(['validation_passed', 'id'], ascending = [False, True])
    errors_df = pd.DataFrame(all_errors)

    print('Validation summary:')
    print(validation_df[['id', 'validation_passed', 'error_count']].to_string(index = False))

    validation_df

    return {

        'manifest_cfg': manifest_cfg,
        'contract_cfg': contract_cfg,
        'validation_df': validation_df,
        'errors_df': errors_df,
    }


def refresh_manifest_progress(
    question_root: Path | None = None,
    auto_download_from_manifest: bool = True,
    write_manifest_status_back: bool = True,
) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    MANIFEST_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'annual_report_manifest_2026q1.json'
    manifest_cfg = json.loads(MANIFEST_PATH.read_text(encoding = 'utf-8'))
    AUTO_DOWNLOAD_FROM_MANIFEST = auto_download_from_manifest
    WRITE_MANIFEST_STATUS_BACK = write_manifest_status_back
    # Manifest-driven download/status refresh.
    AUTO_DOWNLOAD_FROM_MANIFEST = True
    WRITE_MANIFEST_STATUS_BACK = True


    def _refresh_company_status(item: dict, root: Path) -> str:
        """Infer status from file presence."""

        pdf_path = root / item['pdf_path'] if item.get('pdf_path') else None
        out_path = root / item['output_path'] if item.get('output_path') else None

        has_pdf = bool(pdf_path) and pdf_path.exists()
        has_output = bool(out_path) and out_path.exists()

        if has_pdf and has_output:

            return 'complete'

        if has_pdf and not has_output:

            return 'in_progress'

        if item.get('pdf_url'):

            return 'pending'

        return 'blocked'


    download_rows = []

    for item in manifest_cfg['companies']:
        required_now = bool(item.get('required_now', False))
        pdf_url = (item.get('pdf_url') or '').strip()
        pdf_path_str = (item.get('pdf_path') or '').strip()

        if not required_now:

            continue

        if not pdf_url or not pdf_path_str:
            download_rows.append(
                {
                    'id': item['id'],
                    'company': item['company_name'],
                    'action': 'skip_missing_url_or_path',
                    'status_after': item.get('status'),
                }
            )

            continue

        pdf_path = PROJECT_ROOT / pdf_path_str
        pdf_path.parent.mkdir(parents = True, exist_ok = True)

        downloaded = False

        if AUTO_DOWNLOAD_FROM_MANIFEST and not pdf_path.exists():

            try:
                _download_if_missing(pdf_url, pdf_path)
                downloaded = pdf_path.exists()
                action = 'downloaded' if downloaded else 'download_skipped'

            except Exception as exc:
                action = f'download_failed: {exc}'

        else:
            action = 'already_exists' if pdf_path.exists() else 'skipped'

        new_status = _refresh_company_status(item, PROJECT_ROOT)

        if isinstance(action, str) and action.startswith('download_failed'):
            new_status = 'blocked'

        item['status'] = new_status

        download_rows.append(
            {
                'id': item['id'],
                'company': item['company_name'],
                'action': action,
                'downloaded': downloaded,
                'status_after': new_status,
                'pdf_exists': pdf_path.exists(),
            }
        )

    if WRITE_MANIFEST_STATUS_BACK:
        MANIFEST_PATH.write_text(json.dumps(manifest_cfg, indent = 2), encoding = 'utf-8')

    manifest_progress_df = pd.DataFrame(download_rows)
    manifest_progress_df

    return {

        'manifest_cfg': manifest_cfg,
        'manifest_progress_df': manifest_progress_df,
    }


def extract_jpm_pilot(question_root: Path | None = None) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    MANIFEST_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'annual_report_manifest_2026q1.json'
    # JPM 2024 extraction (bank-specific metric proxies).
    JPM_PDF_PATH = PROJECT_ROOT / 'Question_1' / 'data' / 'raw' / 'jpm_annual_report_2024.pdf'
    JPM_OUTPUT_PATH = PROJECT_ROOT / 'Question_1' / 'output' / 'jpm_2024_mini_metrics.json'

    jpm_reader = PdfReader(str(JPM_PDF_PATH))

    jpm_income_page_idx, jpm_income_text = _find_statement_page(
        jpm_reader,
        lambda t: ('Consolidated statements of income' in t and 'Total net revenue' in t),
        'JPM income statement',
    )

    jpm_balance_page_idx, jpm_balance_text = _find_statement_page(
        jpm_reader,
        lambda t: ('Consolidated balance sheets' in t and 'Total liabilities' in t and 'Total stockholders' in t),
        'JPM balance sheet',
    )

    jpm_cashflow_page_idx, jpm_cashflow_text = _find_statement_page(
        jpm_reader,
        lambda t: ('Consolidated statements of cash flows' in t and 'Depreciation and amortization' in t),
        'JPM cash flow statement',
    )

    jpm_income_lines = _normalized_lines(jpm_income_text)
    jpm_balance_lines = _normalized_lines(jpm_balance_text)
    jpm_cashflow_lines = _normalized_lines(jpm_cashflow_text)

    def _extract_amount_with_lookahead(lines: list[str], phrase: str, trailing_columns: int, lookahead: int = 3) -> float:
        """Extract amount when value may appear on following lines due table wrapping."""

        target = phrase.lower()

        for idx, line in enumerate(lines):

            if target in line.lower():
                window = ' '.join(lines[idx:idx + lookahead])

                try:

                    return _extract_amount(window, trailing_columns)

                except Exception:

                    continue

        raise KeyError(f'Could not parse phrase: {phrase}')


    def _extract_first_large_amount_with_lookahead(lines: list[str], phrase: str, lookahead: int = 4, min_abs: float = 1_000.0) -> float:
        """Extract first large numeric value in a phrase window to avoid note-number contamination."""

        target = phrase.lower()

        for idx, line in enumerate(lines):

            if target in line.lower():
                window = ' '.join(lines[idx:idx + lookahead])
                cleaned = re.sub(r'\(Note[^)]*\)', '', window, flags = re.IGNORECASE)
                cleaned = cleaned.replace('$', ' ')
                values = [_parse_number_token(tok) for tok in NUMBER_TOKEN_RE.findall(cleaned)]
                large_values = [v for v in values if abs(v) >= min_abs]

                if large_values:

                    return large_values[0]

        raise KeyError(f'Could not parse large amount for phrase: {phrase}')


    jpm_income_vals = {
        'total_net_revenue': _extract_amount(_find_line(jpm_income_lines, 'Total net revenue'), 3),
        'interest_expense': _extract_amount(_find_line(jpm_income_lines, 'Interest expense'), 3),
        'total_noninterest_expense': _extract_amount(_find_line(jpm_income_lines, 'Total noninterest expense'), 3),
        'income_before_tax': _extract_amount(_find_line(jpm_income_lines, 'Income before income tax expense'), 3),
        'net_income': _extract_amount(_find_line(jpm_income_lines, 'Net income $'), 3),
    }

    jpm_balance_vals = {
        'cash_due_from_banks': _extract_amount(_find_line(jpm_balance_lines, 'Cash and due from banks'), 2),
        'deposits_with_banks': _extract_amount(_find_line(jpm_balance_lines, 'Deposits with banks'), 2),
        'fed_funds_sold_and_resale': _extract_amount(_find_line(jpm_balance_lines, 'Federal funds sold and securities purchased under resale agreements'), 2),
        'total_assets': _extract_first_large_amount_with_lookahead(jpm_balance_lines, 'Total assets'),
        'total_liabilities': _extract_first_large_amount_with_lookahead(jpm_balance_lines, 'Total liabilities(a)'),
        'total_stockholders_equity': _extract_amount(_find_line(jpm_balance_lines, 'Total stockholders'), 2),
    }

    jpm_cashflow_vals = {
        'depreciation_and_amortization': _extract_amount(_find_line(jpm_cashflow_lines, 'Depreciation and amortization'), 3),
    }

    # Bank-specific proxy definitions to stay compatible with common output contract.
    jpm_total_debt_proxy = jpm_balance_vals['total_liabilities']
    jpm_quick_assets_proxy = (
        jpm_balance_vals['cash_due_from_banks']
        + jpm_balance_vals['deposits_with_banks']
        + jpm_balance_vals['fed_funds_sold_and_resale']
    )
    jpm_ebitda_proxy = jpm_income_vals['income_before_tax'] + jpm_cashflow_vals['depreciation_and_amortization']

    jpm_metrics = {
        'net_income_musd': jpm_income_vals['net_income'],
        'cost_to_income_ratio': jpm_income_vals['total_noninterest_expense'] / jpm_income_vals['total_net_revenue'],
        'quick_ratio': jpm_quick_assets_proxy / jpm_balance_vals['total_liabilities'],
        'debt_to_equity_ratio': jpm_total_debt_proxy / jpm_balance_vals['total_stockholders_equity'],
        'debt_to_assets_ratio': jpm_total_debt_proxy / jpm_balance_vals['total_assets'],
        'debt_to_capital_ratio': jpm_total_debt_proxy / (jpm_total_debt_proxy + jpm_balance_vals['total_stockholders_equity']),
        'debt_to_ebitda_ratio': jpm_total_debt_proxy / jpm_ebitda_proxy,
        'interest_coverage_ratio': jpm_income_vals['income_before_tax'] / jpm_income_vals['interest_expense'],
        'total_debt_musd': jpm_total_debt_proxy,
        'ebitda_proxy_musd': jpm_ebitda_proxy,
    }

    jpm_output = {
        'company': 'JPMorgan Chase & Co.',
        'statement_year': 2024,
        'units': 'USD millions unless ratio',
        'source_pdf_url': 'https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf',
        'source_pdf_path': str(JPM_PDF_PATH.as_posix()),
        'statement_pages_pdf_1_based': {
            'income_statement': jpm_income_page_idx + 1,
            'balance_sheet': jpm_balance_page_idx + 1,
            'cash_flow_statement': jpm_cashflow_page_idx + 1,
        },
        'extracted_values': {
            'income_statement': jpm_income_vals,
            'balance_sheet': jpm_balance_vals,
            'cash_flow_statement': jpm_cashflow_vals,
        },
        'metrics': {k: round(v, 6) for k, v in jpm_metrics.items()},
        'assumptions': [
            'Bank format proxy: total liabilities is used as debt proxy.',
            'Cost-to-income (bank efficiency proxy) = total noninterest expense / total net revenue.',
            'Quick ratio proxy = (cash and due from banks + deposits with banks + federal funds sold/resale) / total liabilities.',
            'EBITDA proxy = income before income tax expense + depreciation and amortization.',
            'Interest coverage proxy = income before income tax expense / interest expense.',
        ],
    }

    JPM_OUTPUT_PATH.write_text(json.dumps(jpm_output, indent = 2), encoding = 'utf-8')
    print('Saved:', JPM_OUTPUT_PATH)
    print('JPM Income / Balance / Cash Flow pages:', jpm_income_page_idx + 1, jpm_balance_page_idx + 1, jpm_cashflow_page_idx + 1)

    # Update manifest status for JPM when output exists.
    manifest_live = json.loads(MANIFEST_PATH.read_text(encoding = 'utf-8'))
    for entry in manifest_live['companies']:

        if entry['id'] == 'jpm_2024':
            entry['status'] = 'complete'

            break

    MANIFEST_PATH.write_text(json.dumps(manifest_live, indent = 2), encoding = 'utf-8')

    pd.DataFrame([
        {'metric': k, 'value': v}
        for k, v in jpm_output['metrics'].items()
    ])

    return jpm_output


def extract_full_required_coverage(question_root: Path | None = None) -> dict[str, dict[str, Any]]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    # Tencent + Alibaba
    TENCENT_PDF_PATH = PROJECT_ROOT / 'Question_1' / 'data' / 'raw' / 'tencent_2024_annual_report.pdf'

    if not (TENCENT_PDF_PATH.exists() and TENCENT_PDF_PATH.read_bytes()[:4] == b'%PDF'):
        rc = requests.get('https://static.www.tencent.com/uploads/2025/04/08/1132b72b565389d1b913aea60a648d73.pdf',
            headers = {'User-Agent':'Mozilla/5.0','Accept':'application/pdf,*/*'},
            cookies = {'__tst_status':'699253921#','EO_Bot_Ssid':'1501757440'},timeout = 120,verify = False)
        rc.raise_for_status()

        if not rc.content.startswith(b'%PDF'):

            raise RuntimeError('Tencent anti-bot response not PDF')
        TENCENT_PDF_PATH.parent.mkdir(parents = True,exist_ok = True); TENCENT_PDF_PATH.write_bytes(rc.content)

    tr = PdfReader(str(TENCENT_PDF_PATH))
    ti = _normalized_lines(tr.pages[123].extract_text() or '')
    tb1 = _normalized_lines(tr.pages[125].extract_text() or '')
    tb2 = _normalized_lines(tr.pages[126].extract_text() or '')
    tb3 = _normalized_lines(tr.pages[127].extract_text() or '')
    tc = _normalized_lines(tr.pages[254].extract_text() or '')

    ti_vals = {
     'total_revenue': _extract_amount(_first_line(ti,'660,257'),2),
     'cost_of_revenue': _extract_amount(_find_line(ti,'Cost of revenues'),2),
     'operating_profit': _extract_amount(_find_line(ti,'Operating profit'),2),
     'finance_costs': _extract_amount(_find_line(ti,'Finance costs'),2),
     'profit_for_year': _extract_amount(_find_line(ti,'Profit for the year'),2),
    }
    tb_vals = {
     'cash_and_cash_equivalents': _extract_amount(_find_line(tb1,'Cash and cash equivalents'),2),
     'accounts_receivable': _extract_amount(_find_line(tb1,'Accounts receivable'),2),
     'inventories': _extract_amount(_find_line(tb1,'Inventories'),2),
     'total_current_liabilities_proxy': _extract_amount(_first_line(tb3,'396,909 352,157'),2),
     'total_assets': _extract_amount(_find_line(tb1,'Total assets'),2),
     'total_liabilities': _extract_amount(_find_line(tb3,'Total liabilities'),2),
     'total_equity': _extract_amount(_find_line(tb2,'Total equity'),2),
     'short_term_borrowings': _extract_amount(_find_line(tb3,'Borrowings 36 52,885 41,537'),2),
     'short_term_notes_payable': _extract_amount(_find_line(tb3,'Notes payable 37 8,623 14,161'),2),
     'long_term_borrowings': _extract_amount(_find_line(tb2,'Borrowings 36 146,521 155,819'),2),
     'long_term_notes_payable': _extract_amount(_find_line(tb2,'Notes payable 37 130,586 137,101'),2),
    }
    tc_vals = {
     'depreciation_proxy': _extract_lookahead(tc,'Depreciation of property, plant and equipment',2),
     'amortisation_proxy': _extract_amount(_find_line(tc,'Amortisation of intangible assets and land use rights'),2),
    }
    t_debt = tb_vals['short_term_borrowings']+tb_vals['short_term_notes_payable']+tb_vals['long_term_borrowings']+tb_vals['long_term_notes_payable']
    t_ebitda = ti_vals['operating_profit']+tc_vals['depreciation_proxy']+tc_vals['amortisation_proxy']

    TOUT = PROJECT_ROOT/'Question_1'/'output'/'tencent_2024_mini_metrics.json'
    TOUT.write_text(json.dumps({
     'company':'Tencent Holdings Limited','statement_year':2024,'units':'RMB millions unless ratio',
     'source_pdf_url':'https://static.www.tencent.com/uploads/2025/04/08/1132b72b565389d1b913aea60a648d73.pdf',
     'source_pdf_path':str(TENCENT_PDF_PATH.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':124,'balance_sheet':126,'cash_flow_statement':133},
     'extracted_values':{'income_statement':ti_vals,'balance_sheet':tb_vals,'cash_flow_statement':tc_vals},
     'metrics':{
       'net_income_musd':round(ti_vals['profit_for_year'],6),
       'cost_to_income_ratio':round(abs(ti_vals['cost_of_revenue'])/ti_vals['total_revenue'],6),
       'quick_ratio':round((tb_vals['cash_and_cash_equivalents']+tb_vals['accounts_receivable'])/tb_vals['total_current_liabilities_proxy'],6),
       'debt_to_equity_ratio':round(t_debt/tb_vals['total_equity'],6),
       'debt_to_assets_ratio':round(t_debt/tb_vals['total_assets'],6),
       'debt_to_capital_ratio':round(t_debt/(t_debt+tb_vals['total_equity']),6),
       'debt_to_ebitda_ratio':round(t_debt/t_ebitda,6),
       'interest_coverage_ratio':round(ti_vals['operating_profit']/abs(ti_vals['finance_costs']),6),
       'total_debt_musd':round(t_debt,6),'ebitda_proxy_musd':round(t_ebitda,6)
     },
     'assumptions':['Current liabilities uses subtotal line 396,909 in Tencent statement section.']
    },indent = 2),encoding = 'utf-8')

    ALI_PDF = PROJECT_ROOT/'Question_1'/'data'/'raw'/'alibaba_fy2024_annual_report.pdf'
    ar = PdfReader(str(ALI_PDF))
    ai = _normalized_lines(ar.pages[253].extract_text() or '')
    ab1 = _normalized_lines(ar.pages[255].extract_text() or '')
    ab2 = _normalized_lines(ar.pages[256].extract_text() or '')
    ac = _normalized_lines(ar.pages[260].extract_text() or '')
    ai_vals = {
     'total_revenue':_extract_amount(_find_line(ai,'Revenue 5, 22'),2),
     'cost_of_revenue':_extract_amount(_find_line(ai,'Cost of revenue 22'),2),
     'income_from_operations':_extract_amount(_find_line(ai,'Income from operations'),2),
     'interest_expense':_extract_amount(_find_line(ai,'Interest expense'),2),
     'net_income':_extract_amount(_find_line(ai,'Net income'),2),
    }
    ab_vals = {
     'cash_and_cash_equivalents':_extract_amount(_find_line(ab1,'Cash and cash equivalents'),2),
     'short_term_investments':_extract_amount(_find_line(ab1,'Short-term investments'),2),
     'prepayments_receivables_other_assets_current_proxy':_extract_amount(_find_line(ab1,'Prepayments, receivables and other assets 13 137,072 143,536'),2),
     'total_current_liabilities':_extract_amount(_find_line(ab1,'Total current liabilities'),2),
     'total_assets':_extract_amount(_find_line(ab1,'Total assets'),2),
     'total_liabilities':_extract_amount(_find_line(ab1,'Total liabilities'),2),
     'total_equity':_extract_amount(_find_line(ab2,'Total equity'),2),
     'current_bank_borrowings':_extract_amount(_find_line(ab1,'Current bank borrowings'),2),
     'current_unsecured_senior_notes':_extract_amount(_find_line(ab1,'Current unsecured senior notes'),2),
     'non_current_bank_borrowings':_extract_amount(_find_line(ab1,'Non-current bank borrowings'),2),
     'non_current_unsecured_senior_notes':_extract_amount(_find_line(ab1,'Non-current unsecured senior notes'),2),
    }
    ac_vals = {'depreciation_proxy':_extract_lookahead(ac,'Depreciation and impairment of property and',2),'amortization_proxy':_extract_lookahead(ac,'Amortization of intangible assets and licensed',2)}
    a_debt = ab_vals['current_bank_borrowings']+ab_vals['current_unsecured_senior_notes']+ab_vals['non_current_bank_borrowings']+ab_vals['non_current_unsecured_senior_notes']
    a_ebitda = ai_vals['income_from_operations']+ac_vals['depreciation_proxy']+ac_vals['amortization_proxy']
    AOUT = PROJECT_ROOT/'Question_1'/'output'/'alibaba_2024_mini_metrics.json'
    AOUT.write_text(json.dumps({
     'company':'Alibaba Group Holding Limited','statement_year':2024,'units':'RMB millions unless ratio',
     'source_pdf_url':'https://data.alibabagroup.com/ecms-files/1514443390/5788a02d-696c-412a-ad2a-386d19b21769/Alibaba%20Group%20Holding%20Limited%20Fiscal%20Year%202024%20Annual%20Report.pdf',
     'source_pdf_path':str(ALI_PDF.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':254,'balance_sheet':256,'cash_flow_statement':261},
     'extracted_values':{'income_statement':ai_vals,'balance_sheet':ab_vals,'cash_flow_statement':ac_vals},
     'metrics':{
       'net_income_musd':round(ai_vals['net_income'],6),
       'cost_to_income_ratio':round(abs(ai_vals['cost_of_revenue'])/ai_vals['total_revenue'],6),
       'quick_ratio':round((ab_vals['cash_and_cash_equivalents']+ab_vals['short_term_investments']+ab_vals['prepayments_receivables_other_assets_current_proxy'])/ab_vals['total_current_liabilities'],6),
       'debt_to_equity_ratio':round(a_debt/ab_vals['total_equity'],6),
       'debt_to_assets_ratio':round(a_debt/ab_vals['total_assets'],6),
       'debt_to_capital_ratio':round(a_debt/(a_debt+ab_vals['total_equity']),6),
       'debt_to_ebitda_ratio':round(a_debt/a_ebitda,6),
       'interest_coverage_ratio':round(ai_vals['income_from_operations']/abs(ai_vals['interest_expense']),6),
       'total_debt_musd':round(a_debt,6),'ebitda_proxy_musd':round(a_ebitda,6)
     },
     'assumptions':['Current receivable proxy uses prepayments/receivables/other assets line item in Alibaba report.']
    },indent = 2),encoding = 'utf-8')

    print('Saved:',TOUT) ; print('Saved:',AOUT)

    # Exxon + Volkswagen
    EXXON_URL = 'https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/0000034088-26-000045/0000034088-26-000045.pdf'
    EXXON_PDF = PROJECT_ROOT/'Question_1'/'data'/'raw'/'exxon_2024_10k.pdf'
    _ensure_file(EXXON_URL, EXXON_PDF, headers = {'User-Agent':'Mozilla/5.0'}, verify = False)

    er = PdfReader(str(EXXON_PDF))
    ei = _normalized_lines(er.pages[72].extract_text() or '')
    eb = _normalized_lines(er.pages[74].extract_text() or '')
    ec = _normalized_lines(er.pages[75].extract_text() or '')
    ei_vals = {
     'sales_and_operating_revenue':_extract_amount(_find_line(ei,'Sales and other operating revenue'),2),
     'total_costs_and_other_deductions':_extract_amount(_find_line(ei,'Total costs and other deductions'),2),
     'income_before_income_taxes':_extract_amount(_find_line(ei,'Income (loss) before income taxes'),2),
     'interest_expense':_extract_amount(_find_line(ei,'Interest expense'),2),
     'net_income_attributable_exxonmobil':_extract_amount(_find_line(ei,'Net income (loss) attributable to ExxonMobil'),2),
    }
    eb_vals = {
     'cash_and_cash_equivalents':_extract_amount(_find_line(eb,'Cash and cash equivalents  10,681 23,029'),1),
     'notes_and_accounts_receivable_net':_extract_amount(_find_line(eb,'Notes and accounts receivable'),1),
     'total_current_liabilities':_extract_amount(_find_line(eb,'Total current liabilities'),1),
     'total_assets':_extract_amount(_find_line(eb,'Total Assets'),1),
     'total_liabilities':_extract_amount(_find_line(eb,'Total Liabilities'),1),
     'total_equity':_extract_amount(_find_line(eb,'Total Equity'),1),
     'notes_and_loans_payable':_extract_amount(_find_line(eb,'Notes and loans payable'),1),
     'long_term_debt':_extract_amount(_find_line(eb,'Long-term debt'),1),
    }
    ec_vals = {'depreciation_and_depletion':_extract_amount(_find_line(ec,'Depreciation and depletion (includes impairments)'),2)}
    e_debt = eb_vals['notes_and_loans_payable']+eb_vals['long_term_debt']
    e_ebitda = ei_vals['income_before_income_taxes']+ec_vals['depreciation_and_depletion']
    EOUT = PROJECT_ROOT/'Question_1'/'output'/'exxon_2024_mini_metrics.json'
    EOUT.write_text(json.dumps({
     'company':'Exxon Mobil Corporation','statement_year':2024,'units':'USD millions unless ratio',
     'source_pdf_url':EXXON_URL,'source_pdf_path':str(EXXON_PDF.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':73,'balance_sheet':75,'cash_flow_statement':76},
     'extracted_values':{'income_statement':ei_vals,'balance_sheet':eb_vals,'cash_flow_statement':ec_vals},
     'metrics':{
      'net_income_musd':round(ei_vals['net_income_attributable_exxonmobil'],6),
      'cost_to_income_ratio':round(ei_vals['total_costs_and_other_deductions']/ei_vals['sales_and_operating_revenue'],6),
      'quick_ratio':round((eb_vals['cash_and_cash_equivalents']+eb_vals['notes_and_accounts_receivable_net'])/eb_vals['total_current_liabilities'],6),
      'debt_to_equity_ratio':round(e_debt/eb_vals['total_equity'],6),
      'debt_to_assets_ratio':round(e_debt/eb_vals['total_assets'],6),
      'debt_to_capital_ratio':round(e_debt/(e_debt+eb_vals['total_equity']),6),
      'debt_to_ebitda_ratio':round(e_debt/e_ebitda,6),
      'interest_coverage_ratio':round(ei_vals['income_before_income_taxes']/abs(ei_vals['interest_expense']),6),
      'total_debt_musd':round(e_debt,6),'ebitda_proxy_musd':round(e_ebitda,6)
     },
     'assumptions':['2024 values selected from 2025/2024/2023 columns in Exxon 10-K statements.']
    },indent = 2),encoding = 'utf-8')

    VW_PDF = PROJECT_ROOT/'Question_1'/'data'/'raw'/'volkswagen_annual_report_2024.pdf'
    vr = PdfReader(str(VW_PDF))
    vi = _normalized_lines(vr.pages[471].extract_text() or '')
    vb1 = _normalized_lines(vr.pages[474].extract_text() or '')
    vb2 = _normalized_lines(vr.pages[475].extract_text() or '')
    vc = _normalized_lines(vr.pages[477].extract_text() or '')
    vi_vals = {
     'sales_revenue':_extract_amount(_find_line(vi,'Sales revenue'),2),
     'cost_of_sales':_extract_amount(_find_line(vi,'Cost of sales'),2),
     'distribution_expenses':_extract_amount(_find_line(vi,'Distribution expenses'),2),
     'administrative_expenses':_extract_amount(_find_line(vi,'Administrative expenses'),2),
     'operating_result':_extract_amount(_find_line(vi,'Operating result'),2),
     'interest_expenses':_extract_amount(_find_line(vi,'Interest expenses'),2),
     'earnings_before_tax':_extract_amount(_find_line(vi,'Earnings before tax'),2),
     'earnings_after_tax':_extract_amount(_find_line(vi,'Earnings after tax'),2),
    }
    vb_vals = {
     'cash_and_cash_equivalents':_extract_amount(_find_line(vb1,'Cash and cash equivalents'),2),
     'trade_receivables':_extract_amount(_find_line(vb1,'Trade receivables'),2),
     'inventories':_extract_amount(_find_line(vb1,'Inventories'),2),
     'total_assets':_extract_amount(_find_line(vb1,'Total assets'),2),
     'financial_liabilities_non_current':_extract_amount(_find_line(vb2,'Financial liabilities  25  137,061  122,323'),2),
     'financial_liabilities_current':_extract_amount(_find_line(vb2,'Financial liabilities  25  117,020  110,476'),2),
     'total_current_liabilities_proxy':_extract_amount(_first_line(vb2,'217,039  206,036'),2),
     'total_equity_proxy':_extract_amount(_first_line(vb2,'196,731  189,186'),2),
    }
    vb_vals['total_liabilities_proxy'] = vb_vals['total_assets']-vb_vals['total_equity_proxy']
    vc_vals = {'depreciation_amortization_proxy':_extract_lookahead(vc,'Depreciation and amortization of, and impairment losses on, intangible assets, property, plant and equipment',2),'depreciation_lease_assets_proxy':_extract_amount(_find_line(vc,'Depreciation of and impairment losses on lease assets'),2)}
    v_debt = vb_vals['financial_liabilities_non_current']+vb_vals['financial_liabilities_current']
    v_ebitda = vi_vals['operating_result']+vc_vals['depreciation_amortization_proxy']+vc_vals['depreciation_lease_assets_proxy']
    VOUT = PROJECT_ROOT/'Question_1'/'output'/'vw_2024_mini_metrics.json'
    VOUT.write_text(json.dumps({
     'company':'Volkswagen AG','statement_year':2024,'units':'EUR millions unless ratio',
     'source_pdf_url':'https://annualreport2024.volkswagen-group.com/_assets/downloads/entire-vw-ar24.pdf',
     'source_pdf_path':str(VW_PDF.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':472,'balance_sheet':475,'cash_flow_statement':478},
     'extracted_values':{'income_statement':vi_vals,'balance_sheet':vb_vals,'cash_flow_statement':vc_vals},
     'metrics':{
       'net_income_musd':round(vi_vals['earnings_after_tax'],6),
       'cost_to_income_ratio':round((vi_vals['cost_of_sales']+vi_vals['distribution_expenses']+vi_vals['administrative_expenses'])/vi_vals['sales_revenue'],6),
       'quick_ratio':round((vb_vals['cash_and_cash_equivalents']+vb_vals['trade_receivables'])/vb_vals['total_current_liabilities_proxy'],6),
       'debt_to_equity_ratio':round(v_debt/vb_vals['total_equity_proxy'],6),
       'debt_to_assets_ratio':round(v_debt/vb_vals['total_assets'],6),
       'debt_to_capital_ratio':round(v_debt/(v_debt+vb_vals['total_equity_proxy']),6),
       'debt_to_ebitda_ratio':round(v_debt/v_ebitda,6),
       'interest_coverage_ratio':round(vi_vals['earnings_before_tax']/abs(vi_vals['interest_expenses']),6),
       'total_debt_musd':round(v_debt,6),'ebitda_proxy_musd':round(v_ebitda,6)
     },
     'assumptions':['VW current liabilities/equity use unlabeled subtotal lines in balance section.']
    },indent = 2),encoding = 'utf-8')

    print('Saved:',EOUT); print('Saved:',VOUT)

    # Microsoft + Alphabet via SEC 10-K inline XBRL (FY2024)
    MSFT_URL = 'https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm'
    GOOG_URL = 'https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm'
    MSFT_PATH = PROJECT_ROOT/'Question_1'/'data'/'raw'/'microsoft_2024_10k.html'
    GOOG_PATH = PROJECT_ROOT/'Question_1'/'data'/'raw'/'alphabet_2024_10k.html'
    _ensure_file(MSFT_URL, MSFT_PATH, headers = SEC_HEADERS, verify = True)
    _ensure_file(GOOG_URL, GOOG_PATH, headers = SEC_HEADERS, verify = True)

    mf = _sec_facts('0000789019')
    gf = _sec_facts('0001652044')

    m = {
     'revenue':_pick_sec(mf,'RevenueFromContractWithCustomerExcludingAssessedTax',2024,'2024-06-30')/1e6,
     'cost':_pick_sec(mf,'CostOfGoodsAndServicesSold',2024,'2024-06-30')/1e6,
     'rnd':_pick_sec(mf,'ResearchAndDevelopmentExpense',2024,'2024-06-30')/1e6,
     'sm':_pick_sec(mf,'SellingAndMarketingExpense',2024,'2024-06-30')/1e6,
     'ga':_pick_sec(mf,'GeneralAndAdministrativeExpense',2024,'2024-06-30')/1e6,
     'oper':_pick_sec(mf,'OperatingIncomeLoss',2024,'2024-06-30')/1e6,
     'net':_pick_sec(mf,'NetIncomeLoss',2024,'2024-06-30')/1e6,
     'interest':_pick_sec(mf,'InterestExpense',2024,'2024-06-30')/1e6,
     'dep':_pick_sec(mf,'Depreciation',2024,'2024-06-30')/1e6,
     'assets':_pick_sec(mf,'Assets',2024,'2024-06-30')/1e6,
     'liab':_pick_sec(mf,'Liabilities',2024,'2024-06-30')/1e6,
     'equity':_pick_sec(mf,'StockholdersEquity',2024,'2024-06-30')/1e6,
     'curr_liab':_pick_sec(mf,'LiabilitiesCurrent',2024,'2024-06-30')/1e6,
     'cash':_pick_sec(mf,'CashAndCashEquivalentsAtCarryingValue',2024,'2024-06-30')/1e6,
     'short_inv':_pick_sec(mf,'ShortTermInvestments',2024,'2024-06-30')/1e6,
     'ar':_pick_sec(mf,'AccountsReceivableNetCurrent',2024,'2024-06-30')/1e6,
     'ltd_curr':_pick_sec(mf,'LongTermDebtCurrent',2024,'2024-06-30')/1e6,
     'ltd_noncurr':_pick_sec(mf,'LongTermDebtNoncurrent',2024,'2024-06-30')/1e6,
     'cp':_pick_sec(mf,'CommercialPaper',2024,'2024-06-30')/1e6,
    }
    m_debt = m['ltd_curr']+m['ltd_noncurr']+m['cp']; m_ebitda = m['oper']+m['dep']
    MOUT = PROJECT_ROOT/'Question_1'/'output'/'microsoft_2024_mini_metrics.json'
    MOUT.write_text(json.dumps({
     'company':'Microsoft Corporation','statement_year':2024,'units':'USD millions unless ratio',
     'source_pdf_url':MSFT_URL,'source_pdf_path':str(MSFT_PATH.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':1,'balance_sheet':1,'cash_flow_statement':1},
     'extracted_values':{'income_statement':m,'balance_sheet':m,'cash_flow_statement':{'depreciation_proxy':m['dep']}},
     'metrics':{
       'net_income_musd':round(m['net'],6),
       'cost_to_income_ratio':round((m['cost']+m['rnd']+m['sm']+m['ga'])/m['revenue'],6),
       'quick_ratio':round((m['cash']+m['short_inv']+m['ar'])/m['curr_liab'],6),
       'debt_to_equity_ratio':round(m_debt/m['equity'],6),
       'debt_to_assets_ratio':round(m_debt/m['assets'],6),
       'debt_to_capital_ratio':round(m_debt/(m_debt+m['equity']),6),
       'debt_to_ebitda_ratio':round(m_debt/m_ebitda,6),
       'interest_coverage_ratio':round(m['oper']/m['interest'],6),
       'total_debt_musd':round(m_debt,6),'ebitda_proxy_musd':round(m_ebitda,6)
     },
     'assumptions':['SEC XBRL tag-based extraction; page placeholders set to 1 for non-PDF source.']
    },indent = 2),encoding = 'utf-8')

    g = {
     'revenue':_pick_sec(gf,'RevenueFromContractWithCustomerExcludingAssessedTax',2024,'2024-12-31')/1e6,
     'cost':_pick_sec(gf,'CostOfRevenue',2024,'2024-12-31')/1e6,
     'rnd':_pick_sec(gf,'ResearchAndDevelopmentExpense',2024,'2024-12-31')/1e6,
     'sm':_pick_sec(gf,'SellingAndMarketingExpense',2024,'2024-12-31')/1e6,
     'ga':_pick_sec(gf,'GeneralAndAdministrativeExpense',2024,'2024-12-31')/1e6,
     'oper':_pick_sec(gf,'OperatingIncomeLoss',2024,'2024-12-31')/1e6,
     'net':_pick_sec(gf,'NetIncomeLoss',2024,'2024-12-31')/1e6,
     'interest':_pick_sec(gf,'InterestExpenseNonoperating',2024,'2024-12-31')/1e6,
     'dep':_pick_sec(gf,'Depreciation',2024,'2024-12-31')/1e6,
     'assets':_pick_sec(gf,'Assets',2024,'2024-12-31')/1e6,
     'liab':_pick_sec(gf,'Liabilities',2024,'2024-12-31')/1e6,
     'equity':_pick_sec(gf,'StockholdersEquity',2024,'2024-12-31')/1e6,
     'curr_liab':_pick_sec(gf,'LiabilitiesCurrent',2024,'2024-12-31')/1e6,
     'cash':_pick_sec(gf,'CashAndCashEquivalentsAtCarryingValue',2024,'2024-12-31')/1e6,
     'marketable':_pick_sec(gf,'MarketableSecuritiesCurrent',2024,'2024-12-31')/1e6,
     'ar':_pick_sec(gf,'AccountsReceivableNetCurrent',2024,'2024-12-31')/1e6,
     'ltd_curr':_pick_sec(gf,'LongTermDebtCurrent',2024,'2024-12-31')/1e6,
     'ltd_noncurr':_pick_sec(gf,'LongTermDebtNoncurrent',2024,'2024-12-31')/1e6,
     'cp':_pick_sec(gf,'CommercialPaper',2024,'2024-12-31')/1e6,
    }
    g_debt = g['ltd_curr']+g['ltd_noncurr']+g['cp']; g_ebitda = g['oper']+g['dep']
    GOUT = PROJECT_ROOT/'Question_1'/'output'/'alphabet_2024_mini_metrics.json'
    GOUT.write_text(json.dumps({
     'company':'Alphabet Inc.','statement_year':2024,'units':'USD millions unless ratio',
     'source_pdf_url':GOOG_URL,'source_pdf_path':str(GOOG_PATH.as_posix()),
     'statement_pages_pdf_1_based':{'income_statement':1,'balance_sheet':1,'cash_flow_statement':1},
     'extracted_values':{'income_statement':g,'balance_sheet':g,'cash_flow_statement':{'depreciation_proxy':g['dep']}},
     'metrics':{
       'net_income_musd':round(g['net'],6),
       'cost_to_income_ratio':round((g['cost']+g['rnd']+g['sm']+g['ga'])/g['revenue'],6),
       'quick_ratio':round((g['cash']+g['marketable']+g['ar'])/g['curr_liab'],6),
       'debt_to_equity_ratio':round(g_debt/g['equity'],6),
       'debt_to_assets_ratio':round(g_debt/g['assets'],6),
       'debt_to_capital_ratio':round(g_debt/(g_debt+g['equity']),6),
       'debt_to_ebitda_ratio':round(g_debt/g_ebitda,6),
       'interest_coverage_ratio':round(g['oper']/g['interest'],6),
       'total_debt_musd':round(g_debt,6),'ebitda_proxy_musd':round(g_ebitda,6)
     },
     'assumptions':['SEC XBRL tag-based extraction; page placeholders set to 1 for non-PDF source.']
    },indent = 2),encoding = 'utf-8')

    print('Saved:',MOUT); print('Saved:',GOUT)

    return {

        'tencent_2024': json.loads(TOUT.read_text(encoding = 'utf-8')),
        'alibaba_2024': json.loads(AOUT.read_text(encoding = 'utf-8')),
        'exxon_2024': json.loads(EOUT.read_text(encoding = 'utf-8')),
        'vw_2024': json.loads(VOUT.read_text(encoding = 'utf-8')),
        'microsoft_2024': json.loads(MOUT.read_text(encoding = 'utf-8')),
        'alphabet_2024': json.loads(GOUT.read_text(encoding = 'utf-8')),
    }


def refresh_manifest_and_cross(question_root: Path | None = None) -> dict[str, Any]:

    QUESTION_ROOT = resolve_question_root(question_root)
    PROJECT_ROOT = QUESTION_ROOT.parent
    MANIFEST_PATH = PROJECT_ROOT / 'Question_1' / 'config' / 'annual_report_manifest_2026q1.json'
    CROSS_OUTPUT_PATH = PROJECT_ROOT / 'Question_1' / 'output' / 'mini_cross_company_metrics.json'
    manifest_live = json.loads(MANIFEST_PATH.read_text(encoding = 'utf-8'))
    patch = {
     'exxon_2024':{'pdf_url':'https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/0000034088-26-000045/0000034088-26-000045.pdf','pdf_path':'Question_1/data/raw/exxon_2024_10k.pdf'},
     'microsoft_2024':{'pdf_url':'https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm','pdf_path':'Question_1/data/raw/microsoft_2024_10k.html'},
     'alphabet_2024':{'pdf_url':'https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm','pdf_path':'Question_1/data/raw/alphabet_2024_10k.html'},
    }
    for item in manifest_live['companies']:

        if item['id'] in patch:
            item.update(patch[item['id']])
        pdf_path = PROJECT_ROOT/item['pdf_path']; out_path = PROJECT_ROOT/item['output_path']

        if pdf_path.exists() and out_path.exists():
            item['status'] = 'complete'
    manifest_live['manifest_version'] = '2026q1-v2'
    MANIFEST_PATH.write_text(json.dumps(manifest_live,indent = 2),encoding = 'utf-8')

    cross = {}
    for item in manifest_live['companies']:

        if not item.get('required_now',False):

            continue

        out_path = PROJECT_ROOT/item['output_path']

        if out_path.exists():
            cross[item['id']] = json.loads(out_path.read_text(encoding = 'utf-8'))
    CROSS_OUTPUT_PATH.write_text(json.dumps(cross,indent = 2),encoding = 'utf-8')

    rows = []
    for cid,payload in cross.items():
        m = payload.get('metrics',{})
        rows.append({'id':cid,'company':payload.get('company'),'net_income':m.get('net_income_musd'),'cost_to_income':m.get('cost_to_income_ratio'),'quick_ratio':m.get('quick_ratio'),'debt_to_equity':m.get('debt_to_equity_ratio'),'debt_to_assets':m.get('debt_to_assets_ratio'),'debt_to_capital':m.get('debt_to_capital_ratio'),'debt_to_ebitda':m.get('debt_to_ebitda_ratio'),'interest_coverage':m.get('interest_coverage_ratio')})
    summary = pd.DataFrame(rows).sort_values('id').reset_index(drop = True)
    print('Saved manifest:',MANIFEST_PATH)
    print('Saved cross:',CROSS_OUTPUT_PATH)
    summary

    return {

        'manifest_live': manifest_live,
        'cross_payload': cross,
        'summary': summary,
        'manifest_path': MANIFEST_PATH,
        'cross_output_path': CROSS_OUTPUT_PATH,
    }


__all__ = [
    "build_cross_summary",
    "extract_full_required_coverage",
    "extract_gm_pilot",
    "extract_jpm_pilot",
    "extract_lvmh_extension",
    "refresh_manifest_and_cross",
    "refresh_manifest_progress",
    "resolve_question_root",
    "validate_manifest_outputs",
]
