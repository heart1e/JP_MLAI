"""Bonus 1 rating and shenanigans helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from AutoPDF import extract_configured_pdf_case


COMPARATORS = {
    '<': lambda value, threshold: value < threshold,
    '<=': lambda value, threshold: value <= threshold,
    '>': lambda value, threshold: value > threshold,
    '>=': lambda value, threshold: value >= threshold,
}


@dataclass(frozen = True)
class Bonus1RatingConfig:
    """Typed bundle for the Bonus 1 rating rules."""

    config_path: Path
    non_bank_rules: dict[str, dict[str, Any]]
    bank_rules: dict[str, dict[str, Any]]
    distress_rules: dict[str, list[dict[str, Any]]]
    rating_bands: dict[str, Any]
    methodology: dict[str, Any]


@dataclass(frozen = True)
class Bonus1CaseRegistry:
    """Typed bundle for the distressed PDF case registry."""

    config_path: Path
    config: dict[str, Any]
    cases: list[dict[str, Any]]


@dataclass(frozen = True)
class Bonus1ShenanigansConfig:
    """Typed bundle for the quantitative shenanigans rules."""

    config_path: Path
    methodology: dict[str, Any]
    rules: list[dict[str, Any]]
    warning_bands: dict[str, Any]


def resolve_question_root(question_root: Path | None = None) -> Path:
    """Resolve the Question_1 workspace root."""

    if question_root is not None:
        return question_root.resolve()

    return Path(__file__).resolve().parents[1]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding = 'utf-8'))

    if not isinstance(payload, dict):
        raise TypeError(f'Expected a JSON object in {path.name}.')

    return payload


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)

    if not isinstance(value, dict):
        raise TypeError(f"Expected '{key}' to be a JSON object.")

    return value


def _require_case_list(mapping: dict[str, Any], key: str = 'cases') -> list[dict[str, Any]]:
    value = mapping.get(key)

    if not isinstance(value, list):
        raise TypeError(f"Expected '{key}' to be a JSON array.")

    cases: list[dict[str, Any]] = []

    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"Expected every entry in '{key}' to be a JSON object.")
        cases.append(item)

    return cases


def load_bonus1_rating_config(
    question_root: Path | None = None,
    *,
    config_filename: str = 'bonus1_rating_config.json',
) -> Bonus1RatingConfig:
    """Load the config-driven Bonus 1 rating rules."""

    question_root = resolve_question_root(question_root)
    config_path = question_root / 'config' / config_filename
    payload = _load_json_mapping(config_path)
    metric_rules = _require_mapping(payload, 'metric_rules')
    non_bank_rules = _require_mapping(metric_rules, 'non_bank')
    bank_rules = _require_mapping(metric_rules, 'bank')
    distress_rules = _require_mapping(payload, 'distress_rules')
    rating_bands = _require_mapping(payload, 'rating_bands')
    methodology = dict(_require_mapping(payload, 'methodology'))
    methodology['non_bank_metrics'] = list(non_bank_rules)
    methodology['bank_metrics'] = list(bank_rules)
    methodology['config_path'] = str(config_path.as_posix())
    methodology['guardrails'] = [
        'Ratios that require positive equity or positive capital are suppressed when the denominator is non-positive.',
        'Debt-to-EBITDA is ignored when the extracted ratio is non-positive or missing.',
    ]

    return Bonus1RatingConfig(
        config_path = config_path,
        non_bank_rules = non_bank_rules,
        bank_rules = bank_rules,
        distress_rules = distress_rules,
        rating_bands = rating_bands,
        methodology = methodology,
    )


def load_bonus1_case_registry(
    question_root: Path | None = None,
    *,
    registry_filename: str = 'bonus1_case_sources.json',
) -> Bonus1CaseRegistry:
    """Load the generic distressed-report case registry."""

    question_root = resolve_question_root(question_root)
    config_path = question_root / 'config' / registry_filename
    payload = _load_json_mapping(config_path)
    cases = _require_case_list(payload)

    return Bonus1CaseRegistry(
        config_path = config_path,
        config = payload,
        cases = cases,
    )


def find_bonus1_case(case_registry: Bonus1CaseRegistry, case_id: str) -> dict[str, Any]:
    """Return one configured Bonus 1 case by id."""

    case_cfg = next((case for case in case_registry.cases if case.get('id') == case_id), None)

    if case_cfg is None:
        raise KeyError(f'Case id {case_id} is missing from {case_registry.config_path.name}.')

    return case_cfg


def load_bonus1_payloads(
    question_root: Path | None = None,
    *,
    manifest_filename: str = 'annual_report_sources.json',
) -> dict[str, dict[str, Any]]:
    """Load all in-scope annual-report payloads for Bonus 1 scoring."""

    question_root = resolve_question_root(question_root)
    manifest_path = question_root / 'config' / manifest_filename
    manifest_cfg = _load_json_mapping(manifest_path)
    payloads: dict[str, dict[str, Any]] = {}

    for item in manifest_cfg.get('reports', []):
        if not isinstance(item, dict):
            continue

        if not bool(item.get('required_now', False)):
            continue

        rel_output = str(item.get('output_path', '')).replace('Question_1/', '')

        if not rel_output:
            continue

        output_path = question_root / rel_output

        if output_path.exists():
            payloads[str(item['id'])] = _load_json_mapping(output_path)

    return payloads


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

    except TypeError:
        pass

    return float(value)


def _coalesce_numeric(*values: Any) -> float | None:
    for value in values:
        cleaned = _clean_number(value)

        if cleaned is not None:
            return cleaned

    return None


def _sum_available(*values: Any) -> float | None:
    cleaned_values = [cleaned for cleaned in (_clean_number(v) for v in values) if cleaned is not None]

    return float(sum(cleaned_values)) if cleaned_values else None


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    numerator = _clean_number(numerator)
    denominator = _clean_number(denominator)

    if numerator is None or denominator is None or denominator == 0:
        return None

    return float(numerator / denominator)


def _extract_bonus1_features(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get('metrics', {})
    canonical_values = payload.get('canonical_values', {})

    if not isinstance(canonical_values, dict) or not canonical_values:
        raise KeyError(
            f"Missing canonical_values for {payload.get('company')}. Rebuild Part 2 outputs with AutoPDF.py."
        )

    total_debt = _coalesce_numeric(canonical_values.get('total_debt'), metrics.get('total_debt_musd'))
    liquid_assets = _clean_number(canonical_values.get('liquid_assets'))
    current_assets_proxy = _clean_number(canonical_values.get('current_assets_proxy'))
    cash_like_assets = _clean_number(canonical_values.get('cash_like_assets'))
    assets = _clean_number(canonical_values.get('assets'))
    liabilities = _clean_number(canonical_values.get('liabilities'))
    equity = _clean_number(canonical_values.get('equity'))
    current_liabilities = _clean_number(canonical_values.get('current_liabilities'))
    revenue = _clean_number(canonical_values.get('revenue'))
    operating_income = _clean_number(canonical_values.get('operating_income'))
    net_income = _coalesce_numeric(canonical_values.get('net_income'), metrics.get('net_income_musd'))
    positive_equity = equity if equity is not None and equity > 0 else None
    positive_capital_base = None

    if total_debt is not None and equity is not None and equity > 0 and (total_debt + equity) > 0:
        positive_capital_base = total_debt + equity

    raw_debt_to_assets = _clean_number(metrics.get('debt_to_assets_ratio'))
    raw_debt_to_capital = _clean_number(metrics.get('debt_to_capital_ratio'))
    raw_debt_to_equity = _clean_number(metrics.get('debt_to_equity_ratio'))
    raw_debt_to_ebitda = _clean_number(metrics.get('debt_to_ebitda_ratio'))

    return {
        'is_bank': bool(canonical_values.get('is_bank_like')),
        'equity_proxy': equity,
        'net_income_musd': net_income,
        'current_ratio': _safe_div(current_assets_proxy, current_liabilities),
        'quick_ratio': _coalesce_numeric(metrics.get('quick_ratio'), _safe_div(liquid_assets, current_liabilities)),
        'cash_ratio': _safe_div(cash_like_assets, current_liabilities),
        'debt_to_assets_ratio': _coalesce_numeric(raw_debt_to_assets, _safe_div(total_debt, assets)),
        'debt_to_capital_ratio': _coalesce_numeric(
            raw_debt_to_capital if raw_debt_to_capital is not None and raw_debt_to_capital > 0 else None,
            _safe_div(total_debt, positive_capital_base),
        ),
        'debt_to_equity_ratio': _coalesce_numeric(
            raw_debt_to_equity if raw_debt_to_equity is not None and raw_debt_to_equity > 0 else None,
            _safe_div(total_debt, positive_equity),
        ),
        'debt_to_ebitda_ratio': raw_debt_to_ebitda if raw_debt_to_ebitda is not None and raw_debt_to_ebitda > 0 else None,
        'interest_coverage_ratio': _clean_number(metrics.get('interest_coverage_ratio')),
        'cost_to_income_ratio': _clean_number(metrics.get('cost_to_income_ratio')),
        'equity_to_assets_ratio': _safe_div(equity, assets),
        'liabilities_to_assets_ratio': _safe_div(liabilities, assets),
        'roa_ratio': _safe_div(net_income, assets),
        'net_margin_ratio': _safe_div(net_income, revenue),
        'operating_margin_ratio': _safe_div(operating_income, revenue),
        'cash_to_debt_ratio': _safe_div(cash_like_assets, total_debt),
        'liquid_assets_to_assets_ratio': _safe_div(liquid_assets, assets),
        'total_debt_musd': total_debt,
        'liquid_assets_musd': liquid_assets,
        'current_assets_proxy_musd': current_assets_proxy,
    }


def _score_bonus1_metric(value: Any, rule: dict[str, Any]) -> int | None:
    value = _clean_number(value)

    if value is None:
        return None

    good = rule['good']
    okay = rule['okay']
    weak = rule['weak']
    higher_is_better = rule['higher_is_better']

    if higher_is_better:
        if value >= good:
            return 2
        if value >= okay:
            return 1
        if value >= weak:
            return 0
        return -2

    if value <= good:
        return 2
    if value <= okay:
        return 1
    if value <= weak:
        return 0

    return -2


def _score_metric_bundle(features: dict[str, Any], rule_map: dict[str, dict[str, Any]]) -> dict[str, int]:
    component_scores: dict[str, int] = {}

    for metric_name, rule in rule_map.items():
        score = _score_bonus1_metric(features.get(metric_name), rule)

        if score is not None:
            component_scores[f'{metric_name}_score'] = score

    return component_scores


def _evaluate_quant_rules(features: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    return [
        rule['flag']
        for rule in rules
        if _clean_number(features.get(rule['metric'])) is not None
        and COMPARATORS[rule['operator']](_clean_number(features.get(rule['metric'])), rule['threshold'])
    ]


def _assign_bonus1_rating(
    normalized_score: float | None,
    distress_count: int,
    rating_bands: dict[str, Any],
) -> str:
    if distress_count >= int(rating_bands['d_min_distress_flags']):
        return 'D'

    if distress_count >= int(rating_bands['c_min_distress_flags']):
        return 'C'

    if normalized_score is None:
        return 'C'

    if normalized_score >= float(rating_bands['a_min_normalized_score']):
        return 'A'

    if normalized_score >= float(rating_bands['b_min_normalized_score']):
        return 'B'

    if normalized_score >= float(rating_bands['c_min_normalized_score']):
        return 'C'

    return 'D'


def score_bonus1_payload(payload: dict[str, Any], rating_config: Bonus1RatingConfig) -> dict[str, Any]:
    """Score one annual-report payload into the internal A/B/C/D baseline."""

    features = _extract_bonus1_features(payload)
    rule_map = rating_config.bank_rules if features['is_bank'] else rating_config.non_bank_rules
    component_scores = _score_metric_bundle(features, rule_map)
    distress_rules = rating_config.distress_rules['common'] + rating_config.distress_rules['bank' if features['is_bank'] else 'non_bank']
    distress_flags = _evaluate_quant_rules(features, distress_rules)
    total_score = int(sum(component_scores.values())) if component_scores else 0
    scored_metric_count = len(component_scores)
    normalized_score = float(total_score / (2 * scored_metric_count)) if scored_metric_count else None
    rating = _assign_bonus1_rating(normalized_score, len(distress_flags), rating_config.rating_bands)

    rationale_map = {
        'A': 'Strong annual-report profile across liquidity, leverage, and profitability ratios.',
        'B': 'Acceptable quantitative profile, but not consistently strong across the full ratio set.',
        'C': 'Weak quantitative profile with visible stress in leverage, liquidity, or profitability.',
        'D': 'Distressed quantitative profile with multiple hard warning signals.',
    }

    return {
        'company': payload.get('company'),
        'statement_year': payload.get('statement_year'),
        'units': payload.get('units'),
        'rating': rating,
        'total_score': total_score,
        'normalized_score': normalized_score,
        'scored_metric_count': scored_metric_count,
        'distress_flag_count': len(distress_flags),
        'distress_flags': distress_flags,
        'metrics_used': features,
        'component_scores': component_scores,
        'rationale': rationale_map[rating],
    }


def build_bonus1_rating_artifact(
    question_root: Path | None = None,
    *,
    config_filename: str = 'bonus1_rating_config.json',
    output_filename: str = 'bonus1_internal_rating_baseline.json',
) -> dict[str, Any]:
    """Build and persist the cross-company Bonus 1 rating baseline."""

    question_root = resolve_question_root(question_root)
    rating_config = load_bonus1_rating_config(question_root, config_filename = config_filename)
    payloads = load_bonus1_payloads(question_root)
    rows = []

    for company_id, payload in payloads.items():
        row = {'id': company_id}
        row.update(score_bonus1_payload(payload, rating_config))
        rows.append(row)

    rating_df = pd.DataFrame(rows).sort_values(['rating', 'normalized_score', 'id'], ascending = [True, False, True]).reset_index(drop = True)
    artifact = {
        'methodology': rating_config.methodology,
        'ratings': json.loads(rating_df.to_json(orient = 'records')),
    }
    output_path = question_root / 'output' / output_filename
    output_path.write_text(json.dumps(artifact, indent = 2), encoding = 'utf-8')

    return artifact


def build_bonus1_rating_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert the persisted rating artifact into the notebook display frame."""

    return pd.DataFrame(artifact['ratings']).sort_values(
        ['rating', 'normalized_score', 'id'],
        ascending = [True, False, True],
    ).reset_index(drop = True)


def load_bonus1_shenanigans_config(
    question_root: Path | None = None,
    *,
    config_filename: str = 'bonus1_shenanigans_config.json',
) -> Bonus1ShenanigansConfig:
    """Load the quantitative shenanigans rule set."""

    question_root = resolve_question_root(question_root)
    config_path = question_root / 'config' / config_filename
    payload = _load_json_mapping(config_path)
    rules_raw = payload.get('rules')

    if not isinstance(rules_raw, list):
        raise TypeError("Expected 'rules' to be a JSON array.")

    rules: list[dict[str, Any]] = []

    for rule in rules_raw:
        if not isinstance(rule, dict):
            raise TypeError("Expected every shenanigans rule to be a JSON object.")
        rules.append(rule)

    warning_bands = _require_mapping(payload, 'warning_bands')

    return Bonus1ShenanigansConfig(
        config_path = config_path,
        methodology = dict(payload.get('methodology', {})),
        rules = rules,
        warning_bands = warning_bands,
    )


def _extract_shenanigans_features(payload: dict[str, Any]) -> dict[str, Any]:
    canonical_values = payload.get('canonical_values', {})
    metrics = payload.get('metrics', {})
    total_debt = _coalesce_numeric(canonical_values.get('total_debt'), metrics.get('total_debt_musd'))
    revenue = _clean_number(canonical_values.get('revenue'))
    restricted_cash = _clean_number(canonical_values.get('restricted_cash'))
    unrestricted_cash = _clean_number(canonical_values.get('cash'))
    cash_total = _sum_available(restricted_cash, unrestricted_cash)
    receivables = _coalesce_numeric(canonical_values.get('receivables'), canonical_values.get('trade_receivables'))
    other_receivables = _clean_number(canonical_values.get('other_receivables'))
    current_borrowings = _coalesce_numeric(canonical_values.get('current_borrowings'), canonical_values.get('debt_current'))
    operating_cashflow = _clean_number(canonical_values.get('operating_cashflow'))
    write_down_properties = _clean_number(canonical_values.get('write_down_properties'))
    impairment_financial_assets = _clean_number(canonical_values.get('impairment_financial_assets'))
    trade_payables = _clean_number(canonical_values.get('trade_payables'))

    return {
        'equity_proxy': _clean_number(canonical_values.get('equity')),
        'current_ratio': _safe_div(canonical_values.get('current_assets_proxy'), canonical_values.get('current_liabilities')),
        'cash_to_debt_ratio': _safe_div(canonical_values.get('cash_like_assets'), total_debt),
        'interest_coverage_ratio': _clean_number(metrics.get('interest_coverage_ratio')),
        'net_cash_used_operating_ratio': _safe_div(operating_cashflow, revenue),
        'restricted_cash_share': _safe_div(restricted_cash, cash_total),
        'receivables_to_revenue_ratio': _safe_div(receivables, revenue),
        'other_receivables_to_assets_ratio': _safe_div(other_receivables, canonical_values.get('assets')),
        'write_down_to_revenue_ratio': _safe_div(abs(write_down_properties) if write_down_properties is not None else None, revenue),
        'impairment_to_revenue_ratio': _safe_div(abs(impairment_financial_assets) if impairment_financial_assets is not None else None, revenue),
        'current_borrowings_to_total_debt_ratio': _safe_div(current_borrowings, total_debt),
        'trade_payables_to_revenue_ratio': _safe_div(trade_payables, revenue),
    }


def _assign_warning_band(total_points: int, warning_bands: dict[str, Any]) -> str:
    if total_points <= int(warning_bands['low_max_points']):
        return 'low'

    if total_points <= int(warning_bands['moderate_max_points']):
        return 'moderate'

    if total_points <= int(warning_bands['high_max_points']):
        return 'high'

    return 'severe'


def evaluate_shenanigans_payload(
    payload: dict[str, Any],
    shenanigans_config: Bonus1ShenanigansConfig,
) -> dict[str, Any]:
    """Evaluate one annual-report payload for quantitative warning signals."""

    features = _extract_shenanigans_features(payload)
    triggered_rules = []
    total_points = 0

    for rule in shenanigans_config.rules:
        value = _clean_number(features.get(rule['metric']))

        if value is None:
            continue

        if COMPARATORS[rule['operator']](value, rule['threshold']):
            triggered_rules.append(
                {
                    'id': rule['id'],
                    'metric': rule['metric'],
                    'value': value,
                    'threshold': rule['threshold'],
                    'points': int(rule['points']),
                    'severity': rule['severity'],
                    'description': rule['description'],
                }
            )
            total_points += int(rule['points'])

    warning_band = _assign_warning_band(total_points, shenanigans_config.warning_bands)

    return {
        'company': payload.get('company'),
        'statement_year': payload.get('statement_year'),
        'warning_band': warning_band,
        'total_points': total_points,
        'trigger_count': len(triggered_rules),
        'triggered_rules': triggered_rules,
        'derived_features': features,
    }


def build_bonus1_shenanigans_artifact(
    question_root: Path | None = None,
    *,
    case_id: str = 'evergrande_2022',
    registry_filename: str = 'bonus1_case_sources.json',
    config_filename: str = 'bonus1_shenanigans_config.json',
    output_filename: str = 'bonus1_shenanigans_scan.json',
) -> dict[str, Any]:
    """Build and persist the cross-company shenanigans scan."""

    question_root = resolve_question_root(question_root)
    shenanigans_config = load_bonus1_shenanigans_config(question_root, config_filename = config_filename)
    payloads = dict(load_bonus1_payloads(question_root))
    payloads[case_id] = extract_configured_pdf_case(case_id, question_root)
    scans = []

    for company_id, payload in payloads.items():
        scan = evaluate_shenanigans_payload(payload, shenanigans_config)
        scans.append({'id': company_id, **scan})

    artifact = {
        'methodology': shenanigans_config.methodology,
        'config_path': str(shenanigans_config.config_path.as_posix()),
        'scans': scans,
    }
    output_path = question_root / 'output' / output_filename
    output_path.write_text(json.dumps(artifact, indent = 2), encoding = 'utf-8')

    return artifact


def build_bonus1_shenanigans_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert the persisted shenanigans scan into the notebook display frame."""

    frame = pd.DataFrame(artifact['scans']).copy()
    frame['trigger_ids'] = frame['triggered_rules'].apply(lambda rules: ', '.join(rule['id'] for rule in rules))

    return frame[
        ['id', 'company', 'warning_band', 'total_points', 'trigger_count', 'trigger_ids']
    ].sort_values(['total_points', 'trigger_count', 'id'], ascending = [False, False, True]).reset_index(drop = True)


def build_bonus1_case_artifact(
    case_id: str,
    question_root: Path | None = None,
    *,
    registry_filename: str = 'bonus1_case_sources.json',
    rating_config_filename: str = 'bonus1_rating_config.json',
    output_filename: str | None = None,
) -> dict[str, Any]:
    """Build and persist one configured distressed case rating artifact."""

    question_root = resolve_question_root(question_root)
    case_registry = load_bonus1_case_registry(question_root, registry_filename = registry_filename)
    case_cfg = find_bonus1_case(case_registry, case_id)
    payload = extract_configured_pdf_case(case_id, question_root)
    rating_config = load_bonus1_rating_config(question_root, config_filename = rating_config_filename)
    rating_result = score_bonus1_payload(payload, rating_config)
    source_output_path = question_root / 'output' / f'{case_id}_mini_metrics.json'
    artifact = {
        'case_config_path': str(case_registry.config_path.as_posix()),
        'case_source': case_cfg,
        'rating_result': rating_result,
        'source_output_path': str(source_output_path.as_posix()),
    }
    output_name = output_filename or f'bonus1_{case_id}_rating_case.json'
    output_path = question_root / 'output' / output_name
    output_path.write_text(json.dumps(artifact, indent = 2), encoding = 'utf-8')

    return artifact


def build_bonus1_case_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert a persisted distressed case artifact into the notebook display frame."""

    case_cfg = artifact['case_source']
    rating_result = artifact['rating_result']

    return pd.DataFrame(
        [
            {
                'id': case_cfg['id'],
                'company': rating_result['company'],
                'rating': rating_result['rating'],
                'normalized_score': rating_result['normalized_score'],
                'distress_flag_count': rating_result['distress_flag_count'],
                'distress_flags': ', '.join(rating_result['distress_flags']),
                'total_debt_musd': rating_result['metrics_used']['total_debt_musd'],
                'equity_proxy_musd': rating_result['metrics_used']['equity_proxy'],
                'current_ratio': rating_result['metrics_used']['current_ratio'],
                'cash_to_debt_ratio': rating_result['metrics_used']['cash_to_debt_ratio'],
                'interest_coverage_ratio': rating_result['metrics_used']['interest_coverage_ratio'],
                'rationale': rating_result['rationale'],
            }
        ]
    )


def build_bonus1_bankruptcy_oos_artifact(
    question_root: Path | None = None,
    *,
    registry_filename: str = 'bonus1_case_sources.json',
    rating_config_filename: str = 'bonus1_rating_config.json',
    shenanigans_config_filename: str = 'bonus1_shenanigans_config.json',
    output_filename: str = 'bonus1_bankruptcy_oos_summary.json',
) -> dict[str, Any]:
    """Build and persist the bankruptcy out-of-sample cohort summary."""

    question_root = resolve_question_root(question_root)
    case_registry = load_bonus1_case_registry(question_root, registry_filename = registry_filename)
    rating_config = load_bonus1_rating_config(question_root, config_filename = rating_config_filename)
    shenanigans_config = load_bonus1_shenanigans_config(question_root, config_filename = shenanigans_config_filename)
    case_cfgs = [case for case in case_registry.cases if 'bankruptcy_oos' in case.get('cohorts', [])]
    records = []

    for case_cfg in case_cfgs:
        payload = extract_configured_pdf_case(case_cfg['id'], question_root)
        rating_result = score_bonus1_payload(payload, rating_config)
        shenanigans_result = evaluate_shenanigans_payload(payload, shenanigans_config)
        records.append(
            {
                'id': case_cfg['id'],
                'company_name': case_cfg['company_name'],
                'statement_year': case_cfg['statement_year'],
                'distress_event': case_cfg.get('distress_event'),
                'distress_event_year': case_cfg.get('distress_event_year'),
                'source_output_path': case_cfg['output_path'],
                'rating_result': rating_result,
                'shenanigans_result': shenanigans_result,
            }
        )

    artifact = {
        'cohort_name': 'bankruptcy_oos',
        'case_config_path': str(case_registry.config_path.as_posix()),
        'case_ids': [case['id'] for case in case_cfgs],
        'records': records,
    }
    output_path = question_root / 'output' / output_filename
    output_path.write_text(json.dumps(artifact, indent = 2), encoding = 'utf-8')

    return artifact


def build_bonus1_bankruptcy_oos_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert the bankruptcy cohort artifact into the notebook display frame."""

    rows = []

    for record in artifact['records']:
        rating_result = record['rating_result']
        shenanigans_result = record['shenanigans_result']
        rows.append(
            {
                'id': record['id'],
                'company': rating_result['company'],
                'statement_year': rating_result['statement_year'],
                'distress_event_year': record.get('distress_event_year'),
                'distress_event': record.get('distress_event'),
                'rating': rating_result['rating'],
                'normalized_score': rating_result['normalized_score'],
                'distress_flag_count': rating_result['distress_flag_count'],
                'warning_band': shenanigans_result['warning_band'],
                'total_points': shenanigans_result['total_points'],
                'trigger_count': shenanigans_result['trigger_count'],
                'key_triggers': ', '.join(rule['id'] for rule in shenanigans_result['triggered_rules'][:6]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ['total_points', 'trigger_count', 'normalized_score', 'id'],
        ascending = [False, False, True, True],
    ).reset_index(drop = True)


__all__ = [
    'Bonus1CaseRegistry',
    'Bonus1RatingConfig',
    'Bonus1ShenanigansConfig',
    'build_bonus1_bankruptcy_oos_artifact',
    'build_bonus1_bankruptcy_oos_frame',
    'build_bonus1_case_artifact',
    'build_bonus1_case_frame',
    'build_bonus1_rating_artifact',
    'build_bonus1_rating_frame',
    'build_bonus1_shenanigans_artifact',
    'build_bonus1_shenanigans_frame',
    'evaluate_shenanigans_payload',
    'find_bonus1_case',
    'load_bonus1_case_registry',
    'load_bonus1_payloads',
    'load_bonus1_rating_config',
    'load_bonus1_shenanigans_config',
    'resolve_question_root',
    'score_bonus1_payload',
]
