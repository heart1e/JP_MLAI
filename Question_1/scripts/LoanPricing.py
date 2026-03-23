"""Public-data MVP for Bonus 3 term-loan pricing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen = True)
class Bonus3PricingResult:
    """Typed return payload for Bonus 3 pricing outputs."""

    pricing_df: pd.DataFrame
    pricing_output_path: Path
    market_data: pd.DataFrame
    metrics_df: pd.DataFrame
    metrics_path: Path
    borrower_df: pd.DataFrame
    config_path: Path


@dataclass(frozen = True)
class Bonus3NotebookViews:
    """Typed notebook-facing views for Bonus 3 outputs."""

    result: Bonus3PricingResult
    model_metrics: pd.DataFrame
    pricing_df: pd.DataFrame
    pricing_view: pd.DataFrame
    business_view: pd.DataFrame


def resolve_question_root(question_root: Path | None = None) -> Path:
    """Resolve the Question_1 workspace root."""

    if question_root is not None:

        return question_root.resolve()


    return Path(__file__).resolve().parents[1]


def load_bonus3_config(
                        question_root: Path | None = None,
                        config_filename: str = 'bonus3_pricing_config.json',
                        ) -> tuple[dict[str, Any], Path]:
    """Load the Bonus 3 public-pricing config."""

    question_root = resolve_question_root(question_root)
    config_path = question_root / 'config' / config_filename


    return json.loads(config_path.read_text(encoding = 'utf-8')), config_path


def _fred_csv_url(series_id: str) -> str:
    """Return the direct CSV endpoint for a FRED series."""

    return f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'


def _fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch one FRED series as a tidy time series."""

    df = pd.read_csv(_fred_csv_url(series_id))
    df.columns = ['date', series_id]
    df['date'] = pd.to_datetime(df['date'])
    df[series_id] = pd.to_numeric(df[series_id], errors = 'coerce')


    return df


def fetch_bonus3_public_market_data(
                                    question_root: Path | None = None,
                                    *,
                                    config_filename: str = 'bonus3_pricing_config.json',
                                    force_refresh: bool = False,
                                    ) -> dict[str, Any]:
    """Download and persist public market time series used in Bonus 3."""

    question_root = resolve_question_root(question_root)
    cfg, config_path = load_bonus3_config(question_root, config_filename)
    public_dir = question_root / 'data' / 'public'
    public_dir.mkdir(parents = True, exist_ok = True)
    daily_path = public_dir / 'bonus3_public_market_data_daily.csv'
    monthly_path = public_dir / 'bonus3_public_market_data_monthly.csv'
    series_ids = list(cfg['market_data']['fred_series'].keys())

    if daily_path.exists() and monthly_path.exists() and not force_refresh:
        daily = pd.read_csv(daily_path, parse_dates = ['date']).set_index('date')
        monthly = pd.read_csv(monthly_path, parse_dates = ['date']).set_index('date')

        return {

            'daily': daily,
            'monthly': monthly,
            'daily_path': daily_path,
            'monthly_path': monthly_path,
            'config_path': config_path,
        }

    merged = None

    for series_id in series_ids:
        series_df = _fetch_fred_series(series_id)
        merged = series_df if merged is None else merged.merge(series_df, on = 'date', how = 'outer')

    assert merged is not None

    merged = merged.sort_values('date').set_index('date')
    daily = merged.ffill()
    monthly = daily.resample('ME').last()
    monthly['curve_slope'] = monthly['DGS10'] - monthly['DGS2']
    monthly['dgs5_change_1m'] = monthly['DGS5'].diff()

    daily.reset_index().to_csv(daily_path, index = False)
    monthly.reset_index().to_csv(monthly_path, index = False)


    return {

        'daily': daily,
        'monthly': monthly,
        'daily_path': daily_path,
        'monthly_path': monthly_path,
        'config_path': config_path,
    }


def _time_split_frame(frame: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series frame into train/test by tail holdout."""

    if frame.shape[0] <= holdout_months + 12:
        holdout_months = max(6, min(12, frame.shape[0] // 4))

    split_idx = max(1, frame.shape[0] - holdout_months)


    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def _fit_one_step_model(
                        frame: pd.DataFrame,
                        *,
                        target_col: str,
                        feature_cols: list[str],
                        holdout_months: int,
                        alpha: float = 1.0,
                        ) -> dict[str, Any]:
    """Fit a one-step-ahead Ridge regression model."""

    df = frame.copy()
    df['target_next'] = df[target_col].shift(-1)
    df = df.dropna(subset = feature_cols + ['target_next'])

    train_df, test_df = _time_split_frame(df, holdout_months)
    model = Ridge(alpha = alpha)
    model.fit(train_df[feature_cols], train_df['target_next'])

    train_pred = model.predict(train_df[feature_cols])
    test_pred = model.predict(test_df[feature_cols])
    latest_features = df[feature_cols].iloc[[-1]]
    pred_next = float(model.predict(latest_features)[0])
    current_value = float(df[target_col].iloc[-1])
    residuals = (test_df['target_next'] - test_pred).to_numpy(dtype = float)

    if residuals.size == 0:
        residuals = (train_df['target_next'] - train_pred).to_numpy(dtype = float)

    def mae(actual, pred) -> float:

        return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


    def rmse(actual, pred) -> float:

        return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


    return {

        'model': model,
        'feature_cols': feature_cols,
        'history': df,
        'current_value': current_value,
        'pred_next': pred_next,
        'residuals': residuals,
        'metrics': {
            'target': target_col,
            'train_rows': int(train_df.shape[0]),
            'test_rows': int(test_df.shape[0]),
            'train_mae': mae(train_df['target_next'], train_pred),
            'test_mae': mae(test_df['target_next'], test_pred),
            'train_rmse': rmse(train_df['target_next'], train_pred),
            'test_rmse': rmse(test_df['target_next'], test_pred),
            'latest_date': str(df.index[-1].date()),
            'current_value': current_value,
            'pred_next': pred_next,
        },
    }


def fit_bonus3_public_models(
                            question_root: Path | None = None,
                            *,
                            config_filename: str = 'bonus3_pricing_config.json',
                            force_refresh_data: bool = False,
                            ) -> dict[str, Any]:
    """Fit public-data market proxy models for Bonus 3."""

    question_root = resolve_question_root(question_root)
    cfg, config_path = load_bonus3_config(question_root, config_filename)
    market_data = fetch_bonus3_public_market_data(
        question_root,
        config_filename = config_filename,
        force_refresh = force_refresh_data,
    )

    monthly = market_data['monthly'].copy()

    feature_cols = [
        'DGS2',
        'DGS5',
        'DGS10',
        'curve_slope',
        'AAA10Y',
        'BAMLC0A4CBBB',
        'BAMLH0A0HYM2',
    ]

    holdout_months = int(cfg['market_data'].get('holdout_months', 24))
    model_bundle = {
        'treasury': _fit_one_step_model(
            monthly,
            target_col = 'DGS5',
            feature_cols = feature_cols,
            holdout_months = holdout_months,
        ),
        'spreads': {},
        'monthly': monthly,
        'config_path': config_path,
    }

    for bucket, series_id in cfg['market_data']['target_bucket_map'].items():

        spread_features = feature_cols + [series_id]
        model_bundle['spreads'][bucket] = _fit_one_step_model(
            monthly,
            target_col = series_id,
            feature_cols = spread_features,
            holdout_months = holdout_months,
        )

    metrics_rows = [dict(model_bundle['treasury']['metrics'], model_name = 'treasury_5y')]

    for bucket, model_info in model_bundle['spreads'].items():
        row = dict(model_info['metrics'])
        row['model_name'] = f'spread_bucket_{bucket}'
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    output_path = question_root / 'output' / 'bonus3_public_model_metrics.json'
    output_path.write_text(json.dumps(metrics_rows, indent = 2), encoding = 'utf-8')
    model_bundle['metrics_df'] = metrics_df
    model_bundle['metrics_path'] = output_path


    return model_bundle


def _load_json(path: Path) -> dict[str, Any]:

    return json.loads(path.read_text(encoding = 'utf-8'))


def load_bonus3_borrower_inputs(question_root: Path | None = None) -> pd.DataFrame:
    """Load borrower risk inputs from Bonus 1 and Bonus 2 outputs."""

    question_root = resolve_question_root(question_root)
    output_root = question_root / 'output'
    rating_payload = _load_json(output_root / 'bonus1_internal_rating_baseline.json')
    shenanigans_payload = _load_json(output_root / 'bonus1_shenanigans_scan.json')
    bankruptcy_payload = _load_json(output_root / 'bonus1_bankruptcy_oos_summary.json')
    warning_payload = _load_json(output_root / 'bonus2_bankruptcy_warning_summary.json')

    scan_lookup = {item['id']: item for item in shenanigans_payload['scans']}
    warning_lookup = {item['id']: item for item in warning_payload['records']}
    borrower_rows = []
    seen_ids: set[str] = set()

    for rating_row in rating_payload['ratings']:

        borrower_id = rating_row['id']
        shenanigans_row = scan_lookup.get(borrower_id, {})
        warning_row = warning_lookup.get(borrower_id, {})
        warning_analysis = warning_row.get('analysis', {})
        
        borrower_rows.append(
            {
                'id': borrower_id,
                'company': rating_row['company'],
                'statement_year': rating_row['statement_year'],
                'source_group': 'annual_report_core',
                'rating': rating_row['rating'],
                'normalized_score': float(rating_row['normalized_score']),
                'distress_flag_count': int(rating_row['distress_flag_count']),
                'shenanigans_points': int(shenanigans_row.get('total_points', 0)),
                'shenanigans_trigger_count': int(shenanigans_row.get('trigger_count', 0)),
                'shenanigans_warning_band': shenanigans_row.get('warning_band', 'low'),
                'warning_count': len(warning_analysis.get('warnings', [])),
                'warning_severity': warning_analysis.get('overall_risk_level', 'low'),
                'top_warning_tags': warning_analysis.get('top_warning_tags', []),
            }
        )

        seen_ids.add(borrower_id)

    for record in bankruptcy_payload['records']:
        borrower_id = record['id']

        if borrower_id in seen_ids:

            continue

        rating_row = record['rating_result']
        shenanigans_row = record['shenanigans_result']
        warning_row = warning_lookup.get(borrower_id, {})
        warning_analysis = warning_row.get('analysis', {})
        
        borrower_rows.append(
            {
                'id': borrower_id,
                'company': rating_row['company'],
                'statement_year': rating_row['statement_year'],
                'source_group': 'bankruptcy_oos',
                'rating': rating_row['rating'],
                'normalized_score': float(rating_row['normalized_score']),
                'distress_flag_count': int(rating_row['distress_flag_count']),
                'shenanigans_points': int(shenanigans_row.get('total_points', 0)),
                'shenanigans_trigger_count': int(shenanigans_row.get('trigger_count', 0)),
                'shenanigans_warning_band': shenanigans_row.get('warning_band', 'low'),
                'warning_count': len(warning_analysis.get('warnings', [])),
                'warning_severity': warning_analysis.get('overall_risk_level', 'low'),
                'top_warning_tags': warning_analysis.get('top_warning_tags', []),
            }
        )
        seen_ids.add(borrower_id)

    df = pd.DataFrame(borrower_rows).sort_values(['source_group', 'rating', 'id']).reset_index(drop = True)


    return df


def _borrower_addon_bps(row: pd.Series, cfg: dict[str, Any]) -> dict[str, float]:
    """Compute borrower-specific risk surcharges on top of market spreads."""

    adj_cfg = cfg['borrower_adjustments']
    rating = row['rating']
    base_addon = float(adj_cfg['rating_addon_bps'][rating])
    score_raw = -float(row['normalized_score']) * float(adj_cfg['normalized_score_scale_bps'])
    score_floor, score_cap = adj_cfg['normalized_score_clip_bps']
    score_addon = float(np.clip(score_raw, score_floor, score_cap))
    distress_addon = float(row['distress_flag_count']) * float(adj_cfg['distress_flag_bps'])
    shenanigans_addon = float(row['shenanigans_points']) * float(adj_cfg['shenanigans_point_bps'])
    warning_addon = float(row['warning_count']) * float(adj_cfg['warning_count_bps'])
    severity_addon = float(adj_cfg['severity_addon_bps'].get(row['warning_severity'], 0.0))
    total_addon = base_addon + score_addon + distress_addon + shenanigans_addon + warning_addon + severity_addon


    return {

        'base_addon_bps': base_addon,
        'score_addon_bps': score_addon,
        'distress_addon_bps': distress_addon,
        'shenanigans_addon_bps': shenanigans_addon,
        'warning_addon_bps': warning_addon,
        'severity_addon_bps': severity_addon,
        'total_addon_bps': total_addon,
    }


def _determine_pricing_decision(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Attach a business decision layer on top of the indicative quote."""

    decision_cfg = cfg['decision_layer']
    rating = row['rating']
    severity = row['warning_severity']
    distress_flag_count = int(row['distress_flag_count'])
    shenanigans_points = int(row['shenanigans_points'])

    decline = (
        rating in decision_cfg['decline_rating']
        or severity in decision_cfg['decline_warning_severity']
        or distress_flag_count >= int(decision_cfg['decline_distress_flag_count'])
        or shenanigans_points >= int(decision_cfg['decline_shenanigans_points'])
    )
    manual_review = (
        rating in decision_cfg['manual_review_rating']
        or severity in decision_cfg['manual_review_warning_severity']
        or distress_flag_count >= int(decision_cfg['manual_review_distress_flag_count'])
        or shenanigans_points >= int(decision_cfg['manual_review_shenanigans_points'])
    )

    if decline:
        decision = 'decline'

    elif manual_review:
        decision = 'manual_review'

    else:
        decision = 'quote'

    indicative_only = decision != 'quote'
    reasons = []

    if rating in decision_cfg['decline_rating'] or rating in decision_cfg['manual_review_rating']:
        reasons.append(f'rating={rating}')

    if severity in decision_cfg['decline_warning_severity'] or severity in decision_cfg['manual_review_warning_severity']:
        reasons.append(f'warning_severity={severity}')

    if distress_flag_count > 0:
        reasons.append(f'distress_flags={distress_flag_count}')

    if shenanigans_points > 0:
        reasons.append(f'shenanigans_points={shenanigans_points}')

    return {
        'lending_decision': decision,
        'indicative_only': indicative_only,
        'decision_reason': ', '.join(reasons) if reasons else 'no hard risk gate triggered',
        'business_guidance': decision_cfg['messages'][decision],
    }


def _price_fixed_bullet_loan(
                            coupon_rate_pct: float,
                            discount_rate_pct: float,
                            *,
                            maturity_years: int,
                            months_elapsed: int,
                            par_value: float,
                            ) -> float:
    """Price a stylized fixed-rate bullet term loan."""

    total_months = max(1, maturity_years * 12)
    remaining_months = max(1, total_months - months_elapsed)
    coupon_cash = par_value * coupon_rate_pct / 100.0 / 12.0
    monthly_yield = max(-0.99, discount_rate_pct / 100.0 / 12.0)
    discount_factors = np.power(1.0 + monthly_yield, np.arange(1, remaining_months + 1))
    coupon_pv = float(np.sum(coupon_cash / discount_factors))
    principal_pv = float(par_value / discount_factors[-1])


    return coupon_pv + principal_pv


def build_bonus3_pricing_recommendations(
                                        question_root: Path | None = None,
                                        *,
                                        config_filename: str = 'bonus3_pricing_config.json',
                                        force_refresh_data: bool = False,
                                        bootstrap_seed: int = 42,
                                        ) -> Bonus3PricingResult:
    """Build the public-data Bonus 3 MVP and persist the outputs."""

    question_root = resolve_question_root(question_root)
    cfg, config_path = load_bonus3_config(question_root, config_filename)
    model_bundle = fit_bonus3_public_models(
        question_root,
        config_filename = config_filename,
        force_refresh_data = force_refresh_data,
    )

    borrowers = load_bonus3_borrower_inputs(question_root)
    output_root = question_root / 'output'
    pricing_output_path = output_root / 'bonus3_public_pricing_recommendations.json'
    par_value = float(cfg['market_data'].get('par_value', 100.0))
    maturity_years = int(cfg['market_data'].get('benchmark_maturity_years', 5))
    bootstrap_draws = int(cfg['market_data'].get('bootstrap_draws', 500))
    current_treasury_pct = float(model_bundle['treasury']['current_value'])
    pred_treasury_pct = float(model_bundle['treasury']['pred_next'])
    treasury_residuals = np.asarray(model_bundle['treasury']['residuals'], dtype = float)
    rng = np.random.default_rng(bootstrap_seed)
    pricing_rows = []

    for _, row in borrowers.iterrows():

        model_bucket = row['rating']
        spread_model = model_bundle['spreads'][model_bucket]
        current_market_spread_bps = float(spread_model['current_value'] * 100.0)
        pred_market_spread_bps = float(spread_model['pred_next'] * 100.0)
        addons = _borrower_addon_bps(row, cfg)
        decision_meta = _determine_pricing_decision(row, cfg)

        current_spread_bps = current_market_spread_bps + addons['total_addon_bps']
        pred_spread_bps = pred_market_spread_bps + addons['total_addon_bps']
        suggested_rate_pct = current_treasury_pct + current_spread_bps / 100.0
        resale_yield_pct = pred_treasury_pct + pred_spread_bps / 100.0
        
        price_in_one_month = _price_fixed_bullet_loan(
            coupon_rate_pct = suggested_rate_pct,
            discount_rate_pct = resale_yield_pct,
            maturity_years = maturity_years,
            months_elapsed = 1,
            par_value = par_value,
        )

        spread_residuals = np.asarray(spread_model['residuals'], dtype = float) * 100.0

        if spread_residuals.size == 0:
            spread_residuals = np.array([0.0])

        if treasury_residuals.size == 0:
            treasury_residuals = np.array([0.0])

        spread_draws = pred_market_spread_bps + rng.choice(spread_residuals, size = bootstrap_draws, replace = True) + addons['total_addon_bps']
        treasury_draws = pred_treasury_pct + rng.choice(treasury_residuals, size = bootstrap_draws, replace = True)
        price_draws = [
            _price_fixed_bullet_loan(
                coupon_rate_pct = suggested_rate_pct,
                discount_rate_pct = float(treasury_draw + spread_draw / 100.0),
                maturity_years = maturity_years,
                months_elapsed = 1,
                par_value = par_value,
            )
            for treasury_draw, spread_draw in zip(treasury_draws, spread_draws)
        ]

        pricing_rows.append(
            {
                'id': row['id'],
                'company': row['company'],
                'source_group': row['source_group'],
                'rating': row['rating'],
                'model_bucket': model_bucket,
                'normalized_score': float(row['normalized_score']),
                'distress_flag_count': int(row['distress_flag_count']),
                'shenanigans_points': int(row['shenanigans_points']),
                'warning_count': int(row['warning_count']),
                'warning_severity': row['warning_severity'],
                'top_warning_tags': list(row['top_warning_tags']),
                'lending_decision': decision_meta['lending_decision'],
                'indicative_only': decision_meta['indicative_only'],
                'decision_reason': decision_meta['decision_reason'],
                'business_guidance': decision_meta['business_guidance'],
                'current_treasury_pct': current_treasury_pct,
                'predicted_treasury_pct_1m': pred_treasury_pct,
                'current_market_spread_bps': current_market_spread_bps,
                'predicted_market_spread_bps_1m': pred_market_spread_bps,
                'borrower_addon_bps': addons['total_addon_bps'],
                'spread_components_bps': addons,
                'suggested_spread_bps': current_spread_bps,
                'suggested_interest_rate_pct': suggested_rate_pct,
                'predicted_spread_bps_1m': pred_spread_bps,
                'predicted_resale_yield_pct_1m': resale_yield_pct,
                'predicted_price_in_1m': price_in_one_month,
                'price_ci_95_low': float(np.quantile(price_draws, 0.025)),
                'price_ci_95_high': float(np.quantile(price_draws, 0.975)),
            }
        )

    pricing_df = pd.DataFrame(pricing_rows).sort_values(
        ['suggested_spread_bps', 'predicted_price_in_1m', 'id'],
        ascending = [False, True, True],
    ).reset_index(drop = True)

    pricing_output = {
        'method': 'public-data proxy term-loan pricing MVP',
        'config_path': str(config_path.as_posix()),
        'market_data_path': str((question_root / 'data' / 'public' / 'bonus3_public_market_data_monthly.csv').as_posix()),
        'pricing_rows': json.loads(pricing_df.to_json(orient = 'records')),
    }
    pricing_output_path.write_text(json.dumps(pricing_output, indent = 2), encoding = 'utf-8')


    return Bonus3PricingResult(
        pricing_df = pricing_df,
        pricing_output_path = pricing_output_path,
        market_data = model_bundle['monthly'],
        metrics_df = model_bundle['metrics_df'],
        metrics_path = model_bundle['metrics_path'],
        borrower_df = borrowers,
        config_path = config_path,
    )


def build_bonus3_notebook_views(
                               question_root: Path | None = None,
                               *,
                               config_filename: str = 'bonus3_pricing_config.json',
                               force_refresh_data: bool = False,
                               bootstrap_seed: int = 42,
                               use_cached_output: bool = False,
                               ) -> Bonus3NotebookViews:
    """Build typed notebook display frames for Bonus 3."""

    question_root = resolve_question_root(question_root)

    if use_cached_output:
        _, config_path = load_bonus3_config(question_root, config_filename)
        metrics_path = question_root / 'output' / 'bonus3_public_model_metrics.json'
        pricing_output_path = question_root / 'output' / 'bonus3_public_pricing_recommendations.json'
        pricing_output = json.loads(pricing_output_path.read_text(encoding = 'utf-8'))
        metrics_df = pd.DataFrame(json.loads(metrics_path.read_text(encoding = 'utf-8')))
        pricing_df = pd.DataFrame(pricing_output['pricing_rows'])
        market_data_path = Path(pricing_output['market_data_path'])
        market_data = pd.read_csv(market_data_path) if market_data_path.exists() else pd.DataFrame()
        borrower_df = pricing_df.copy()
        result = Bonus3PricingResult(
            pricing_df = pricing_df,
            pricing_output_path = pricing_output_path,
            market_data = market_data,
            metrics_df = metrics_df,
            metrics_path = metrics_path,
            borrower_df = borrower_df,
            config_path = config_path,
        )

    else:
        result = build_bonus3_pricing_recommendations(
            question_root = question_root,
            config_filename = config_filename,
            force_refresh_data = force_refresh_data,
            bootstrap_seed = bootstrap_seed,
        )
    pricing_cols = [
        'id',
        'company',
        'source_group',
        'rating',
        'lending_decision',
        'indicative_only',
        'warning_severity',
        'suggested_spread_bps',
        'suggested_interest_rate_pct',
        'predicted_price_in_1m',
        'price_ci_95_low',
        'price_ci_95_high',
    ]
    business_cols = [
        'id',
        'company',
        'rating',
        'lending_decision',
        'decision_reason',
        'business_guidance',
    ]

    pricing_df = result.pricing_df.copy()

    return Bonus3NotebookViews(
        result = result,
        model_metrics = result.metrics_df.copy(),
        pricing_df = pricing_df,
        pricing_view = pricing_df.reindex(columns = pricing_cols).copy(),
        business_view = pricing_df.reindex(columns = business_cols).copy(),
    )


__all__ = [
    'Bonus3NotebookViews',
    'Bonus3PricingResult',
    'build_bonus3_notebook_views',
    'build_bonus3_pricing_recommendations',
    'fetch_bonus3_public_market_data',
    'fit_bonus3_public_models',
    'load_bonus3_borrower_inputs',
    'load_bonus3_config',
    'resolve_question_root',
]
