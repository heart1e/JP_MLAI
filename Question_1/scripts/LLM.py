"""Codex/ChatGPT helpers for Part 2 balance-sheet forecasting."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_question_root(question_root: Path | None = None) -> Path:
    if question_root is not None:
        return question_root.resolve()
    
    return Path(__file__).resolve().parents[1]


def find_codex_executable() -> str:
    ext_root = Path.home() / '.vscode' / 'extensions'
    patterns = [
        'openai.chatgpt-*/bin/windows-x86_64/codex.exe',
        'openai.chatgpt-*/bin/linux-x86_64/codex',
        'openai.chatgpt-*/bin/darwin-arm64/codex',
        'openai.chatgpt-*/bin/darwin-x86_64/codex',
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(ext_root.glob(pattern), reverse = True))
    direct = shutil.which('codex')
    if direct and 'WindowsApps' not in direct:
        candidates.append(Path(direct))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError('Could not locate a Codex executable.')


def _run_cmd(args: list[str], *, env: dict[str, str], capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, env = env, text = True, capture_output = capture_output)


def _slug_model(model: str) -> str:
    return model.lower().replace('-', '').replace('.', '')


def run_sign_in_smoke_test(
    *,
    test_codex_home: Path | None = None,
    reset_test_home: bool = False,
    do_login: bool = False,
    auto_open_browser: bool = True,
    smoke_prompt: str = 'Reply with exactly one line: Codex ChatGPT auth OK',
    login_url: str = 'https://auth.openai.com/codex/device',
    codex_path: str | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    codex_path = codex_path or find_codex_executable()
    test_home = Path(test_codex_home or (Path.home() / '.codex-free-test'))

    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if echo:
            print(message)

    if reset_test_home and test_home.exists():
        shutil.rmtree(test_home)
    test_home.mkdir(parents = True, exist_ok = True)

    env = os.environ.copy()
    env['CODEX_HOME'] = str(test_home)

    emit(f'isolated_codex_home: {test_home}')
    emit(f'codex_path: {codex_path}')
    emit('This call does not touch your default ~/.codex login.')

    status = _run_cmd([codex_path, 'login', 'status'], env = env)
    status_text = (status.stdout or status.stderr or '').strip()
    if status_text:
        emit('login_status:')
        emit(status_text)

    if do_login:
        emit('Starting device auth for the isolated test home.')
        emit(f'Open this URL if the browser does not pop automatically: {login_url}')
        if auto_open_browser:
            webbrowser.open(login_url)
        process = subprocess.Popen(
            [codex_path, 'login', '--device-auth'],
            env = env,
            text = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            bufsize = 1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            emit(raw_line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'Device auth failed with code {return_code}')
        status = _run_cmd([codex_path, 'login', 'status'], env = env)
        status_text = (status.stdout or status.stderr or '').strip()
        if status_text:
            emit('post_login_status:')
            emit(status_text)

    result: dict[str, Any] = {
        'isolated_codex_home': str(test_home),
        'codex_path': codex_path,
        'logged_in': status.returncode == 0,
        'login_status': status_text,
        'logs': logs,
    }

    if status.returncode != 0:
        emit('Not logged in under the isolated test home.')
        return result

    last_message_path = test_home / 'codex_chatgpt_smoke_last_message.txt'
    if last_message_path.exists():
        last_message_path.unlink()

    cmd = [
        codex_path,
        'exec',
        '--ephemeral',
        '--skip-git-repo-check',
        '-s',
        'read-only',
        '-o',
        str(last_message_path),
        smoke_prompt,
    ]
    smoke = _run_cmd(cmd, env = env)
    result['smoke_returncode'] = smoke.returncode
    result['smoke_stdout'] = smoke.stdout
    result['smoke_stderr'] = smoke.stderr

    if smoke.stdout.strip():
        emit('stdout:')
        emit(smoke.stdout.strip())
    if smoke.stderr.strip():
        emit('stderr:')
        emit(smoke.stderr.strip())
    smoke.check_returncode()

    if last_message_path.exists():
        last_message = last_message_path.read_text(encoding = 'utf-8')
        result['last_message'] = last_message
        emit('last_message:')
        emit(last_message.strip())
    else:
        result['last_message'] = None
        emit('No last message file was produced.')

    return result


def _series_to_native_mn(row: pd.Series, cols: list[str]) -> dict[str, float]:
    return {col: round(float(row[col]) / 1e6, 2) for col in cols}


def _series_to_feature_dict(row: pd.Series, cols: list[str]) -> dict[str, float]:
    return {col: round(float(row[col]), 6) for col in cols}


def build_prompt_payload(
    example_rows: pd.DataFrame,
    target_row: pd.Series,
    *,
    feature_cols: list[str],
    prev_cols: list[str],
) -> dict[str, Any]:
    examples = []
    for ex_date, ex_row in example_rows.iterrows():
        examples.append({
            'date': str(pd.Timestamp(ex_date).date()),
            'ticker': ex_row['ticker'],
            'statement_frequency': ex_row['stmt_freq'],
            'prev_state_mn': _series_to_native_mn(ex_row, prev_cols),
            'lagged_features': _series_to_feature_dict(ex_row, feature_cols),
            'actual_current_totals_mn': {
                'Total Liabilities': round(float(ex_row['Total Liabilities']) / 1e6, 2),
                'Total Equity': round(float(ex_row['Total Equity']) / 1e6, 2),
                'Total Assets': round(float(ex_row['Total Assets']) / 1e6, 2),
            },
        })

    return {
        'task': 'Forecast current-period Total Liabilities and Total Equity from lagged balance-sheet state and financial drivers.',
        'units': "All monetary values are in the company's native reporting currency, millions.",
        'return_format': 'JSON only',
        'examples': examples,
        'target': {
            'date': str(pd.Timestamp(target_row.name).date()),
            'ticker': target_row['ticker'],
            'statement_frequency': target_row['stmt_freq'],
            'prev_state_mn': _series_to_native_mn(target_row, prev_cols),
            'lagged_features': _series_to_feature_dict(target_row, feature_cols),
        },
        'instruction': 'Use only the information in the examples and target row. Return point forecasts for Total Liabilities and Total Equity in millions. Do not add commentary.',
    }


def build_codex_prompt(
    example_rows: pd.DataFrame,
    target_row: pd.Series,
    *,
    feature_cols: list[str],
    prev_cols: list[str],
) -> str:
    payload = build_prompt_payload(
        example_rows,
        target_row,
        feature_cols = feature_cols,
        prev_cols = prev_cols,
    )
    return json.dumps(payload, indent = 2, ensure_ascii = False)


def _run_codex_json(
    *,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    codex_path: str,
    codex_home: Path,
    model: str,
    reasoning: str,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    env = os.environ.copy()
    env['CODEX_HOME'] = str(codex_home)
    status = subprocess.run([codex_path, 'login', 'status'], env = env, text = True, capture_output = True)
    if status.returncode != 0:
        raise RuntimeError('The isolated Codex test home is not logged in. Run the sign-in cell with DO_LOGIN = True first.')

    if output_path.exists():
        output_path.unlink()

    cmd = [
        codex_path,
        'exec',
        '-m',
        model,
        '-c',
        f'model_reasoning_effort="{reasoning}"',
        '--ephemeral',
        '--skip-git-repo-check',
        '-s',
        'read-only',
        '--output-schema',
        str(schema_path),
        '-o',
        str(output_path),
        prompt,
    ]
    result = subprocess.run(cmd, env = env, text = True, capture_output = True, timeout = timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(f'Codex forecast call failed. stdout={result.stdout} stderr={result.stderr}')
    return json.loads(output_path.read_text(encoding = 'utf-8').strip())


def build_llm_context(
    *,
    question_root: Path | None,
    aligned: pd.DataFrame,
    train_mask,
    val_mask,
    test_mask,
    bs_predicate,
    target_le: list[str],
    feature_cols: list[str],
    prev_cols: list[str],
    model: str = 'gpt-5.4-mini',
    reasoning: str = 'medium',
    max_examples: int = 2,
    force_rerun: bool = False,
    llm_weight_grid: list[float] | None = None,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    question_root = resolve_question_root(question_root)
    output_dir = question_root / 'output'
    output_dir.mkdir(parents = True, exist_ok = True)
    model_slug = _slug_model(model)
    artifact_prefix = f'codex_{model_slug}_{reasoning}'

    aligned_train = aligned.iloc[train_mask].copy()
    aligned_val = aligned.iloc[val_mask].copy()
    aligned_test = aligned.iloc[test_mask].copy()
    part1_val_pred = pd.DataFrame(bs_predicate[val_mask], columns = target_le, index = aligned_val.index)
    part1_test_pred = pd.DataFrame(bs_predicate[test_mask], columns = target_le, index = aligned_test.index)

    schema_path = output_dir / 'codex_bs_forecast_schema.json'
    schema = {
        'type': 'object',
        'properties': {
            'pred_total_liabilities_mn': {'type': 'number'},
            'pred_total_equity_mn': {'type': 'number'},
        },
        'required': ['pred_total_liabilities_mn', 'pred_total_equity_mn'],
        'additionalProperties': False,
    }
    schema_path.write_text(json.dumps(schema, indent = 2), encoding = 'utf-8')

    return {
        'question_root': question_root,
        'output_dir': output_dir,
        'model': model,
        'label': f'Codex {model}',
        'reasoning': reasoning,
        'codex_home': Path(codex_home or (Path.home() / '.codex-free-test')),
        'max_examples': max_examples,
        'force_rerun': force_rerun,
        'llm_weight_grid': llm_weight_grid or np.round(np.arange(0.5, 1.001, 0.05), 2).tolist(),
        'schema_path': schema_path,
        'test_pred_path': output_dir / f'{artifact_prefix}_oos_llm_bs_predictions.json',
        'val_pred_path': output_dir / f'{artifact_prefix}_val_llm_bs_predictions.json',
        'compare_path': output_dir / f'{artifact_prefix}_oos_comparison.json',
        'weight_search_path': output_dir / f'{artifact_prefix}_val_weight_search.json',
        'robustness_path': output_dir / f'{artifact_prefix}_robustness.json',
        'artifact_prefix': artifact_prefix,
        'feature_cols': list(feature_cols),
        'prev_cols': list(prev_cols),
        'target_le': list(target_le),
        'aligned_train': aligned_train,
        'aligned_val': aligned_val,
        'aligned_test': aligned_test,
        'part1_val_pred': part1_val_pred,
        'part1_test_pred': part1_test_pred,
    }


def _forecast_llm_split(
    *,
    split_frame: pd.DataFrame,
    aligned_train: pd.DataFrame,
    feature_cols: list[str],
    prev_cols: list[str],
    cache_path: Path,
    split_name: str,
    codex_path: str,
    schema_path: Path,
    codex_home: Path,
    model: str,
    reasoning: str,
    max_examples: int,
    force_rerun: bool,
) -> pd.DataFrame:
    cached_preds: dict[str, Any] = {}
    if cache_path.exists() and not force_rerun:
        cached_preds = {item['row_key']: item for item in json.loads(cache_path.read_text(encoding = 'utf-8'))}

    rows = []
    for row_pos, (row_idx, row) in enumerate(split_frame.iterrows()):
        row_key = f"{row['ticker']}|{pd.Timestamp(row_idx).date()}"
        if row_key in cached_preds and not force_rerun:
            rows.append(cached_preds[row_key])
            continue

        ticker_history = aligned_train[aligned_train['ticker'] == row['ticker']].sort_index().tail(max_examples)
        if ticker_history.empty:
            raise RuntimeError(f'No train examples available for {row_key}')

        prompt = build_codex_prompt(
            ticker_history,
            row,
            feature_cols = feature_cols,
            prev_cols = prev_cols,
        )
        tmp_output = Path(tempfile.gettempdir()) / f'codex_llm_bs_{split_name}_{row_pos}.json'
        payload = _run_codex_json(
            prompt = prompt,
            schema_path = schema_path,
            output_path = tmp_output,
            codex_path = codex_path,
            codex_home = codex_home,
            model = model,
            reasoning = reasoning,
        )
        rows.append({
            'row_key': row_key,
            'ticker': row['ticker'],
            'date': str(pd.Timestamp(row_idx).date()),
            'pred_total_liabilities': float(payload['pred_total_liabilities_mn']) * 1e6,
            'pred_total_equity': float(payload['pred_total_equity_mn']) * 1e6,
            'examples_used': int(ticker_history.shape[0]),
            'model': model,
            'reasoning': reasoning,
            'split': split_name,
        })
        print(f'Completed {split_name}: {row_key}')

    rows = sorted(rows, key = lambda item: (item['ticker'], item['date']))
    cache_path.write_text(json.dumps(rows, indent = 2), encoding = 'utf-8')
    pred = pd.DataFrame(rows)
    pred['date'] = pd.to_datetime(pred['date'])
    return pred.sort_values(['ticker', 'date']).reset_index(drop = True)


def build_part1_view(split_frame: pd.DataFrame, split_pred: pd.DataFrame) -> pd.DataFrame:
    base = split_frame.reset_index().rename(columns = {'index': 'date'}).copy()
    base['date'] = pd.to_datetime(base['date'])
    base = base.sort_values(['ticker', 'date']).reset_index(drop = True)
    pred = split_pred.reset_index().rename(columns = {
        'index': 'date',
        'Total Liabilities': 'part1_total_liabilities',
        'Total Equity': 'part1_total_equity',
    })
    pred['date'] = pd.to_datetime(pred['date'])
    return pd.concat([
        base[['ticker', 'date']],
        pred[['part1_total_liabilities', 'part1_total_equity']],
    ], axis = 1)


def build_comparison_base(split_frame: pd.DataFrame, split_pred: pd.DataFrame, llm_pred: pd.DataFrame) -> pd.DataFrame:
    base = split_frame.reset_index().rename(columns = {'index': 'date'}).copy()
    base['date'] = pd.to_datetime(base['date'])
    base = base.sort_values(['ticker', 'date']).reset_index(drop = True)
    part1_view = build_part1_view(split_frame, split_pred)
    out = base.merge(part1_view, on = ['ticker', 'date'], how = 'left')
    out = out.merge(llm_pred, on = ['ticker', 'date'], how = 'left')
    out['llm_total_liabilities'] = out['pred_total_liabilities']
    out['llm_total_equity'] = out['pred_total_equity']
    return out


def run_llm_baseline(context: dict[str, Any], codex_path: str | None = None) -> dict[str, Any]:
    codex_path = codex_path or find_codex_executable()
    llm_val_pred = _forecast_llm_split(
        split_frame = context['aligned_val'],
        aligned_train = context['aligned_train'],
        feature_cols = context['feature_cols'],
        prev_cols = context['prev_cols'],
        cache_path = context['val_pred_path'],
        split_name = 'val',
        codex_path = codex_path,
        schema_path = context['schema_path'],
        codex_home = context['codex_home'],
        model = context['model'],
        reasoning = context['reasoning'],
        max_examples = context['max_examples'],
        force_rerun = context['force_rerun'],
    )
    llm_test_pred = _forecast_llm_split(
        split_frame = context['aligned_test'],
        aligned_train = context['aligned_train'],
        feature_cols = context['feature_cols'],
        prev_cols = context['prev_cols'],
        cache_path = context['test_pred_path'],
        split_name = 'test',
        codex_path = codex_path,
        schema_path = context['schema_path'],
        codex_home = context['codex_home'],
        model = context['model'],
        reasoning = context['reasoning'],
        max_examples = context['max_examples'],
        force_rerun = context['force_rerun'],
    )

    context = dict(context)
    context.update({
        'codex_path': codex_path,
        'llm_val_pred': llm_val_pred,
        'llm_test_pred': llm_test_pred,
        'comparison_val': build_comparison_base(context['aligned_val'], context['part1_val_pred'], llm_val_pred),
        'comparison_test': build_comparison_base(context['aligned_test'], context['part1_test_pred'], llm_test_pred),
    })
    return context


def run_llm_robustness(
    context: dict[str, Any],
    *,
    runs: int = 5,
    timeout_seconds: int = 600,
    max_retries: int = 2,
    force_rerun: bool = False,
    preferred_ticker: str = 'AAPL',
) -> tuple[dict[str, Any], pd.DataFrame]:
    robustness_path = context['robustness_path']
    codex_path = context.get('codex_path') or find_codex_executable()

    if robustness_path.exists() and not force_rerun:
        artifact = json.loads(robustness_path.read_text(encoding = 'utf-8'))
        return artifact, pd.DataFrame([artifact['summary']])

    robustness_pool = context['aligned_test'][context['aligned_test']['ticker'] == preferred_ticker]
    if robustness_pool.empty:
        robustness_target = context['aligned_test'].sort_values(['ticker']).iloc[0]
    else:
        robustness_target = robustness_pool.sort_index().iloc[0]

    robustness_examples = context['aligned_train'][context['aligned_train']['ticker'] == robustness_target['ticker']].sort_index().tail(context['max_examples'])
    robustness_prompt = build_codex_prompt(
        robustness_examples,
        robustness_target,
        feature_cols = context['feature_cols'],
        prev_cols = context['prev_cols'],
    )
    codex_cli_version = subprocess.run([codex_path, '--version'], text = True, capture_output = True, check = True).stdout.strip()

    def run_with_retry(prompt: str, output_path: Path) -> dict[str, Any]:
        for attempt in range(1, max_retries + 1):
            try:
                return _run_codex_json(
                    prompt = prompt,
                    schema_path = context['schema_path'],
                    output_path = output_path,
                    codex_path = codex_path,
                    codex_home = context['codex_home'],
                    model = context['model'],
                    reasoning = context['reasoning'],
                    timeout_seconds = timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                if attempt == max_retries:
                    raise
        raise RuntimeError('Unexpected robustness execution state')

    robustness_rows = []
    for run_id in range(1, runs + 1):
        tmp_output = Path(tempfile.gettempdir()) / f'codex_llm_robustness_{run_id}.json'
        payload = run_with_retry(robustness_prompt, tmp_output)
        pred_liab = float(payload['pred_total_liabilities_mn']) * 1e6
        pred_equity = float(payload['pred_total_equity_mn']) * 1e6
        robustness_rows.append({
            'run_id': run_id,
            'pred_total_liabilities': pred_liab,
            'pred_total_equity': pred_equity,
            'pred_total_assets': pred_liab + pred_equity,
        })

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_summary = pd.DataFrame([
        {
            'model': context['model'],
            'model_reasoning_effort': context['reasoning'],
            'tool_version': codex_cli_version,
            'sample_ticker': robustness_target['ticker'],
            'sample_date': str(pd.Timestamp(robustness_target.name).date()),
            'runs': runs,
            'liab_mean': robustness_df['pred_total_liabilities'].mean(),
            'liab_std': robustness_df['pred_total_liabilities'].std(ddof = 0),
            'equity_mean': robustness_df['pred_total_equity'].mean(),
            'equity_std': robustness_df['pred_total_equity'].std(ddof = 0),
            'assets_mean': robustness_df['pred_total_assets'].mean(),
            'assets_std': robustness_df['pred_total_assets'].std(ddof = 0),
        }
    ])

    artifact = {
        'model': context['model'],
        'model_reasoning_effort': context['reasoning'],
        'tool_version': codex_cli_version,
        'prompt_template': 'Section 7.2 balance-sheet forecast prompt',
        'sample_ticker': robustness_target['ticker'],
        'sample_date': str(pd.Timestamp(robustness_target.name).date()),
        'runs': robustness_rows,
        'summary': json.loads(robustness_summary.to_json(orient = 'records'))[0],
    }
    robustness_path.write_text(json.dumps(artifact, indent = 2), encoding = 'utf-8')
    return artifact, robustness_summary


__all__ = [
    'build_comparison_base',
    'build_llm_context',
    'build_part1_view',
    'find_codex_executable',
    'run_llm_baseline',
    'run_llm_robustness',
    'run_sign_in_smoke_test',
]
