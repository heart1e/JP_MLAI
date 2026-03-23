"""Notebook infrastructure helpers for Question 1."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PLOT_THEME: dict[str, str] = {}
PLOT_SERIES: dict[str, str] = {}
PLOT_STYLE_CONFIG: dict[str, Any] = {}


def ensure_distutils_compat() -> None:
    """Patch the distutils import used by tensorflow_probability on Python 3.13."""

    try:
        import distutils.version  # type: ignore  # noqa: F401

    except ModuleNotFoundError:
        import setuptools._distutils.version as _distutils_version

        distutils = types.ModuleType('distutils')
        distutils.version = _distutils_version
        sys.modules['distutils'] = distutils
        sys.modules['distutils.version'] = _distutils_version


def apply_plot_theme(style_config: dict[str, Any]) -> None:
    """Apply the shared dark plotting theme in place."""

    PLOT_STYLE_CONFIG.clear()
    PLOT_STYLE_CONFIG.update(dict(style_config))

    PLOT_THEME.clear()
    PLOT_THEME.update(PLOT_STYLE_CONFIG.get('plot_theme', {}))

    PLOT_SERIES.clear()
    PLOT_SERIES.update(PLOT_STYLE_CONFIG.get('series_colors', {}))

    grid_alpha = float(PLOT_STYLE_CONFIG.get('grid_alpha', 0.32))
    grid_linestyle = str(PLOT_STYLE_CONFIG.get('grid_linestyle', '--'))
    legend_frame_alpha = float(PLOT_STYLE_CONFIG.get('legend_frame_alpha', 0.92))

    plt.style.use('default')
    plt.rcParams.update(
        {
            'figure.facecolor': PLOT_THEME.get('figure_face', '#0b0f17'),
            'axes.facecolor': PLOT_THEME.get('axes_face', '#141925'),
            'savefig.facecolor': PLOT_THEME.get('figure_face', '#0b0f17'),
            'axes.edgecolor': PLOT_THEME.get('spine', '#465065'),
            'axes.labelcolor': PLOT_THEME.get('text', '#f3f4f6'),
            'axes.titlecolor': PLOT_THEME.get('text', '#f3f4f6'),
            'xtick.color': PLOT_THEME.get('muted', '#b8c0cc'),
            'ytick.color': PLOT_THEME.get('muted', '#b8c0cc'),
            'grid.color': PLOT_THEME.get('grid', '#313a4d'),
            'text.color': PLOT_THEME.get('text', '#f3f4f6'),
            'legend.facecolor': PLOT_THEME.get('legend_face', '#101621'),
            'legend.edgecolor': PLOT_THEME.get('spine', '#465065'),
            'legend.framealpha': legend_frame_alpha,
            'axes.grid': True,
            'grid.alpha': grid_alpha,
            'grid.linestyle': grid_linestyle,
            'axes.axisbelow': True,
        }
    )


def style_axis(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply the shared panel styling to one axis."""

    ax.set_facecolor(PLOT_THEME.get('panel_face', '#1a2130'))

    for spine in ax.spines.values():
        spine.set_color(PLOT_THEME.get('spine', '#465065'))
        spine.set_linewidth(1.0)

    ax.tick_params(colors = PLOT_THEME.get('muted', '#b8c0cc'))
    ax.grid(True, alpha = float(PLOT_STYLE_CONFIG.get('grid_alpha', 0.28)), linewidth = 0.8)

    panel_title_fontsize = int(PLOT_STYLE_CONFIG.get('panel_title_fontsize', 11))

    if title is not None:
        ax.set_title(
            title,
            color = PLOT_THEME.get('text', '#f3f4f6'),
            fontsize = panel_title_fontsize,
            pad = 10,
            fontweight = 'semibold',
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel, color = PLOT_THEME.get('muted', '#b8c0cc'))

    if ylabel is not None:
        ax.set_ylabel(ylabel, color = PLOT_THEME.get('muted', '#b8c0cc'))


def style_legend(ax, *, fontsize: int = 7) -> None:
    """Apply the shared legend styling if the axis has a legend."""

    legend = ax.legend(fontsize = fontsize)

    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_facecolor(PLOT_THEME.get('legend_face', '#101621'))
    frame.set_edgecolor(PLOT_THEME.get('spine', '#465065'))
    frame.set_linewidth(0.9)

    for text in legend.get_texts():
        text.set_color(PLOT_THEME.get('text', '#f3f4f6'))


def finish_figure(fig, title: str) -> None:
    """Apply the shared figure-level styling."""

    title_fontsize = int(PLOT_STYLE_CONFIG.get('title_fontsize', 14))
    fig.patch.set_facecolor(PLOT_THEME.get('figure_face', '#0b0f17'))
    fig.suptitle(
        title,
        y = 1.02,
        color = PLOT_THEME.get('text', '#f3f4f6'),
        fontsize = title_fontsize,
        fontweight = 'semibold',
    )
    fig.tight_layout()


def load_json_artifact(path: Path) -> Any:
    """Load one JSON artifact with a hard existence check."""

    if not path.exists():
        raise FileNotFoundError(f'Required artifact is missing: {path}')

    return json.loads(path.read_text(encoding = 'utf-8'))


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load one JSON object artifact."""

    payload = load_json_artifact(path)

    if not isinstance(payload, dict):
        raise TypeError(f'Expected a JSON object in {path.name}.')

    return payload


def load_json_list(path: Path) -> list[Any]:
    """Load one JSON array artifact."""

    payload = load_json_artifact(path)

    if not isinstance(payload, list):
        raise TypeError(f'Expected a JSON array in {path.name}.')

    return payload


def load_llm_prediction_artifact(path: Path) -> pd.DataFrame:
    """Load cached LLM predictions as a DataFrame."""

    frame = pd.DataFrame(load_json_list(path))

    if 'date' in frame.columns:
        frame['date'] = pd.to_datetime(frame['date'])

    return frame


def build_cross_summary_frame(cross_payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the Part 2 cross-company payload into a summary frame."""

    rows = []

    for company_id, payload in cross_payload.items():
        if not isinstance(payload, dict):
            continue

        metrics = payload.get('metrics', {})

        if not isinstance(metrics, dict):
            metrics = {}

        rows.append(
            {
                'id': company_id,
                'company': payload.get('company'),
                'net_income': metrics.get('net_income_musd'),
                'cost_to_income': metrics.get('cost_to_income_ratio'),
                'quick_ratio': metrics.get('quick_ratio'),
                'debt_to_equity': metrics.get('debt_to_equity_ratio'),
                'debt_to_assets': metrics.get('debt_to_assets_ratio'),
                'debt_to_capital': metrics.get('debt_to_capital_ratio'),
                'debt_to_ebitda': metrics.get('debt_to_ebitda_ratio'),
                'interest_coverage': metrics.get('interest_coverage_ratio'),
            }
        )

    summary = pd.DataFrame(rows)

    if 'id' in summary.columns:
        summary = summary.sort_values('id').reset_index(drop = True)

    return summary


def load_required_coverage_payloads(question_root: Path) -> dict[str, dict[str, Any]]:
    """Load the in-scope annual-report outputs from the manifest."""

    manifest_cfg = load_json_mapping(question_root / 'config' / 'annual_report_sources.json')
    payloads: dict[str, dict[str, Any]] = {}

    for item in manifest_cfg.get('reports', []):
        if not isinstance(item, dict):
            continue

        if not bool(item.get('required_now', False)):
            continue

        output_path = str(item.get('output_path', '')).replace('Question_1/', '')

        if not output_path:
            continue

        payloads[str(item['id'])] = load_json_mapping(question_root / output_path)

    return payloads


def build_manifest_progress_snapshot(question_root: Path) -> dict[str, Any]:
    """Build a non-mutating manifest status snapshot for display-only notebook runs."""

    manifest_cfg = load_json_mapping(question_root / 'config' / 'annual_report_sources.json')
    project_root = question_root.parent
    rows = []

    for item in manifest_cfg.get('reports', []):
        if not isinstance(item, dict):
            continue

        if not bool(item.get('required_now', False)):
            continue

        pdf_path = project_root / str(item.get('pdf_path', '')) if item.get('pdf_path') else None
        output_path = project_root / str(item.get('output_path', '')) if item.get('output_path') else None

        rows.append(
            {
                'id': item.get('id'),
                'company': item.get('company_name'),
                'action': 'skipped_run__loaded_manifest_state',
                'downloaded': False,
                'status_after': item.get('status'),
                'pdf_exists': bool(pdf_path and pdf_path.exists()),
                'output_exists': bool(output_path and output_path.exists()),
            }
        )

    manifest_progress_df = pd.DataFrame(rows)

    if 'id' in manifest_progress_df.columns:
        manifest_progress_df = manifest_progress_df.sort_values('id').reset_index(drop = True)

    return {
        'manifest_cfg': manifest_cfg,
        'manifest_progress_df': manifest_progress_df,
    }


def require_codex_ready(preflight: dict[str, Any], *, purpose: str) -> None:
    """Raise a clear notebook-facing error when Codex is unavailable."""

    if not bool(preflight.get('codex_found', False)):
        raise EnvironmentError(
            f'Cannot run {purpose}: Codex executable was not found. Install the OpenAI ChatGPT/Codex tooling or add codex to PATH.'
        )

    if not bool(preflight.get('logged_in', False)):
        raise EnvironmentError(
            f"Cannot run {purpose}: Codex is installed but not logged in under {preflight.get('codex_home')}."
        )


__all__ = [
    'PLOT_SERIES',
    'PLOT_STYLE_CONFIG',
    'PLOT_THEME',
    'apply_plot_theme',
    'build_cross_summary_frame',
    'build_manifest_progress_snapshot',
    'ensure_distutils_compat',
    'finish_figure',
    'load_json_artifact',
    'load_json_list',
    'load_json_mapping',
    'load_llm_prediction_artifact',
    'load_required_coverage_payloads',
    'require_codex_ready',
    'style_axis',
    'style_legend',
]
