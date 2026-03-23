# JP_MLAI - Question 1

## Overview
This repository contains a complete working delivery for **Question 1** of the JPM MLCOE TSRL 2026 intern interview project.

The project is organized as one main analytical notebook plus a small set of reusable scripts:
- **Part 1**: balance-sheet and earnings forecasting from cleaned financial statement histories
- **Part 2**: annual-report extraction, LLM baseline comparison, ensemble analysis, and CFO/CEO recommendation
- **Bonus 1**: annual-report-only internal rating baseline plus quantitative shenanigans scan
- **Bonus 2**: warning paragraph retrieval plus Codex-based classification
- **Bonus 3**: public-data proxy term-loan pricing MVP

The repo is now structured as a reusable project rather than a single long notebook. Heavy logic lives in `Question_1/scripts/`, while the notebook focuses on orchestration, evaluation, plots, and question-facing writeup.

## Part 1 Balance Sheet Forecasting
This repository implements Part 1 of the JPM MLCOE TSRL 2026 Question 1: a TensorFlow + TensorFlow Probability (TFP) model that forecasts balance sheet items while enforcing accounting identities, and produces a probabilistic earnings forecast.

### Literature Alignment (Brief)
- **Pareja (2007, 2009)**: balance-sheet forecasting without plugs; enforce internal consistency and avoid circularity.
- **Pelaez (2011)**: analytical handling of circularity constraints in financial statements.
- **Shahnazarian (2004)** and **Samonas**: dynamic simulation view of corporate financials, supporting a time-series formulation.

### Mathematical Formulation (Summary)
$$
\begin{aligned}
& y_{t+1} = f_{\theta}(x_t, y_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma) \\
& y_t = \begin{bmatrix} L_t \\ E_t \end{bmatrix}, \quad A_t = L_t + E_t \\
& x_t = \{\mathrm{DSO}_t, \mathrm{DPO}_t, \mathrm{DIH}_t, \text{margins}, \text{growth}, \text{seasonality}, y_t\} \\
& A_{t+1} = L_{t+1} + E_{t+1}
\end{aligned}
$$

### Data
- **Source**: Yahoo Finance (`yfinance`) plus a canonical cleaning layer.
- **Universe / paths / aliases**: controlled in `Question_1/config/pipeline_config.json`.
- **Clean Part 1 inputs**: `Question_1/data/clean/yfinance/`.
- **Raw refresh utilities**: `Question_1/scripts/fetch_yf_statements.py`, `Question_1/scripts/fetch_sec_xbrl.py`, `Question_1/scripts/clean_financials.py`.

### Model (TensorFlow + TFP)
- **AlgebraicBS layer**: predicts liabilities and equity while enforcing identity constraints.
- **Earnings head**: outputs Normal distribution parameters (`loc`, `scale`).
- **Loss**:
  - Balance sheet: MAE
  - Earnings: TFP negative log-likelihood (NLL)

### Training and Evaluation
- **Split**: time-ordered train / validation / test to reduce look-ahead leakage.
- **Scaling**: z-score normalization for stable optimization.
- **Evaluation**: per-ticker metrics, identity residuals, and diagnostic plots.

### Out-of-Sample (OOS) Test
The notebook includes a dedicated OOS evaluation section that:
- Holds out the tail of the time series as test.
- Reports MAE for liabilities, equity, assets, and net income.
- Computes identity residuals on OOS predictions.

### Answers to Part 1 Prompts (Short Form)
- **Is it a time series?** Yes. Each period depends on lagged balance sheet state and exogenous drivers.
- **How to handle identities?** Hard constraints: predict liabilities / equity and derive assets.
- **How to test quality?** Time-ordered validation plus OOS holdout metrics and residual checks.
- **Can we forecast earnings?** Yes, via a probabilistic TFP head (`Normal` NLL).
- **What improves the model?** More features, better sequence models (GRU / Transformer), regime-aware splits, hierarchical or multi-task learning, calibrated uncertainty, and richer macro covariates.

### Testing Plan and Results
- **Unit tests**: balance-sheet identity residual and feature engineering sanity checks.
- **Integration tests**: end-to-end forward pass and output shape validation.
- **Execution**: running the notebook executes tests and prints results.

## Part 2 (Core)
This repository includes the core Part 2 deliverables requested in `Project_Instruction/Intern interview 2026 question 1 ver 3.docx`.

- **Chosen LLM baseline**: `gpt-5.4-mini` via Codex CLI with `model_reasoning_effort = medium`.
- **Same data as Part 1**: the LLM balance-sheet forecast uses the same time-ordered train / validation / test rows built in `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb`.
- **LLM vs Part 1**: the notebook compares `Part1`, `Codex gpt-5.4-mini`, and ensemble forecasts on the same OOS rows using MAE, RMSE, identity residual, and normalized MAE.
- **Cross-currency mitigation**: normalized MAE scales each absolute error by current-period total assets, which avoids mixing raw USD / HKD / CHF / RMB magnitudes in one unscaled score.
- **Ensemble search**: validation-time tuning searches `LLM weight` from `0.50` to `1.00` in `0.05` steps. The best validation weight is `1.00`, so the tuned ensemble collapses to the pure LLM baseline instead of improving on it.
- **Management recommendation**: the notebook generates a CFO / CEO memo for one analyzed company (`MCD` in the current run) based on the tuned forecast outputs.
- **Annual-report extraction**: GM, LVMH, Tencent, Alibaba, JPMorgan Chase, Exxon Mobil, Volkswagen, Microsoft, and Alphabet / Google extraction outputs are stored under `Question_1/output/`.

### Part 2 Results Snapshot
- **Primary Part 2 baseline**: `Codex gpt-5.4-mini`
- **Best validation LLM weight**: `1.00`
- **Test mean normalized MAE**:
  - `Codex gpt-5.4-mini`: `0.0547`
  - `Ensemble tuned llm=1.00`: `0.0547`
  - `Ensemble 50/50`: `1.6465`
  - `Part1`: `3.2640`

### Part 2 Reproducibility and Versioning
- **Notebook sections**:
  - `Section 6`: annual-report extraction
  - `Section 7`: LLM baseline
  - `Section 8`: Part 1 vs LLM vs ensemble comparison
  - `Section 9`: closeout summary, recommendation, and robustness
- **Model / tool identifiers used in the current implementation**:
  - Model: `gpt-5.4-mini`
  - Reasoning effort: `medium`
  - Tool version: `codex-cli 0.115.0-alpha.27`
- **Robustness check**:
  - The same LLM forecast contract is rerun `5` times on a representative OOS row.
  - The notebook records run-level outputs plus mean / std for liabilities, equity, and assets.
  - Artifact: `Question_1/output/codex_gpt54mini_medium_robustness.json`
- **Other Part 2 artifacts**:
  - `Question_1/output/codex_gpt54mini_medium_val_llm_bs_predictions.json`
  - `Question_1/output/codex_gpt54mini_medium_oos_llm_bs_predictions.json`
  - `Question_1/output/codex_gpt54mini_medium_val_weight_search.json`
  - `Question_1/output/codex_gpt54mini_medium_oos_comparison.json`

### Part 2 Prompt Alignment
- **"Choose your favourite LLM"**: satisfied with `gpt-5.4-mini`.
- **"Use the same set of data as collected in part 1"**: satisfied by reusing the Part 1 aligned dataset and the same train / validation / test split.
- **"Does the LLM perform better or worse than your model?"**: in the current run, the LLM performs better than Part 1 on both raw and normalized OOS metrics.
- **"Is it possible to create an ensemble model that performs better?"**: a simple `50/50` ensemble is worse; validation-based tuning with `LLM weight >= 50%` selects `1.00`, so no blended model beats the pure LLM baseline in the current experiment.
- **"Pick a company and recommend something to the CFO / CEO"**: satisfied by the memo generated in Section 9.
- **"If you are using LLM, is the output robust to different runs, and what version did you use?"**: satisfied by the robustness section and artifact above.

## Bonus 1
### What it does
- Builds an **annual-report-only internal issuer rating baseline**.
- Restricts the output to `A / B / C / D`.
- Uses only extracted annual-report values, canonical financial metrics, and quantitative shenanigans signals.
- Does **not** claim to be an official agency rating.

### Design
- Runtime logic lives in `Question_1/scripts/RatingModel.py`.
- Config lives in:
  - `Question_1/config/bonus1_rating_config.json`
  - `Question_1/config/bonus1_case_sources.json`
  - `Question_1/config/bonus1_shenanigans_config.json`
- Upstream annual-report payloads come from `Question_1/output/*_mini_metrics.json` and now include:
  - `extracted_values`
  - `canonical_values`
  - `metrics`

### Outputs
- `Question_1/output/bonus1_internal_rating_baseline.json`
- `Question_1/output/bonus1_evergrande_rating_case.json`
- `Question_1/output/bonus1_shenanigans_scan.json`
- `Question_1/output/bonus1_bankruptcy_oos_summary.json`

### Current coverage
- Internal baseline across the main annual-report company set.
- Evergrande case run under the same generic extraction / scoring pipeline.
- Bankruptcy OOS cohort including:
  - `evergrande_2022`
  - `thomas_cook_2018`
  - `debenhams_2018`
  - `intu_2019`

## Bonus 2
### What it does
- Extracts **warning passages** from annual reports.
- Starts from high-value sections such as:
  - auditor report
  - basis for opinion
  - going concern
  - internal control
  - debt / liquidity
  - impairment / valuation
- Uses Codex to classify candidate passages into a structured warning schema.

### Design
- Retrieval logic lives in `Question_1/scripts/AutoPDF.py`.
- LLM warning analysis lives in `Question_1/scripts/LLM.py`.
- Config lives in `Question_1/config/bonus2_warning_config.json`.
- The notebook keeps the final orchestration, cohort assembly, and result display.

### Outputs
- `Question_1/output/bonus2_bankruptcy_warning_summary.json`
- `Question_1/output/bonus2_evergrande_2022_warning_analysis.json`
- `Question_1/output/bonus2_thomas_cook_2018_warning_analysis.json`
- `Question_1/output/bonus2_debenhams_2018_warning_analysis.json`
- `Question_1/output/bonus2_intu_2019_warning_analysis.json`

### Current cohort result
- The same four bankruptcy / distressed cases are used as the OOS cohort.
- `Thomas Cook 2018` is intentionally left as an honest generic-pipeline result rather than being forced into a stronger warning outcome.

## Bonus 3
### What it does
- Builds a **public-data proxy** for term-loan pricing.
- Produces:
  - indicative spread in bps
  - indicative interest rate
  - 1-month resale-price proxy
  - `95%` interval
  - lending decision layer: `quote / manual_review / decline`

### Design
- Runtime logic lives in `Question_1/scripts/LoanPricing.py`.
- Config lives in `Question_1/config/bonus3_pricing_config.json`.
- Public market data artifacts live in:
  - `Question_1/data/public/bonus3_public_market_data_daily.csv`
  - `Question_1/data/public/bonus3_public_market_data_monthly.csv`

### Outputs
- `Question_1/output/bonus3_public_model_metrics.json`
- `Question_1/output/bonus3_public_pricing_recommendations.json`

### Limitation boundary
- This is a **public-data proxy MVP**, not a bank-grade commercial loan-pricing system.
- The repo does not include proprietary facility-level origination spread tapes or secondary loan transaction tapes.
- The notebook explicitly states those limitations in the Bonus 3 markdown section.

## Mathematical Details
We model the balance sheet as a constrained dynamical system with exogenous drivers:

$$
y_{t+1} = f_{\theta}(x_t, y_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma)
$$

where the state is

$$
y_t = \begin{bmatrix} L_t \\ E_t \end{bmatrix}, \quad A_t = L_t + E_t
$$

and the drivers include working-capital ratios, margins, growth, and seasonality:

$$
x_t = \{\mathrm{DSO}_t, \mathrm{DPO}_t, \mathrm{DIH}_t, \text{margins}, \text{growth}, \text{seasonality}, y_t\}
$$

Working-capital updates:

$$
AR_{t+1} = \mathrm{DSO}_t \cdot \frac{\mathrm{Revenue}_t}{365}, \quad
AP_{t+1} = \mathrm{DPO}_t \cdot \frac{\mathrm{COGS}_t}{365}, \quad
INV_{t+1} = \mathrm{DIH}_t \cdot \frac{\mathrm{COGS}_t}{365}
$$

Fixed assets and retained earnings:

$$
PPE_{t+1} = PPE_t + \mathrm{Capex}_t - \mathrm{Dep}_t
$$

$$
RE_{t+1} = RE_t + NI_t - DIV_t
$$

Equity and liabilities dynamics:

$$
E_{t+1} = RE_{t+1} + \mathrm{OtherEquity}_t
$$

$$
L_{t+1} = AP_{t+1} + \mathrm{OtherLiab}_t (1 + g_t)
$$

Accounting identity (hard constraint):

$$
A_{t+1} = L_{t+1} + E_{t+1}
$$

Earnings are modeled probabilistically with TFP:

$$
NI_t \sim \mathcal{N}(\mu_t, \sigma_t^2)
$$

## Main Entry Points
### Primary notebook
- `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb`
- This is the **main submission notebook**.
- It is the best starting point if the goal is to review the final work product.

### Data maintenance notebook
- `Question_1/notebooks/JPM_Q1_Data.ipynb`
- This is the **data refresh / smoke-test notebook**.
- Use it when raw Yahoo Finance / SEC data needs to be refreshed without touching the main analytical notebook.

### Main script modules
- `Question_1/scripts/Infra.py`
  - shared notebook infrastructure: plotting theme, artifact loading, manifest snapshots, Codex preflight guard
- `Question_1/scripts/AutoPDF.py`
  - annual-report extraction, warning-passage retrieval, manifest refresh, validation
- `Question_1/scripts/LLM.py`
  - Codex sign-in smoke test, LLM balance-sheet forecast pipeline, robustness checks, Bonus 2 warning analysis
- `Question_1/scripts/RatingModel.py`
  - Bonus 1 rating baseline and shenanigans model logic
- `Question_1/scripts/LoanPricing.py`
  - Bonus 3 public-data pricing MVP
- `Question_1/scripts/fetch_yf_statements.py`
  - raw Yahoo Finance fetch utility
- `Question_1/scripts/fetch_sec_xbrl.py`
  - raw SEC XBRL fetch utility
- `Question_1/scripts/clean_financials.py`
  - canonical cleaning layer for Part 1 inputs

## Recommended Workflow
### If the goal is only to review the finished work
1. Open `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb`
2. Run `Run All`
3. Leave all execution switches in `Question_1/config/pipeline_config.json` at their default `false`

That path loads the cached JSON artifacts under `Question_1/output/` and does **not** recompute the heavy sections.

### If the goal is to refresh only the data caches
1. Open `Question_1/notebooks/JPM_Q1_Data.ipynb`
2. Run the smoke-test path first
3. Only then decide whether a broader refresh is needed

### If the goal is to recompute one specific section
Edit `Question_1/config/pipeline_config.json`:
- `execution.part2.*`
- `execution.llm.*`
- `execution.bonus1.*`
- `execution.bonus2.*`
- `execution.bonus3.*`

Only the section whose flag is set to `true` will recompute and overwrite its output artifact.

## Project Layout
- `.python-version`
  - pinned interpreter version used for this deliverable
- `requirements.txt`
  - pip install list for the validated environment
- `environment.yml`
  - conda / mamba environment file for the same pinned baseline
- `Question_1/config/`
  - project configuration JSON files
- `Question_1/data/`
  - cached data inputs and downloaded annual reports
- `Question_1/output/`
  - persisted analytical artifacts used by the notebook
- `Question_1/notebooks/`
  - user-facing notebooks
- `Question_1/scripts/`
  - reusable implementation modules

## Config JSON Guide
### Core project config
- `Question_1/config/pipeline_config.json`
  - central runtime config
  - controls:
    - data paths
    - ticker universe and special mappings
    - SEC fetch fields
    - Part 1 training aliases / targets
    - visualization theme
    - execution switches for safe `Run All`

### Scope / delivery metadata
- `Question_1/config/delivery_scope.json`
  - frozen delivery scope for Part 2 and Bonus work
- `Question_1/config/scope_freeze_2026q1.md`
  - human-readable legacy planning note

### Part 2 config
- `Question_1/config/annual_report_sources.json`
  - manifest of annual-report sources and output targets
  - tells the pipeline which companies are in scope and where their artifacts live
- `Question_1/config/annual_report_schema.json`
  - validation contract for Part 2 extraction outputs
  - defines required keys / sections / metrics

### Bonus 1 config
- `Question_1/config/bonus1_rating_config.json`
  - rating rules, metric thresholds, distress overrides, rating bands
- `Question_1/config/bonus1_case_sources.json`
  - generic distressed-case registry and extraction recipes
- `Question_1/config/bonus1_shenanigans_config.json`
  - quantitative warning-signal rules and severity bands

### Bonus 2 config
- `Question_1/config/bonus2_warning_config.json`
  - retrieval section specs, risk taxonomy, severity levels, prompt contract settings

### Bonus 3 config
- `Question_1/config/bonus3_pricing_config.json`
  - public market series map, rating bucket mapping, borrower add-ons, lending decision policy

## Output Artifact Guide
### Part 2 extraction outputs
These are the main annual-report JSON payloads used by later sections.
- `Question_1/output/gm_2023_mini_metrics.json`
- `Question_1/output/lvmh_2024_mini_metrics.json`
- `Question_1/output/jpm_2024_mini_metrics.json`
- `Question_1/output/tencent_2024_mini_metrics.json`
- `Question_1/output/alibaba_2024_mini_metrics.json`
- `Question_1/output/exxon_2024_mini_metrics.json`
- `Question_1/output/vw_2024_mini_metrics.json`
- `Question_1/output/microsoft_2024_mini_metrics.json`
- `Question_1/output/alphabet_2024_mini_metrics.json`
- `Question_1/output/mini_cross_company_metrics.json`

Each payload includes:
- `extracted_values`
- `canonical_values`
- `metrics`
- statement page references

### Part 2 LLM artifacts
- `Question_1/output/codex_gpt54mini_medium_val_llm_bs_predictions.json`
- `Question_1/output/codex_gpt54mini_medium_oos_llm_bs_predictions.json`
- `Question_1/output/codex_gpt54mini_medium_val_weight_search.json`
- `Question_1/output/codex_gpt54mini_medium_oos_comparison.json`
- `Question_1/output/codex_gpt54mini_medium_robustness.json`

### Bonus 1 artifacts
- `Question_1/output/bonus1_internal_rating_baseline.json`
- `Question_1/output/bonus1_evergrande_rating_case.json`
- `Question_1/output/bonus1_shenanigans_scan.json`
- `Question_1/output/bonus1_bankruptcy_oos_summary.json`

### Bonus 2 artifacts
- `Question_1/output/bonus2_bankruptcy_warning_summary.json`
- `Question_1/output/bonus2_evergrande_2022_warning_analysis.json`
- `Question_1/output/bonus2_thomas_cook_2018_warning_analysis.json`
- `Question_1/output/bonus2_debenhams_2018_warning_analysis.json`
- `Question_1/output/bonus2_intu_2019_warning_analysis.json`

### Bonus 3 artifacts
- `Question_1/output/bonus3_public_model_metrics.json`
- `Question_1/output/bonus3_public_pricing_recommendations.json`
- `Question_1/data/public/bonus3_public_market_data_daily.csv`
- `Question_1/data/public/bonus3_public_market_data_monthly.csv`

## How the Main Notebook Now Behaves
The main notebook has been made **safe for direct `Run All`**.

### Default behavior
By default, all heavy execution flags are `false`.
That means:
- PDF extraction loads cached JSONs
- LLM baseline loads cached predictions and cached comparison outputs
- Bonus 1 loads cached rating / shenanigans / bankruptcy outputs
- Bonus 2 loads cached warning-analysis outputs
- Bonus 3 loads cached pricing outputs

### When recomputation happens
Recomputation only happens when the relevant switch is explicitly turned on in `pipeline_config.json`.

### Codex preflight
The notebook now checks Codex availability before any Codex-backed execution.
If Codex is missing or not logged in:
- default `Run All` still succeeds, because cached artifacts are used
- real Codex execution will fail with a clear preflight error instead of a raw `no codex executable found`

## How to Run
### Pinned environment
`venv + pip`
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`conda / mamba`
```bash
conda env create -f environment.yml
conda activate jpm-q1-py313
```

### Main notebook
```bash
jupyter execute --inplace Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb
```

### Data notebook
```bash
jupyter execute --inplace Question_1/notebooks/JPM_Q1_Data.ipynb
```

### Optional raw data refresh utilities
```bash
python Question_1/scripts/fetch_yf_statements.py --tickers AAPL GOOG
python Question_1/scripts/fetch_sec_xbrl.py --tickers AAPL
python Question_1/scripts/clean_financials.py
```

## Environment and Reproducibility
- **Pinned baseline**: `Python 3.13.1`
- **Validated runtime**: Windows 11 plus the dependency set in `requirements.txt` / `environment.yml`
- **Guarantee level**: pinned-environment reproducibility, not universal OS-agnostic reproducibility
- **TensorFlow Probability note**: the notebook uses a small `distutils` compatibility shim through `Question_1/scripts/Infra.py` before importing `tensorflow_probability`
- **External prerequisite**: Codex-backed sections still require a working Codex executable plus ChatGPT / Codex login

## Question Coverage Summary
### Part 1
- TensorFlow / TFP balance-sheet plus earnings model
- clean data input layer
- OOS evaluation and plots
- accounting identity enforcement

### Part 2
- annual-report extraction for the full required company set
- LLM baseline using the same Part 1 data split
- comparison vs Part 1
- ensemble test and weight search
- robustness run summary
- CFO / CEO recommendation

### Bonus 1
- annual-report-only internal `A / B / C / D` rating baseline
- quantitative shenanigans detector
- Evergrande case
- bankruptcy OOS cohort

### Bonus 2
- warning paragraph retrieval from key sections
- Codex-based semantic classification
- bankrupt-company cohort evaluation

### Bonus 3
- public-data proxy loan-pricing MVP
- decision layer: `quote / manual_review / decline`
- 1-month resale-price proxy with `95% CI`

## Current Architecture Principle
The notebook is intentionally used for:
- orchestration
- evaluation
- comparison
- plots
- writeup / narrative answers

The scripts are used for:
- data and PDF infrastructure
- reusable model logic
- artifact IO and cached-result loading

That split is deliberate. It keeps the notebook readable while leaving the core implementation reusable and testable.
