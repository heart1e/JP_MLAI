# Scope Freeze - JPM MLCOE TSRL 2026 Q1 (Part 2 + Bonus)

This document freezes execution scope for Part 2 and Bonus sections.

## 1. Objective
- Deliver a complete, reproducible Part 2 + Bonus submission.
- Prioritize end-to-end working pipelines over broad but shallow coverage.

## 2. Frozen Scope

### 2.1 Part 2 (Core)
- Build annual report extraction for financial statements and ratio computation.
- Run LLM-based financial statement analysis and compare against Part 1 model.
- Build simple ensemble (Part 1 model + LLM model) and compare performance.
- Produce one management recommendation memo (CFO/CEO) based on model outputs.
- Report LLM robustness across repeated runs and pin API model version.

### 2.2 Bonus B (Credit Rating)
- Define mathematical model for annual-report-to-credit-rating mapping.
- Build trainable prototype with sourced label data.
- Run Evergrande case study.
- Add financial-shenanigans warning checks and validate on bankrupt examples.

### 2.3 Bonus C (Risk Warnings)
- Build automatic extraction engine for risk-warning paragraphs.
- Include qualified-opinion style cues and additional warning classes.
- Validate on bankrupt-company annual reports.

### 2.4 Bonus D (Loan Pricing)
- Build term-loan spread model and brief literature survey.
- Handle private borrower case (no tradable debt/equity) with fallback model.
- Forecast 1-month resale loan price.
- Output 95% confidence interval for 1-month price forecast.

## 3. Company Universe (Frozen)

### 3.1 PDF Extraction and Generalization Set
- GM (required by prompt)
- LVMH (required by prompt)
- Tencent
- Alibaba
- JPMorgan Chase
- Exxon Mobil
- Volkswagen
- Microsoft
- Alphabet (Google)

### 3.2 Distress / Bankruptcy Set
- Evergrande (required by prompt)
- At least 2 additional bankrupt historical companies

## 4. Deliverables and Acceptance Criteria

### D1. Extraction Engine
- Input: annual report PDF.
- Output: normalized JSON with statement values + ratios.
- Must run on GM and LVMH plus at least 3 additional companies from 3.1.

### D2. LLM vs Part1 vs Ensemble
- Use same underlying financial-statement dataset as Part 1 for forecast comparison.
- Report common metrics (MAE at minimum; add scale-free metric where possible).
- Ensemble must show either metric improvement or clear failure analysis.

### D3. Robustness and Versioning
- Pin and report exact API model version.
- Run repeated inference (minimum 5 runs per prompt template).
- Report mean/std for extracted key outputs.

### D4. Bonus B
- Training data source is documented and data snapshot stored in repository.
- Evergrande scoring output generated.
- Shenanigans warning module tested on bankrupt-company sample.

### D5. Bonus C
- Risk paragraph extraction output includes source page/section references.
- Bankrupt-company test report includes hit/miss summary.

### D6. Bonus D
- Spread model includes feature definition and calibration approach.
- Private-client fallback logic implemented.
- 1-month resale price forecast and 95% CI produced.

## 5. Out of Scope (For This Submission)
- Full production deployment.
- Real-time market data ingestion infra.
- Full legal/compliance production controls.

## 6. Folder and Artifact Contract
- Raw reports: `Question_1/data/raw/`
- Parsed outputs: `Question_1/output/`
- Config and manifests: `Question_1/config/`
- Notebook integration: `Question_1/notebooks/JPM_MLCOE_TSRL_2026_Q1.ipynb`

## 7. Milestone Sequence (Execution Order)
1. Scope Freeze (this document)
2. Unified extraction + schema validation
3. Part 2 model comparison + ensemble + recommendation
4. Bonus B implementation
5. Bonus C implementation
6. Bonus D implementation
7. Reproducibility pass and final packaging
