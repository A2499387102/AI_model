# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a credit risk scoring and binary classification modeling framework used in Chinese fintech/banking contexts. Work is done in Jupyter notebooks using Python 3.9+. There are no build steps, test suites, or linting pipelines — the primary workflow is iterative notebook execution.

## Running the Environment

```bash
# Launch Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Key dependencies: `lightgbm`, `xgboost`, `hyperopt`, `toad`, `statsmodels`, `scikit-learn`, `pandas`, `numpy`, `joblib`.

## Notebook Architecture

The two main notebooks follow the same pipeline structure:

1. **Data ingestion** — merge label file + feature tables + wide tables by key; split into train / test / OOT (out-of-time) sets
2. **Feature screening** — sequential filters: missing rate (>0.99 drop), IV (<0.01 drop), correlation (>0.8 drop), PSI (>0.15 or >0.5 depending on threshold), stability trend across channels/periods
3. **Model training** — Bayesian hyperparameter search via Hyperopt (TPE sampler); XGBoost uses `gpu_hist` + `scale_pos_weight`; LightGBM uses `is_unbalance=True`; both use early stopping on AUC
4. **Feature refinement** — importance-based filtering after a first-pass model, then retrain on reduced feature set
5. **Evaluation** — KS and AUC across train/test/OOT, plus stratified by channel and time period; decile analysis
6. **Score standardization** — GLM rescaling via statsmodels to PDO-based score range (typically 300–900)
7. **Ensemble** — stepwise regression combining multiple model scores, feature selection by AIC
8. **Export** — pickle / joblib / native model format / PMML for production deployment

## Domain Conventions

- **KS** (Kolmogorov-Smirnov) = max(TPR − FPR) — the primary model quality metric in this domain
- **IV** (Information Value) and **WOE** (Weight of Evidence) — standard feature quality metrics for credit scoring
- **PSI** (Population Stability Index) — measures distributional shift between train and OOT; threshold 0.15 is a common cutoff
- **OOT** (Out-of-Time) — held-out future data used to test temporal generalization, treated as the most important validation set
- **PDO** (Points to Double Odds) — score scaling convention; score output should map to a business-friendly integer range

## Key Patterns

- Sample weights are applied when handling class imbalance in XGBoost; LightGBM uses `is_unbalance` instead
- SHAP values are computed for model explainability and regulatory documentation
- Multi-segment evaluation (by channel, by month) is standard — never report only aggregate metrics
- Hyperopt `fmin` with `tpe.suggest` is the tuning method; loss function is `1 - AUC` on validation set
- Feature importance is extracted three ways in XGBoost: `gain`, `cover`, `weight` — `gain` is the primary signal
