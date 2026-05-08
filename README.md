# Fuzzy Portfolio Optimization

Comprehensive implementation of a fuzzy portfolio optimization workflow using coherent triangular fuzzy numbers, multi-objective optimization, and optional deep-learning forecasts.

This README documents the repository layout, how to reproduce the baseline full-pipeline results, and how to run the two supported modifications:
- Modification 1: LSTM-based forecasts for NSE (daily → monthly)
- Modification 2: Rolling-window quantile fuzzy fit (rolling fuzzy fit)

Note: NSGA optimizer modification is intentionally omitted from these instructions.

**Repository layout**
- **data/raw/**: Raw input CSV files (monthly and daily returns, metadata). Primary files used by the pipeline live here.
- **results/**: Outputs produced by single-run scripts (tables and PNGs).
- **results_full_replication/**: Outputs produced by the full replication pipeline (baseline and modified runs, organized subfolders).
- **scripts/**: High-level runners for common tasks:
  - `run_tables.py` — build tables and portfolios for the three models.
  - `run_benchmark.py` — generate benchmark comparisons vs NIFTY50.
  - `run_full_replication.py` — orchestrates the full pipeline and writes into `results_full_replication/`.
  - `train_lstm_daily_nse.py` — helper script to train the NSE daily-to-monthly LSTM (used by Modification 1).
- **src/**: Core code:
  - `src/data/` — data loaders and preprocessing utilities.
  - `src/fuzzy/` — triangular fuzzy-number fittings (static and rolling).
  - `src/moments/` — fuzzy-moment objective implementations (mean, semivariance, MASD, CVaR, skewness, semikurtosis).
  - `src/models/` — model definitions and objective wrappers.
  - `src/optimization/` — optimization operators and Pareto utilities.
  - `src/clustering/` — k-medoids representative portfolio selection.
  - `src/visualization/` — plotting helpers used by the scripts.

Requirements
-----------
Install Python dependencies from the provided `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Reproduce the baseline (full pipeline)
------------------------------------
This will run the three models, generate tables, portfolios, plots and benchmark outputs and save them under `results_full_replication/`.

From the repository root:

```bash
python scripts/run_full_replication.py
```

Outputs (examples)
- `results_full_replication/tables_and_model_plots/` — CSVs and model plots for Model I/II/III (tables, pareto front images, allocation images).
- `results_full_replication/benchmark_vs_nifty50/` — benchmark CSVs and benchmark cumulative return PNGs comparing each model to NIFTY50.

Important options
- `--rolling-window <int>` — when provided, the pipeline runs the rolling-window fuzzy fit (see Modification 2).
- `--baseline-name`, `--modified-name` — choose output subfolder names for multiple-run comparisons.

Using Modification 1 — LSTM daily→monthly NSE forecasts
----------------------------------------------------
Purpose: use richer daily history for NSE tickers to produce monthly one-step-ahead forecasts that feed into the optimization pipeline.

Steps:
1. Train the NSE daily LSTM and write predictions (the repository includes `scripts/train_lstm_daily_nse.py`):

```bash
python scripts/train_lstm_daily_nse.py --daily_csv data/raw/daily_returns.csv
```

This script trains a daily LSTM (windowed daily inputs → monthly target) and writes a prediction CSV the pipeline or notebook can consume (see the script `--help` for the exact output path). It was created to be robust to pandas versions and to produce the monthly-aligned forecasts used by the notebook.

2. Re-run the notebook (optional) or the pipeline:
- If you prefer notebooks: open `Modification_1_LSTM.ipynb` and re-run Section 5 (NSE LSTM section) then continue Sections 9+ to produce downstream surrogates and portfolios.
- If you use the scripted pipeline, ensure the file produced by `train_lstm_daily_nse.py` is placed where the pipeline expects (by default the notebook/script looks for processed NSE monthly predictions in `data/processed/` or uses the notebook's in-memory path). If necessary, inspect `scripts/train_lstm_daily_nse.py` for the target filename and copy it into `data/processed/`.

Notes:
- The notebook `Credibilistic_DL_M1_M3_new.ipynb` contains a rewritten Section 5 that can consume the LSTM outputs. After running the training script, re-run that Section 5 cell to regenerate `nse_pred_df`, then continue the notebook from Section 9 onward.

Using Modification 2 — Rolling-window fuzzy fit
----------------------------------------------
Purpose: replace the static fuzzy fit with a rolling quantile fuzzy fit (data-driven triangular fuzzy numbers calculated on a rolling window).

Steps:
1. Run the full replication pipeline with a `--rolling-window` argument (example uses 30-period window):

```bash
python scripts/run_full_replication.py --rolling-window 30
```

This will create a parallel set of outputs for the rolling fit under `results_full_replication/modified_rolling/` (and comparison plots under `comparison_plots` if run together with baseline).

Notes:
- The rolling fit implementation lives in `src/fuzzy/rolling_quantiles.py`. The pipeline will call it when a non-zero `--rolling-window` is supplied.

Reproducing both modifications together
-------------------------------------
To evaluate the effect of both the LSTM NSE forecasts and the rolling fuzzy fit together, first produce the LSTM predictions (Modification 1), then run the full replication with the `--rolling-window` flag. Example:

```bash
python scripts/train_lstm_daily_nse.py --daily_csv data/raw/daily_returns.csv
python scripts/run_full_replication.py --rolling-window 30
```

Results, plots and what they mean
--------------------------------
Key output locations (full replication):
- `results_full_replication/tables_and_model_plots/` — tables and per-model plots.
  - `modelX_table.csv` — summary table for Model X (risk/return statistics, fuzzy moments, etc.).
  - `modelX_portfolios.csv` — portfolio allocations for representative solutions.
  - `modelX_global_pareto_table.csv` / `modelX_filtered_pareto_table.csv` — global and filtered Pareto front numeric exports.
- `results_full_replication/benchmark_vs_nifty50/` — benchmark comparisons vs NIFTY 50
  - `benchmark_monthly_returns.csv` — monthly returns series for models and benchmark.
  - `benchmark_quarterly_returns.csv` — aggregated quarterly returns.
  - `benchmark_average_allocations.csv` — average allocations used for backtests.
  - `benchmark_cumulative_monthly_returns.png` — cumulative return plot; X-axis: date, Y-axis: cumulative return (1+R). Lines: each model and NIFTY50 benchmark.

Representative plots and labels
- Pareto front plots (`modelX_pareto_front.png`):
  - X-axis: one risk measure (e.g., semivariance or CVaR), Y-axis: another objective (e.g., negative mean or skewness) depending on model variant. Points: Pareto solutions; filtered/representative solutions are highlighted.
- Risk vs Skewness (`modelX_risk_vs_skewness.png`):
  - X-axis: risk metric (semivariance or MASD), Y-axis: skewness metric. Shows trade-offs across solutions.
- Efficient frontier (`modelX_efficient_frontier.png`):
  - X-axis: portfolio risk, Y-axis: expected return (or fuzzy equivalent). Shows efficient frontier and selected portfolios.
- Portfolio allocations (`modelX_portfolio_allocations.png`):
  - Bar or stacked plots showing weight breakdowns for representative solutions.

Where to inspect results for comparisons
- `results_full_replication/comparison_plots/` — side-by-side comparison images between baseline and modified runs (rolling fit and/or LSTM modifications when created).
- `results_full_replication/modified_rolling/` — outputs from the rolling-window modification run.
- `results_full_replication/baseline_static/` — baseline (static fuzzy fit) outputs.

Recommended reproduction checklist
--------------------------------
1. Create a clean Python environment and install `requirements.txt`.
2. Verify `data/raw/` contains `monthly_returns.csv` and `nifty50_monthly_returns.csv`.
3. (Optional) For Modification 1, run the LSTM trainer:

```bash
python scripts/train_lstm_daily_nse.py --daily_csv data/raw/daily_returns.csv
```

4. Run the full replication baseline:

```bash
python scripts/run_full_replication.py
```

5. To run the rolling fit modification:

```bash
python scripts/run_full_replication.py --rolling-window 30
```

6. Compare the generated folders in `results_full_replication/` and open the plots described above.

