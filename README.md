# Fuzzy Portfolio Optimization

Python implementation of a fuzzy portfolio optimization workflow using coherent triangular fuzzy numbers and a multi-objective genetic algorithm (MOGA).

## Features

- Loads monthly returns data from CSV
- Builds coherent triangular fuzzy numbers from portfolio returns
- Supports a rolling quantile fuzzy fit for data-driven triangular fuzzy numbers
- Evaluates multiple fuzzy-moment objectives:
  - Mean
  - Semivariance
  - MASD
  - CVaR
  - Skewness
  - Semikurtosis
- Runs three model variants and generates comparison tables
- Saves portfolio tables and plot images into the results folder
- Compares model performance against NIFTY 50 on the test period

## Project Structure

- `data/raw`: input data files
- `results`: generated CSV tables and PNG plots when running scripts individually
- `results_full_replication`: generated outputs from the full pipeline
- `scripts/run_tables.py`: full table and portfolio generation for all models
- `scripts/run_benchmark.py`: benchmark comparison against NIFTY 50
- `scripts/run_full_replication.py`: full pipeline for models, plots, and benchmark outputs
- `src/data`: loaders and preprocessing
- `src/fuzzy`: coherent triangular fuzzy fitting
- `src/moments`: fuzzy moment functions
- `src/models`: objective definitions
- `src/optimization`: MOGA operators and Pareto utilities
- `src/clustering`: k-medoids representative selection
- `src/visualization/plot.py`: plot generation from saved results

## Requirements

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Guide

Run from repository root.

### Generate model result tables and portfolios

```bash
python scripts/run_tables.py
```

This writes files like:

- `results/model1_table.csv`
- `results/model2_table.csv`
- `results/model3_table.csv`
- `results/model1_portfolios.csv`
- `results/model2_portfolios.csv`
- `results/model3_portfolios.csv`
- `results/model1_global_pareto_table.csv`
- `results/model1_filtered_pareto_table.csv`

### Generate plots from saved tables

```bash
python src/visualization/plot.py
```

This writes files like:

- `results/model1_pareto_front.png` (global, filtered, and representative solutions overlaid)
- `results/model1_risk_vs_skewness.png` (global, filtered, and representative solutions overlaid)
- `results/model1_efficient_frontier.png` (global, filtered, and representative frontiers overlaid)
- `results/model1_portfolio_allocations.png`
- `results/model2_pareto_front.png`
- `results/model2_risk_vs_skewness.png`
- `results/model2_efficient_frontier.png`
- `results/model2_portfolio_allocations.png`
- `results/model3_pareto_front.png`
- `results/model3_risk_vs_skewness.png`
- `results/model3_efficient_frontier.png`
- `results/model3_portfolio_allocations.png`

### Generate benchmark comparison outputs

```bash
python scripts/run_benchmark.py
```

This runs the benchmark comparison against NIFTY 50 and writes files like:

- `results/model1_average_weights.csv`
- `results/model2_average_weights.csv`
- `results/model3_average_weights.csv`
- `results/benchmark_monthly_returns.csv`
- `results/benchmark_quarterly_returns.csv`
- `results/benchmark_average_allocations.csv`
- `results/benchmark_cumulative_monthly_returns.png`
- `results/benchmark_cumulative_quarterly_returns.png`

### Run the full pipeline

```bash
python scripts/run_full_replication.py
```

This runs the three models once, then uses the same results to generate:

- model tables
- portfolio CSVs
- model plots
- benchmark comparison outputs against NIFTY 50

This writes files into:

- `results_full_replication/tables_and_model_plots`
- `results_full_replication/benchmark_vs_nifty50`

### Optimizer and fuzzy mode via CLI

You can now choose the optimizer and fuzzy fit mode from command line runners.

Tables run:

```bash
python scripts/run_tables.py --optimizer nsga2 --fuzzy-fit-method static
```

Benchmark run:

```bash
python scripts/run_benchmark.py --optimizer nsga2 --fuzzy-fit-method rolling --rolling-window 30
```

Supported options:

- `--optimizer moga|nsga2`
- `--fuzzy-fit-method static|rolling`
- `--rolling-window <int>`
- `--fuzzy-quantiles q1,q2,q3`
- `--fuzzy-min-periods <int>`
- `--crossover-mode sbx|convex`

### Compare baseline and rolling-quantile fuzzy fits

The full replication script can now run both the original static fuzzy fit and the new rolling quantile fit, then save comparison plots without NIFTY 50.

```bash
python scripts/run_full_replication.py --rolling-window 30
```

This writes separate outputs for:

- `results_full_replication/baseline_static`
- `results_full_replication/modified_rolling`
- `results_full_replication/comparison_plots`

The comparison plots include cumulative return lines for Model I, Model II, and Model III before and after the fuzzy-number change, without the benchmark series.

### Compare baseline MOGA vs NSGA-II (without rolling)

Use this to keep fuzzy mode static in both runs and compare only optimizer changes.

```bash
python scripts/run_full_replication.py \
  --baseline-name baseline_static_moga \
  --baseline-optimizer moga \
  --baseline-fuzzy-fit-method static \
  --modified-name modified_nsga2_static \
  --modified-optimizer nsga2 \
  --modified-fuzzy-fit-method static \
  --comparison-dir comparison_plots_nsga2
```

This writes separate outputs for:

- `results_full_replication/baseline_static_moga`
- `results_full_replication/modified_nsga2_static`
- `results_full_replication/comparison_plots_nsga2`

You can also combine NSGA-II with rolling mode by setting `--modified-fuzzy-fit-method rolling --rolling-window 30`.

## Data Format

Main input file:

- `data/raw/monthly_returns.csv`

Benchmark input file:

- `data/raw/nifty50_monthly_returns.csv`

Expected columns:

- `monthly_returns.csv`: `Date` and one numeric return column per asset
- `nifty50_monthly_returns.csv`: `Date`, `NIFTY50`
