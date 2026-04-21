# Fuzzy Portfolio Optimization

Python implementation of a fuzzy portfolio optimization workflow using coherent triangular fuzzy numbers and a multi-objective genetic algorithm (MOGA).

## Features

- Loads monthly returns data from CSV
- Builds coherent triangular fuzzy numbers from portfolio returns
- Evaluates multiple fuzzy-moment objectives:
  - Mean
  - Semivariance
  - MASD
  - CVaR-style downside proxy
  - Skewness
  - Semikurtosis
- Runs three model variants and generates comparison tables
- Saves portfolio tables and plot images into the results folder

## Project Structure

- data/raw: input data files
- data/processed: processed outputs
- results: generated CSV tables and PNG plots
- scripts/run_experiment.py: quick baseline experiment
- scripts/run_tables.py: full table + portfolio generation for all models
- scripts/run_comparison.py: comparison run across model variants
- src/data: loaders and preprocessing
- src/fuzzy: coherent triangular fuzzy fitting
- src/moments: fuzzy moment functions
- src/models: objective definitions
- src/optimization: MOGA operators and Pareto utilities
- src/clustering: k-medoids representative selection
- src/visualization/plot.py: plot generation from saved results

## Requirements


Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Guide

Run from repository root.

1) Quick sanity run

```bash
python scripts/run_experiment.py
```

2) Generate model result tables and portfolios

```bash
python scripts/run_tables.py
```

This writes files like:

- results/model1_table.csv
- results/model2_table.csv
- results/model3_table.csv
- results/model1_portfolios.csv
- results/model2_portfolios.csv
- results/model3_portfolios.csv

3) Generate plots from saved tables

```bash
python src/visualization/plot.py
```

This writes files like:

- results/model1_pareto_front.png
- results/model1_risk_vs_skewness.png
- results/model1_efficient_frontier.png
- results/model2_pareto_front.png
- results/model2_risk_vs_skewness.png
- results/model2_efficient_frontier.png
- results/model3_pareto_front.png
- results/model3_risk_vs_skewness.png
- results/model3_efficient_frontier.png

## Data Format

Main input file:

- data/raw/monthly_returns.csv

Expected columns:

- Date
- one numeric return column per asset
