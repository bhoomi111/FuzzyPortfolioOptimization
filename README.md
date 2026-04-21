# Fuzzy Portfolio Optimization

A Python implementation of a fuzzy portfolio optimization workflow using coherent triangular fuzzy numbers and a multi-objective genetic algorithm (MOGA) foundation.

## What This Repo Does

- Loads monthly returns or prices from CSV files
- Computes portfolio return series from asset returns and weights
- Fits a coherent triangular fuzzy number to historical portfolio returns
- Computes fuzzy moments used for portfolio modeling:
  - Credibilistic mean
  - Semivariance
  - Skewness
  - Semikurtosis
- Provides optimization building blocks (constraints, operators, Pareto sorting, crowding distance, MOGA)

## Project Layout

- `data/raw/`: input datasets (returns, prices, metadata)
- `data/processed/`: processed outputs
- `scripts/run_experiment.py`: baseline experiment entrypoint
- `src/data/`: data loading and preprocessing
- `src/fuzzy/`: coherent triangular fuzzy fitting
- `src/moments/`: risk/shape moment functions
- `src/models/`: objective composition
- `src/optimization/`: MOGA and related operators
- `src/clustering/`: k-medoids helper

## Requirements

- Python 3.11 (recommended for this workspace)
- pip

Install dependencies:

```bash
py -3.11 -m pip install -r requirements.txt
```

## Quick Start

From repository root:

```bash
py -3.11 scripts/run_experiment.py
```

Expected output format:

```text
Mean: <value>
Fuzzy params: <b1> <b2> <b3> <k>
```

## Data Expectations

The baseline script reads:

- `data/raw/monthly_returns.csv`

Expected shape:

- A `Date` column
- One column per asset with numeric returns

## Notes

- The repository currently includes core optimization primitives and a baseline experiment.
- `src/moments/cvar.py` and `src/moments/masd.py` are present as placeholders and can be implemented as future extensions.
