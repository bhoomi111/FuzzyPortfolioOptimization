from pathlib import Path
import argparse
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    CP_VALUES,
    FUZZY_FIT_METHOD,
    FUZZY_QUANTILES,
    FUZZY_ROLLING_WINDOW,
    MOMENTUM_LOOKBACK,
    MOMENTUM_WEIGHT,
    MP_VALUES,
    MONTHLY_RETURNS_FILE,
    POP_SIZE,
    RESULTS_DIR,
    RUNS,
    GENERATIONS,
    TEST_MONTHS,
)
from src.evaluation.benchmark import load_monthly_returns, split_last_n_months
from src.models.model1 import model1_objectives
from src.models.model2 import model2_objectives
from src.models.model3 import model3_objectives

from scripts.build_table import build_table
from scripts.run_model import run_model

MODEL_SPECS = [
    ("MODEL I RESULTS", "model1", model1_objectives),
    ("MODEL II RESULTS", "model2", model2_objectives),
    ("MODEL III RESULTS", "model3", model3_objectives),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model tables and representative portfolios.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--optimizer", type=str, default="moga", choices=["moga", "nsga2"])
    parser.add_argument("--fuzzy-fit-method", type=str, default=FUZZY_FIT_METHOD, choices=["static", "rolling"])
    parser.add_argument("--rolling-window", type=int, default=FUZZY_ROLLING_WINDOW)
    parser.add_argument("--fuzzy-quantiles", type=str, default=",".join(map(str, FUZZY_QUANTILES)))
    parser.add_argument("--fuzzy-min-periods", type=int, default=None)
    parser.add_argument("--crossover-mode", type=str, default="sbx", choices=["sbx", "convex"])
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("No representative solutions passed the filtration criteria.")
        return

    display_df = df.copy()
    metric_cols = [c for c in ["b1", "b2", "b3", "k", "Mean", "Risk", "Skewness", "Semikurtosis"] if c in display_df.columns]
    for col in metric_cols:
        display_df[col] = display_df[col].astype(float).map(lambda x: f"{x:.5e}")

    print(display_df.to_string(index=False))


def save_portfolios(results: list[dict], filename: Path, columns, label_prefix: str = "R") -> None:
    rows = []

    for i, result in enumerate(results):
        row = {"Rep": f"{label_prefix}{i+1}"}
        for j, col in enumerate(columns):
            row[col] = result["weights"][j]
        rows.append(row)

    portfolio_columns = ["Rep", *list(columns)]
    pd.DataFrame(rows, columns=portfolio_columns).to_csv(filename, index=False)


def load_dataset_split(test_months: int = TEST_MONTHS):
    returns = load_monthly_returns(MONTHLY_RETURNS_FILE)
    train_returns, test_returns, split_note = split_last_n_months(returns, test_months=test_months)
    return returns, train_returns, test_returns, split_note


def run_model_suite(
    train_returns,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    runs: int = RUNS,
    verbose: bool = True,
    momentum_weight: float = MOMENTUM_WEIGHT,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    fuzzy_fit_method: str = FUZZY_FIT_METHOD,
    fuzzy_window: int = FUZZY_ROLLING_WINDOW,
    fuzzy_quantiles: tuple[float, float, float] = FUZZY_QUANTILES,
    fuzzy_min_periods: int | None = None,
    optimizer: str = "moga",
    crossover_mode: str = "sbx",
):
    n_assets = train_returns.shape[1]
    outputs = {}

    for title, slug, objective_fn in MODEL_SPECS:
        results = run_model(
            train_returns,
            objective_fn,
            CP_VALUES,
            MP_VALUES,
            n_assets,
            pop_size=pop_size,
            generations=generations,
            runs=runs,
            verbose=verbose,
            detailed_logging=False,
            model_label=title.replace(" RESULTS", ""),
            momentum_weight=momentum_weight,
            momentum_lookback=momentum_lookback,
            fuzzy_fit_method=fuzzy_fit_method,
            fuzzy_window=fuzzy_window,
            fuzzy_quantiles=fuzzy_quantiles,
            fuzzy_min_periods=fuzzy_min_periods,
            optimizer=optimizer,
            crossover_mode=crossover_mode,
        )
        outputs[slug] = {"title": title, **results}

    return outputs


def run_tables_pipeline(
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    runs: int = RUNS,
    results_dir: Path = RESULTS_DIR,
    model_outputs: dict | None = None,
    show_tables: bool = False,
    emit_setup_logs: bool = True,
    fuzzy_fit_method: str = FUZZY_FIT_METHOD,
    fuzzy_window: int = FUZZY_ROLLING_WINDOW,
    fuzzy_quantiles: tuple[float, float, float] = FUZZY_QUANTILES,
    fuzzy_min_periods: int | None = None,
    optimizer: str = "moga",
    crossover_mode: str = "sbx",
):
    results_dir.mkdir(parents=True, exist_ok=True)

    returns, train_returns, test_returns, split_note = load_dataset_split(test_months=TEST_MONTHS)

    if emit_setup_logs:
        log(
            f"Loaded dataset with {len(returns)} monthly rows, {returns.shape[1]} assets, "
            f"date range {returns.index.min().date()} to {returns.index.max().date()}."
        )
        log(split_note)
        log(f"Using config settings | population={pop_size} | generations={generations} | runs={runs}")

    outputs = model_outputs if model_outputs is not None else run_model_suite(
        train_returns,
        pop_size=pop_size,
        generations=generations,
        runs=runs,
        verbose=True,
        fuzzy_fit_method=fuzzy_fit_method,
        fuzzy_window=fuzzy_window,
        fuzzy_quantiles=fuzzy_quantiles,
        fuzzy_min_periods=fuzzy_min_periods,
        optimizer=optimizer,
        crossover_mode=crossover_mode,
    )

    for title, slug, _objective_fn in MODEL_SPECS:
        representatives = outputs[slug]["representatives"]
        global_pareto = outputs[slug]["global_pareto"]
        filtered_pareto = outputs[slug]["filtered_pareto"]

        table = build_table(representatives, slug)
        table_path = results_dir / f"{slug}_table.csv"
        portfolios_path = results_dir / f"{slug}_portfolios.csv"
        table.to_csv(table_path, index=False)
        save_portfolios(representatives, portfolios_path, returns.columns, label_prefix="R")

        global_table = build_table(global_pareto, slug)
        global_table_path = results_dir / f"{slug}_global_pareto_table.csv"
        global_portfolios_path = results_dir / f"{slug}_global_pareto_portfolios.csv"
        global_table.to_csv(global_table_path, index=False)
        save_portfolios(global_pareto, global_portfolios_path, returns.columns, label_prefix="G")

        filtered_table = build_table(filtered_pareto, slug)
        filtered_table_path = results_dir / f"{slug}_filtered_pareto_table.csv"
        filtered_portfolios_path = results_dir / f"{slug}_filtered_pareto_portfolios.csv"
        filtered_table.to_csv(filtered_table_path, index=False)
        save_portfolios(filtered_pareto, filtered_portfolios_path, returns.columns, label_prefix="F")

        outputs[slug].update(
            {
                "table": table,
                "table_path": table_path,
                "portfolios_path": portfolios_path,
                "global_table": global_table,
                "global_table_path": global_table_path,
                "global_portfolios_path": global_portfolios_path,
                "filtered_table": filtered_table,
                "filtered_table_path": filtered_table_path,
                "filtered_portfolios_path": filtered_portfolios_path,
            }
        )
        if show_tables:
            print_table(title, table)
        log(
            f"{title.replace(' RESULTS', '')}: saved "
            f"{len(global_table)} global Pareto, {len(filtered_table)} filtered Pareto, "
            f"{len(table)} representatives."
        )

    log(f"Tables saved in {results_dir}")
    if emit_setup_logs:
        log(f"Training rows: {len(train_returns)} | Unseen rows reserved: {len(test_returns)}")
    return outputs


def main() -> None:
    args = parse_args()
    quantiles = tuple(float(x) for x in args.fuzzy_quantiles.split(","))
    run_tables_pipeline(
        results_dir=args.results_dir,
        show_tables=False,
        optimizer=args.optimizer,
        fuzzy_fit_method=args.fuzzy_fit_method,
        fuzzy_window=args.rolling_window,
        fuzzy_quantiles=quantiles,
        fuzzy_min_periods=args.fuzzy_min_periods,
        crossover_mode=args.crossover_mode,
    )


if __name__ == "__main__":
    main()
