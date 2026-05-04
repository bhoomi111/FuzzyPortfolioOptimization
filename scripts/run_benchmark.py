from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_tables import MODEL_SPECS, run_model_suite
from src.config import (
    BENCHMARK_RETURNS_PATH,
    BENCHMARK_COLUMN,
    MONTHLY_RETURNS_FILE,
    POP_SIZE,
    RESULTS_DIR,
    RUNS,
    GENERATIONS,
    TEST_MONTHS,
)
from src.evaluation.benchmark import (
    average_portfolio_returns_from_results,
    average_weights,
    backtest_models,
    load_benchmark_returns,
    load_monthly_returns,
    representative_weights_frame,
    split_last_n_months,
)
from src.visualization.plot import (
    plot_cumulative_returns_from_table,
    plot_quarterly_returns_from_table,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style benchmark comparison and plots.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--benchmark-returns", type=Path, default=BENCHMARK_RETURNS_PATH)
    parser.add_argument("--benchmark-prices", type=Path, default=None)
    parser.add_argument("--benchmark-column", type=str, default=BENCHMARK_COLUMN)
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_benchmark_pipeline(
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    runs: int = RUNS,
    results_dir: Path = RESULTS_DIR,
    benchmark_returns: Path | None = None,
    benchmark_prices: Path | None = None,
    benchmark_column: str | None = None,
    allow_proxy: bool = True,
    representative_results: dict[str, list[dict]] | None = None,
    emit_setup_logs: bool = True,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    returns = load_monthly_returns(MONTHLY_RETURNS_FILE)
    train_returns, test_returns, split_note = split_last_n_months(returns, test_months=TEST_MONTHS)
    n_assets = returns.shape[1]
    asset_names = returns.columns.tolist()

    if emit_setup_logs:
        log(
            f"Loaded dataset with {len(returns)} monthly rows, "
            f"{n_assets} assets, date range {returns.index.min().date()} to {returns.index.max().date()}."
        )
        log(split_note)
        log(f"Using config settings | population={pop_size} | generations={generations} | runs={runs}")

    if representative_results is None:
        computed_outputs = run_model_suite(
            train_returns,
            pop_size=pop_size,
            generations=generations,
            runs=runs,
            verbose=True,
        )
        representative_results = {slug: payload["results"] for slug, payload in computed_outputs.items()}

    averaged_weights = {}
    model_monthly_returns = {}
    skipped_models = []
    ordered_model_titles = [title for title, _slug, _objective_fn in MODEL_SPECS]

    for title, slug, _objective_fn in MODEL_SPECS:
        results = representative_results.get(slug, [])

        reps_df = representative_weights_frame(results, asset_names)
        reps_df.to_csv(results_dir / f"{slug}_representative_portfolios.csv", index=False)

        avg_weights = average_weights(results, asset_names)
        avg_weights.rename("weight").to_frame().to_csv(results_dir / f"{slug}_average_weights.csv")

        averaged_weights[title] = avg_weights
        if avg_weights.sum() > 0:
            model_monthly_returns[title] = average_portfolio_returns_from_results(test_returns, results, title)
        else:
            skipped_models.append(title)

        log(f"{title}: benchmark inputs saved for {len(results)} representatives.")

    if skipped_models:
        log(f"Skipping benchmark backtest for models with no surviving representatives: {', '.join(skipped_models)}")

    if not averaged_weights:
        raise RuntimeError(
            "No model produced representative portfolios under the current settings. "
            "Try a less strict filter or a larger run profile."
        )

    if not allow_proxy and benchmark_returns is None and benchmark_prices is None:
        raise RuntimeError("A real benchmark file is required. Provide --benchmark-returns or --benchmark-prices.")

    proxy_series = test_returns.mean(axis=1).rename("Universe EW Proxy") if allow_proxy else None
    loaded_benchmark_returns, benchmark_label = load_benchmark_returns(
        target_index=test_returns.index,
        returns_path=benchmark_returns,
        prices_path=benchmark_prices,
        column=benchmark_column,
        proxy=proxy_series,
    )

    monthly_returns, quarterly_returns = backtest_models(
        test_returns=test_returns,
        model_monthly_returns=model_monthly_returns,
        benchmark_returns=loaded_benchmark_returns,
        expected_model_names=ordered_model_titles,
    )

    monthly_returns.to_csv(results_dir / "benchmark_monthly_returns.csv", index_label="Month")
    quarterly_returns.to_csv(results_dir / "benchmark_quarterly_returns.csv", index_label="Quarter")

    allocation_table = pd.DataFrame(averaged_weights).rename_axis("Asset")
    allocation_table.to_csv(results_dir / "benchmark_average_allocations.csv")

    monthly_plot = plot_cumulative_returns_from_table(
        monthly_returns,
        title=f"Cumulative Monthly Returns vs {benchmark_label}",
        file_name="benchmark_cumulative_monthly_returns.png",
        show=False,
        results_dir=results_dir,
    )
    quarterly_plot = plot_quarterly_returns_from_table(
        quarterly_returns,
        title=f"Cumulative Quarterly Average Returns vs {benchmark_label}",
        file_name="benchmark_cumulative_quarterly_returns.png",
        show=False,
        results_dir=results_dir,
    )

    log(f"Benchmark label: {benchmark_label}")
    if emit_setup_logs:
        log(f"Training months: {len(train_returns)} | Test months: {len(test_returns)}")
    log(f"Saved monthly comparison: {results_dir / 'benchmark_monthly_returns.csv'}")
    log(f"Saved quarterly comparison: {results_dir / 'benchmark_quarterly_returns.csv'}")
    log(f"Saved average allocations: {results_dir / 'benchmark_average_allocations.csv'}")
    log(f"Saved plots: {monthly_plot} | {quarterly_plot}")

    return {
        "representative_results": representative_results,
        "averaged_weights": averaged_weights,
        "monthly_returns": monthly_returns,
        "quarterly_returns": quarterly_returns,
        "results_dir": results_dir,
        "benchmark_label": benchmark_label,
    }


def main() -> None:
    args = parse_args()
    run_benchmark_pipeline(
        results_dir=args.results_dir,
        benchmark_returns=args.benchmark_returns,
        benchmark_prices=args.benchmark_prices,
        benchmark_column=args.benchmark_column,
        allow_proxy=True,
    )


if __name__ == "__main__":
    main()
