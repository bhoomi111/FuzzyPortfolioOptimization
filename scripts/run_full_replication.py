from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_benchmark import run_benchmark_pipeline
from scripts.run_tables import load_dataset_split, run_model_suite, run_tables_pipeline
from src.config import BENCHMARK_COLUMN, BENCHMARK_RETURNS_PATH, FULL_REPLICATION_DIR
from src.visualization.plot import generate_plots_from_results


DEFAULT_NIFTY_RETURNS = BENCHMARK_RETURNS_PATH
DEFAULT_OUTPUT_ROOT = FULL_REPLICATION_DIR


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tables, model plots, and NIFTY benchmark outputs together.")
    parser.add_argument("--benchmark-returns", type=Path, default=DEFAULT_NIFTY_RETURNS)
    parser.add_argument("--benchmark-column", type=str, default=BENCHMARK_COLUMN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.benchmark_returns.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {args.benchmark_returns}. "
            "Download or place the real NIFTY 50 monthly returns file before running full replication."
        )

    tables_dir = args.output_root / "tables_and_model_plots"
    benchmark_dir = args.output_root / "benchmark_vs_nifty50"
    tables_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    log("Full replication started using config.py settings.")

    returns, train_returns, _test_returns, _split_note = load_dataset_split()
    log(
        f"Loaded dataset with {len(returns)} monthly rows, {returns.shape[1]} assets, "
        f"date range {returns.index.min().date()} to {returns.index.max().date()}."
    )
    model_outputs = run_model_suite(train_returns, verbose=True)

    log("Step 1/3: saving tables and representative portfolio CSVs.")
    model_outputs = run_tables_pipeline(
        results_dir=tables_dir,
        model_outputs=model_outputs,
        show_tables=False,
        emit_setup_logs=False,
    )

    log("Step 2/3: generating model plots from saved tables.")
    generated_plots = generate_plots_from_results(results_dir=tables_dir, show=False)
    log(f"Generated {len(generated_plots)} model plot files.")

    log("Step 3/3: generating benchmark comparison against real NIFTY 50.")
    run_benchmark_pipeline(
        results_dir=benchmark_dir,
        benchmark_returns=args.benchmark_returns,
        benchmark_prices=None,
        benchmark_column=args.benchmark_column,
        allow_proxy=False,
        representative_results={slug: payload["representatives"] for slug, payload in model_outputs.items()},
        emit_setup_logs=False,
    )

    log(f"Finished. Tables/model plots: {tables_dir}")
    log(f"Finished. Benchmark outputs: {benchmark_dir}")


if __name__ == "__main__":
    main()
