from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_benchmark import run_benchmark_pipeline
from scripts.run_tables import run_model_suite, run_tables_pipeline
from src.config import (
    BENCHMARK_COLUMN,
    FULL_REPLICATION_DIR,
    FUZZY_FIT_METHOD,
    FUZZY_QUANTILES,
    FUZZY_ROLLING_WINDOW,
    GENERATIONS,
    MOMENTUM_LOOKBACK,
    MOMENTUM_WEIGHT,
    POP_SIZE,
    RUNS,
    TEST_MONTHS,
)
from src.evaluation.benchmark import load_monthly_returns, split_last_n_months
from src.visualization.plot import generate_plots_from_results, plot_cumulative_returns_from_table


MODEL_COLUMN_ALIASES = {
    "model i results": "MODEL I RESULTS",
    "model ii results": "MODEL II RESULTS",
    "model iii results": "MODEL III RESULTS",
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _normalize_model_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def _canonicalize_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        key = _normalize_model_name(column)
        if key in MODEL_COLUMN_ALIASES:
            renamed[column] = MODEL_COLUMN_ALIASES[key]
    return df.rename(columns=renamed)


def _drop_benchmark_column(df: pd.DataFrame, benchmark_column: str) -> pd.DataFrame:
    normalized_target = _normalize_model_name(benchmark_column)
    to_drop = [c for c in df.columns if _normalize_model_name(c) == normalized_target]
    if to_drop:
        return df.drop(columns=to_drop)
    return df


def _comparison_frame(baseline_path: Path, modified_path: Path, benchmark_column: str) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path).set_index("Month")
    modified = pd.read_csv(modified_path).set_index("Month")

    baseline = _canonicalize_monthly_columns(baseline)
    modified = _canonicalize_monthly_columns(modified)

    baseline = _drop_benchmark_column(baseline, benchmark_column)
    modified = _drop_benchmark_column(modified, benchmark_column)

    common_columns = [c for c in baseline.columns if c in modified.columns]
    if not common_columns:
        raise ValueError(
            "No common model columns found for comparison. "
            f"Baseline columns={list(baseline.columns)} | Modified columns={list(modified.columns)}"
        )

    baseline = baseline[common_columns].add_prefix("Baseline - ")
    modified = modified[common_columns].add_prefix("Modified - ")
    return pd.concat([baseline, modified], axis=1)


def _run_variant(
    variant_name: str,
    root_dir: Path,
    pop_size: int,
    generations: int,
    runs: int,
    momentum_weight: float,
    momentum_lookback: int,
    optimizer: str,
    crossover_mode: str,
    fuzzy_fit_method: str,
    fuzzy_window: int,
    fuzzy_quantiles: tuple[float, float, float],
    fuzzy_min_periods: int | None,
):
    variant_dir = root_dir / variant_name
    tables_dir = variant_dir / "tables_and_model_plots"
    benchmark_dir = variant_dir / "benchmark_vs_nifty50"

    returns = load_monthly_returns(ROOT / "data" / "raw" / "monthly_returns.csv")
    _train_returns, _test_returns, _split_note = split_last_n_months(returns, test_months=TEST_MONTHS)
    log(
        f"[{variant_name}] Loaded dataset with {len(returns)} monthly rows, "
        f"{returns.shape[1]} assets, date range {returns.index.min().date()} to {returns.index.max().date()}."
    )

    model_outputs = run_model_suite(
        _train_returns,
        pop_size=pop_size,
        generations=generations,
        runs=runs,
        verbose=True,
        momentum_weight=momentum_weight,
        momentum_lookback=momentum_lookback,
        fuzzy_fit_method=fuzzy_fit_method,
        fuzzy_window=fuzzy_window,
        fuzzy_quantiles=fuzzy_quantiles,
        fuzzy_min_periods=fuzzy_min_periods,
        optimizer=optimizer,
        crossover_mode=crossover_mode,
    )

    run_tables_pipeline(
        pop_size=pop_size,
        generations=generations,
        runs=runs,
        results_dir=tables_dir,
        model_outputs=model_outputs,
        show_tables=False,
        emit_setup_logs=False,
        fuzzy_fit_method=fuzzy_fit_method,
        fuzzy_window=fuzzy_window,
        fuzzy_quantiles=fuzzy_quantiles,
        fuzzy_min_periods=fuzzy_min_periods,
        optimizer=optimizer,
        crossover_mode=crossover_mode,
    )

    generated = generate_plots_from_results(results_dir=tables_dir, show=False)
    log(f"[{variant_name}] Generated {len(generated)} model plot files.")

    reps = {slug: payload["representatives"] for slug, payload in model_outputs.items()}
    benchmark_payload = run_benchmark_pipeline(
        pop_size=pop_size,
        generations=generations,
        runs=runs,
        results_dir=benchmark_dir,
        benchmark_column=BENCHMARK_COLUMN,
        allow_proxy=True,
        representative_results=reps,
        emit_setup_logs=False,
        rolling_years=None,
        momentum_weight=momentum_weight,
        momentum_lookback=momentum_lookback,
        fuzzy_fit_method=fuzzy_fit_method,
        fuzzy_window=fuzzy_window,
        fuzzy_quantiles=fuzzy_quantiles,
        fuzzy_min_periods=fuzzy_min_periods,
        optimizer=optimizer,
        crossover_mode=crossover_mode,
    )

    return {
        "variant_dir": variant_dir,
        "tables_dir": tables_dir,
        "benchmark_dir": benchmark_dir,
        "benchmark_label": benchmark_payload["benchmark_label"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline and modified full replication variants, then build comparison plots."
    )
    parser.add_argument("--results-dir", type=Path, default=FULL_REPLICATION_DIR)

    parser.add_argument("--baseline-name", type=str, default="baseline_static_moga")
    parser.add_argument("--baseline-optimizer", type=str, default="moga", choices=["moga", "nsga2"])
    parser.add_argument("--baseline-fuzzy-fit-method", type=str, default=FUZZY_FIT_METHOD, choices=["static", "rolling"])

    parser.add_argument("--modified-name", type=str, default="modified_nsga2_static")
    parser.add_argument("--modified-optimizer", type=str, default="nsga2", choices=["moga", "nsga2"])
    parser.add_argument("--modified-fuzzy-fit-method", type=str, default="static", choices=["static", "rolling"])

    parser.add_argument("--rolling-window", type=int, default=FUZZY_ROLLING_WINDOW)
    parser.add_argument("--fuzzy-quantiles", type=str, default=",".join(map(str, FUZZY_QUANTILES)))
    parser.add_argument("--fuzzy-min-periods", type=int, default=None)
    parser.add_argument("--crossover-mode", type=str, default="sbx", choices=["sbx", "convex"])

    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--momentum-weight", type=float, default=MOMENTUM_WEIGHT)
    parser.add_argument("--momentum-lookback", type=int, default=MOMENTUM_LOOKBACK)
    parser.add_argument("--comparison-dir", type=str, default="comparison_plots_nsga2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantiles = tuple(float(x) for x in args.fuzzy_quantiles.split(","))

    log("Full replication started using config.py settings.")

    log("Step 1/4: running baseline pipeline.")
    baseline = _run_variant(
        variant_name=args.baseline_name,
        root_dir=args.results_dir,
        pop_size=args.pop_size,
        generations=args.generations,
        runs=args.runs,
        momentum_weight=args.momentum_weight,
        momentum_lookback=args.momentum_lookback,
        optimizer=args.baseline_optimizer,
        crossover_mode=args.crossover_mode,
        fuzzy_fit_method=args.baseline_fuzzy_fit_method,
        fuzzy_window=args.rolling_window,
        fuzzy_quantiles=quantiles,
        fuzzy_min_periods=args.fuzzy_min_periods,
    )

    log("Step 2/4: running modified pipeline.")
    modified = _run_variant(
        variant_name=args.modified_name,
        root_dir=args.results_dir,
        pop_size=args.pop_size,
        generations=args.generations,
        runs=args.runs,
        momentum_weight=args.momentum_weight,
        momentum_lookback=args.momentum_lookback,
        optimizer=args.modified_optimizer,
        crossover_mode=args.crossover_mode,
        fuzzy_fit_method=args.modified_fuzzy_fit_method,
        fuzzy_window=args.rolling_window,
        fuzzy_quantiles=quantiles,
        fuzzy_min_periods=args.fuzzy_min_periods,
    )

    log("Step 3/4: building comparison line plots without benchmark.")
    comparison_dir = args.results_dir / args.comparison_dir
    comparison_dir.mkdir(parents=True, exist_ok=True)

    baseline_monthly = baseline["benchmark_dir"] / "benchmark_monthly_returns.csv"
    modified_monthly = modified["benchmark_dir"] / "benchmark_monthly_returns.csv"
    frame = _comparison_frame(
        baseline_path=baseline_monthly,
        modified_path=modified_monthly,
        benchmark_column=BENCHMARK_COLUMN,
    )

    frame.to_csv(comparison_dir / "comparison_monthly_returns.csv", index_label="Month")

    all_plot = plot_cumulative_returns_from_table(
        frame,
        title=(
            f"Cumulative Monthly Returns: {args.baseline_name} vs {args.modified_name} "
            f"(without {BENCHMARK_COLUMN})"
        ),
        file_name="comparison_cumulative_monthly_returns.png",
        show=False,
        results_dir=comparison_dir,
    )

    canonical_models = ["MODEL I RESULTS", "MODEL II RESULTS", "MODEL III RESULTS"]
    for model_name in canonical_models:
        baseline_col = f"Baseline - {model_name}"
        modified_col = f"Modified - {model_name}"
        if baseline_col not in frame.columns or modified_col not in frame.columns:
            continue
        per_model = frame[[baseline_col, modified_col]].copy()
        safe = model_name.lower().replace(" ", "_")
        plot_cumulative_returns_from_table(
            per_model,
            title=f"Cumulative Monthly Returns: {model_name}",
            file_name=f"comparison_{safe}.png",
            show=False,
            results_dir=comparison_dir,
        )

    log(f"Comparison frame saved to {comparison_dir / 'comparison_monthly_returns.csv'}")
    log(f"Comparison plot saved to {all_plot}")

    log("Step 4/4: full replication completed successfully.")


if __name__ == "__main__":
    main()
