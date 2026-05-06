from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROLLING_TEST_YEARS


def load_monthly_returns(csv_path: Path) -> pd.DataFrame:
    returns = pd.read_csv(csv_path)
    returns["Date"] = pd.to_datetime(returns["Date"])
    returns = returns.set_index("Date").sort_index()
    return returns.astype(float)


def split_last_n_months(
    returns: pd.DataFrame,
    test_months: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(returns) <= test_months:
        raise ValueError(
            f"Need more than {test_months} monthly observations to create a train/test split; "
            f"found only {len(returns)} rows."
        )

    train_returns = returns.iloc[:-test_months].copy()
    test_returns = returns.iloc[-test_months:].copy()
    split_note = (
        f"Dataset-driven split used: training on the first {len(train_returns)} months "
        f"({train_returns.index.min().date()} to {train_returns.index.max().date()}) and "
        f"testing on the last {len(test_returns)} months "
        f"({test_returns.index.min().date()} to {test_returns.index.max().date()})."
    )
    return train_returns, test_returns, split_note


def split_by_calendar_year(
    returns: pd.DataFrame,
    test_year: int,
    train_start_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if returns.empty:
        raise ValueError("Cannot split an empty returns frame.")

    start_year = train_start_year or int(returns.index.min().year)
    train_start = f"{start_year}-01-01"
    train_end = f"{test_year - 1}-12-31"
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"

    train_returns = returns.loc[train_start:train_end].copy()
    test_returns = returns.loc[test_start:test_end].copy()

    if train_returns.empty:
        raise ValueError(
            f"No training data available before test year {test_year}. "
            f"Requested train window {train_start} to {train_end}."
        )

    if test_returns.empty:
        raise ValueError(f"No test data available for calendar year {test_year}.")

    split_note = (
        f"Calendar-year split used: training on {train_returns.index.min().date()} to {train_returns.index.max().date()} "
        f"and testing on {test_returns.index.min().date()} to {test_returns.index.max().date()}."
    )
    return train_returns, test_returns, split_note


def rolling_year_splits(
    returns: pd.DataFrame,
    test_years: tuple[int, ...] = ROLLING_TEST_YEARS,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, str]]:
    splits = []
    for test_year in test_years:
        train_returns, test_returns, split_note = split_by_calendar_year(returns, test_year)
        splits.append((test_year, train_returns, test_returns, split_note))
    return splits


def representative_weights_frame(results: list[dict], asset_names: list[str]) -> pd.DataFrame:
    columns = [
        "Rep",
        "Center position",
        "cp",
        "mp",
        "run",
        "Total solutions",
        *asset_names,
    ]
    if not results:
        return pd.DataFrame(columns=columns)

    rows = []

    for i, result in enumerate(results, start=1):
        row = {
            "Rep": f"R{i}",
            "Center position": result.get("center_position"),
            "cp": result["cp"],
            "mp": result["mp"],
            "run": result.get("run", 1),
            "Total solutions": result.get("total_pareto_solutions"),
        }
        row.update(dict(zip(asset_names, result["weights"])))
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def average_weights(results: list[dict], asset_names: list[str]) -> pd.Series:
    if not results:
        return pd.Series(0.0, index=asset_names, dtype=float)

    matrix = np.vstack([result["weights"] for result in results])
    avg = matrix.mean(axis=0)
    avg = avg / avg.sum()
    return pd.Series(avg, index=asset_names, dtype=float)


def average_portfolio_returns_from_results(returns: pd.DataFrame, results: list[dict], name: str) -> pd.Series:
    if not results:
        raise ValueError(f"No representative results available for {name}.")

    series_list = []
    for result in results:
        weights = pd.Series(result["weights"], index=returns.columns, dtype=float)
        series_list.append(portfolio_returns_from_weights(returns, weights))

    average_series = pd.concat(series_list, axis=1).mean(axis=1)
    average_series.name = name
    return average_series


def portfolio_returns_from_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(returns.columns).fillna(0.0)
    total = aligned.sum()

    if total <= 0:
        raise ValueError("Portfolio weights sum to zero; cannot compute benchmark returns.")

    normalized = aligned / total
    series = returns.mul(normalized, axis=1).sum(axis=1)
    series.name = "return"
    return series


def equal_weight_proxy_returns(returns: pd.DataFrame, name: str = "Universe EW Proxy") -> pd.Series:
    proxy = returns.mean(axis=1)
    proxy.name = name
    return proxy


def load_benchmark_returns(
    target_index: pd.Index,
    returns_path: Path | None = None,
    prices_path: Path | None = None,
    column: str | None = None,
    proxy: pd.Series | None = None,
) -> tuple[pd.Series, str]:
    benchmark = None
    label = column or "Benchmark"
    target = pd.DatetimeIndex(pd.to_datetime(target_index))
    if getattr(target, "tz", None) is not None:
        target = target.tz_localize(None)
    target_periods = target.to_period("M")

    if returns_path and returns_path.exists():
        benchmark = pd.read_csv(returns_path)
        benchmark["Date"] = pd.to_datetime(benchmark["Date"])
        benchmark = benchmark.set_index("Date").sort_index()
    elif prices_path and prices_path.exists():
        prices = pd.read_csv(prices_path)
        prices["Date"] = pd.to_datetime(prices["Date"])
        prices = prices.set_index("Date").sort_index()
        benchmark = prices.pct_change().dropna(how="all")

    if benchmark is not None:
        candidate_cols = [c for c in benchmark.columns if c != "Date"]
        selected = column if column in benchmark.columns else candidate_cols[0]
        benchmark_index = pd.DatetimeIndex(pd.to_datetime(benchmark.index))
        if getattr(benchmark_index, "tz", None) is not None:
            benchmark_index = benchmark_index.tz_localize(None)
        benchmark_series = benchmark[selected].copy()
        benchmark_series.index = benchmark_index.to_period("M")
        benchmark_series = benchmark_series[~benchmark_series.index.duplicated(keep="last")]
        aligned = benchmark_series.reindex(target_periods)
        aligned.index = target_index
        aligned.name = selected
        return aligned.astype(float), selected

    if proxy is None:
        raise FileNotFoundError("No benchmark series was provided and no proxy series was available.")

    aligned_proxy = proxy.reindex(target_index)
    aligned_proxy.name = proxy.name or "Universe EW Proxy"
    return aligned_proxy.astype(float), aligned_proxy.name


def cumulative_from_returns(returns: pd.Series) -> pd.Series:
    cumulative = (1.0 + returns).cumprod() - 1.0
    cumulative.name = returns.name
    return cumulative


def quarterly_average_returns(returns_frame: pd.DataFrame) -> pd.DataFrame:
    quarterly = returns_frame.copy()
    quarterly.index = pd.to_datetime(quarterly.index)
    if getattr(quarterly.index, "tz", None) is not None:
        quarterly.index = quarterly.index.tz_localize(None)
    grouped = quarterly.groupby(quarterly.index.to_period("Q")).mean()
    grouped.index = pd.RangeIndex(1, len(grouped) + 1, name="Quarter")
    return grouped


def backtest_models(
    test_returns: pd.DataFrame,
    model_monthly_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    expected_model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.DataFrame(index=test_returns.index)

    ordered_model_names = expected_model_names or list(model_monthly_returns.keys())
    for model_name in ordered_model_names:
        monthly_series = model_monthly_returns.get(model_name)
        if monthly_series is None:
            monthly[model_name] = np.nan
        else:
            monthly[model_name] = monthly_series.reindex(test_returns.index)

    monthly[benchmark_returns.name] = benchmark_returns.reindex(monthly.index)
    quarterly = quarterly_average_returns(monthly)

    monthly = monthly.copy()
    monthly.index = pd.RangeIndex(1, len(monthly) + 1, name="Month")
    return monthly, quarterly
