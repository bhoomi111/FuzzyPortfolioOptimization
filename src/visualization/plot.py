from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from src.config import RESULTS_DIR


def _save_and_optionally_show(fig, file_name: str, show: bool = True, results_dir: Path = RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / file_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _risk_label(model_name: str) -> str:
    labels = {
        "model1": "Risk (Semivariance)",
        "model2": "Risk (MASD)",
        "model3": "Risk (CVaR)",
    }
    return labels.get(model_name.lower(), "Risk")


# Pareto risk-return scatter.
def plot_pareto(population, objective_fn, show: bool = True, results_dir: Path = RESULTS_DIR):
    risks = []
    returns_ = []

    for w in population:
        f = objective_fn(w)
        risks.append(f[1])
        returns_.append(-f[0])  # objective is minimization, so flip return sign.

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(risks, returns_, alpha=0.7)
    ax.set_xlabel("Risk (Semivariance)")
    ax.set_ylabel("Return")
    ax.set_title("Pareto Front: Risk vs Return")
    ax.grid(True, alpha=0.3)

    return _save_and_optionally_show(fig, "pareto_front.png", show=show, results_dir=results_dir)


# Risk-skewness profile of the same population.
def plot_skewness(population, objective_fn, show: bool = True, results_dir: Path = RESULTS_DIR):
    risks = []
    skew = []

    for w in population:
        f = objective_fn(w)
        risks.append(f[1])
        skew.append(-f[2])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(risks, skew, color="green", alpha=0.75)
    ax.set_xlabel("Risk")
    ax.set_ylabel("Skewness")
    ax.set_title("Risk vs Skewness")
    ax.grid(True, alpha=0.3)

    return _save_and_optionally_show(fig, "risk_vs_skewness.png", show=show, results_dir=results_dir)


# Heatmap of asset allocations across selected portfolios.
def plot_allocations(portfolios, show: bool = True, results_dir: Path = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(portfolios, aspect="auto")
    fig.colorbar(im, ax=ax, label="Weight")
    ax.set_xlabel("Assets")
    ax.set_ylabel("Portfolio Index")
    ax.set_title("Portfolio Allocations")

    return _save_and_optionally_show(fig, "portfolio_allocations.png", show=show, results_dir=results_dir)


# Sorted frontier line plot.
def plot_frontier(population, objective_fn, show: bool = True, results_dir: Path = RESULTS_DIR):
    points = []

    for w in population:
        f = objective_fn(w)
        points.append((f[1], -f[0]))

    if not points:
        raise ValueError("Population is empty; cannot plot frontier.")

    points.sort()
    risks, returns_ = zip(*points)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(risks, returns_, marker="o")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier (Approx)")
    ax.grid(True, alpha=0.3)

    return _save_and_optionally_show(fig, "efficient_frontier.png", show=show, results_dir=results_dir)


# Table-based plotting for saved model outputs.
def plot_pareto_from_table(table: pd.DataFrame, model_name: str, show: bool = False, results_dir: Path = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(table["Risk"], table["Mean"], alpha=0.75)
    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Return (Mean)")
    ax.set_title(f"Pareto Front: {model_name}")
    ax.grid(True, alpha=0.3)

    file_name = f"{model_name.lower()}_pareto_front.png"
    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def plot_skewness_from_table(table: pd.DataFrame, model_name: str, show: bool = False, results_dir: Path = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(table["Risk"], table["Skewness"], color="green", alpha=0.75)
    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Skewness")
    ax.set_title(f"Risk vs Skewness: {model_name}")
    ax.grid(True, alpha=0.3)

    file_name = f"{model_name.lower()}_risk_vs_skewness.png"
    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def plot_frontier_from_table(table: pd.DataFrame, model_name: str, show: bool = False, results_dir: Path = RESULTS_DIR):
    sorted_table = table.sort_values("Risk")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sorted_table["Risk"], sorted_table["Mean"], marker="o")
    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Return (Mean)")
    ax.set_title(f"Efficient Frontier (Approx): {model_name}")
    ax.grid(True, alpha=0.3)

    file_name = f"{model_name.lower()}_efficient_frontier.png"
    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def plot_allocations_from_table(portfolios: pd.DataFrame, model_name: str, show: bool = False, results_dir: Path = RESULTS_DIR):
    value_cols = [c for c in portfolios.columns if c != "Rep"]
    matrix = portfolios[value_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto")
    fig.colorbar(im, ax=ax, label="Weight")
    ax.set_xlabel("Assets")
    ax.set_ylabel("Portfolio Index")
    ax.set_title(f"Portfolio Allocations: {model_name}")

    file_name = f"{model_name.lower()}_portfolio_allocations.png"
    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def plot_cumulative_returns_from_table(
    returns_table: pd.DataFrame,
    title: str,
    file_name: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    cumulative = (1.0 + returns_table).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column], marker="o", linewidth=2, label=column)

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Cumulative Return")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(list(cumulative.index))

    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def plot_quarterly_returns_from_table(
    returns_table: pd.DataFrame,
    title: str,
    file_name: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    cumulative = (1.0 + returns_table).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    columns = list(cumulative.columns)
    x = list(cumulative.index)
    width = 0.18 if len(columns) >= 4 else 0.22
    offsets = [((i - (len(columns) - 1) / 2) * width) for i in range(len(columns))]

    for idx, column in enumerate(columns):
        xpos = [value + offsets[idx] for value in x]
        ax.bar(xpos, cumulative[column], width=width, label=column)

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Quarterly Average Return")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])

    return _save_and_optionally_show(fig, file_name, show=show, results_dir=results_dir)


def generate_plots_from_results(results_dir: Path = RESULTS_DIR, show: bool = False):
    generated = []

    for model_name in ["model1", "model2", "model3"]:
        table_path = results_dir / f"{model_name}_table.csv"
        portfolios_path = results_dir / f"{model_name}_portfolios.csv"

        if table_path.exists():
            try:
                table = pd.read_csv(table_path)
            except pd.errors.EmptyDataError:
                table = pd.DataFrame()

            if not table.empty:
                generated.append(plot_pareto_from_table(table, model_name, show=show, results_dir=results_dir))
                generated.append(plot_frontier_from_table(table, model_name, show=show, results_dir=results_dir))
                generated.append(plot_skewness_from_table(table, model_name, show=show, results_dir=results_dir))

        if portfolios_path.exists():
            try:
                portfolios = pd.read_csv(portfolios_path)
            except pd.errors.EmptyDataError:
                portfolios = pd.DataFrame()

            if not portfolios.empty:
                generated.append(plot_allocations_from_table(portfolios, model_name, show=show, results_dir=results_dir))
    return generated


if __name__ == "__main__":
    output_paths = generate_plots_from_results(show=False)
    if not output_paths:
        print(f"No results tables found in {RESULTS_DIR}. Run scripts/run_tables.py first.")
    else:
        print("Saved plot files:")
        for path in output_paths:
            print(path)
