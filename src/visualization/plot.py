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


def _remove_stale_plot(file_name: str, results_dir: Path = RESULTS_DIR):
    output_path = results_dir / file_name
    if output_path.exists():
        output_path.unlink()


def _model_table_plot_names(model_name: str) -> list[str]:
    slug = model_name.lower()
    return [
        f"{slug}_pareto_front.png",
        f"{slug}_efficient_frontier.png",
        f"{slug}_risk_vs_skewness.png",
        f"{slug}_global_pareto_front.png",
        f"{slug}_global_efficient_frontier.png",
        f"{slug}_global_risk_vs_skewness.png",
        f"{slug}_filtered_pareto_front.png",
        f"{slug}_filtered_efficient_frontier.png",
        f"{slug}_filtered_risk_vs_skewness.png",
    ]


def _model_portfolio_plot_names(model_name: str) -> list[str]:
    slug = model_name.lower()
    return [
        f"{slug}_portfolio_allocations.png",
        f"{slug}_global_portfolio_allocations.png",
        f"{slug}_filtered_portfolio_allocations.png",
    ]


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _stage_styles():
    return {
        "global": {
            "label": "Global Pareto",
            "color": "#9aa5b1",
            "alpha": 0.35,
            "marker": "o",
            "linewidth": 1.2,
            "zorder": 1,
        },
        "filtered": {
            "label": "Filtered Pareto",
            "color": "#1f77b4",
            "alpha": 1.0,
            "marker": "o",
            "linewidth": 1.6,
            "zorder": 2,
        },
        "representatives": {
            "label": "Representatives",
            "color": "#d62728",
            "alpha": 0.95,
            "marker": "D",
            "linewidth": 2.1,
            "zorder": 3,
        },
    }


def _plot_combined_pareto(
    tables: dict[str, pd.DataFrame],
    model_name: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for stage_name, style in _stage_styles().items():
        table = tables.get(stage_name, pd.DataFrame())
        if table.empty:
            continue
        if stage_name == "filtered":
            ax.scatter(
                table["Risk"],
                table["Mean"],
                facecolors="none",
                edgecolors=style["color"],
                alpha=style["alpha"],
                marker=style["marker"],
                s=120,
                linewidths=2.0,
                label=style["label"],
                zorder=style["zorder"],
            )
        else:
            ax.scatter(
                table["Risk"],
                table["Mean"],
                color=style["color"],
                alpha=style["alpha"],
                marker=style["marker"],
                s=60 if stage_name == "representatives" else 36,
                label=style["label"],
                zorder=style["zorder"],
            )

    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Return (Mean)")
    ax.set_title(f"Pareto Front: {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_and_optionally_show(fig, f"{model_name.lower()}_pareto_front.png", show=show, results_dir=results_dir)


def _plot_combined_frontier(
    tables: dict[str, pd.DataFrame],
    model_name: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for stage_name, style in _stage_styles().items():
        table = tables.get(stage_name, pd.DataFrame())
        if table.empty:
            continue
        sorted_table = table.sort_values("Risk")
        ax.plot(
            sorted_table["Risk"],
            sorted_table["Mean"],
            color=style["color"],
            alpha=style["alpha"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            label=style["label"],
            zorder=style["zorder"],
        )

    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Return (Mean)")
    ax.set_title(f"Efficient Frontier (Approx): {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_and_optionally_show(fig, f"{model_name.lower()}_efficient_frontier.png", show=show, results_dir=results_dir)


def _plot_combined_skewness(
    tables: dict[str, pd.DataFrame],
    model_name: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for stage_name, style in _stage_styles().items():
        table = tables.get(stage_name, pd.DataFrame())
        if table.empty:
            continue
        if stage_name == "filtered":
            ax.scatter(
                table["Risk"],
                table["Skewness"],
                facecolors="none",
                edgecolors=style["color"],
                alpha=style["alpha"],
                marker=style["marker"],
                s=120,
                linewidths=2.0,
                label=style["label"],
                zorder=style["zorder"],
            )
        else:
            ax.scatter(
                table["Risk"],
                table["Skewness"],
                color=style["color"],
                alpha=style["alpha"],
                marker=style["marker"],
                s=60 if stage_name == "representatives" else 36,
                label=style["label"],
                zorder=style["zorder"],
            )

    ax.set_xlabel(_risk_label(model_name))
    ax.set_ylabel("Skewness")
    ax.set_title(f"Risk vs Skewness: {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_and_optionally_show(fig, f"{model_name.lower()}_risk_vs_skewness.png", show=show, results_dir=results_dir)


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


def plot_table_set(
    table: pd.DataFrame,
    portfolios: pd.DataFrame,
    model_name: str,
    file_stub: str,
    title_suffix: str,
    show: bool = False,
    results_dir: Path = RESULTS_DIR,
):
    generated = []

    if not table.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(table["Risk"], table["Mean"], alpha=0.75)
        ax.set_xlabel(_risk_label(model_name))
        ax.set_ylabel("Return (Mean)")
        ax.set_title(f"Pareto Front: {model_name} {title_suffix}".strip())
        ax.grid(True, alpha=0.3)
        generated.append(_save_and_optionally_show(fig, f"{file_stub}_pareto_front.png", show=show, results_dir=results_dir))

        sorted_table = table.sort_values("Risk")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(sorted_table["Risk"], sorted_table["Mean"], marker="o")
        ax.set_xlabel(_risk_label(model_name))
        ax.set_ylabel("Return (Mean)")
        ax.set_title(f"Efficient Frontier (Approx): {model_name} {title_suffix}".strip())
        ax.grid(True, alpha=0.3)
        generated.append(_save_and_optionally_show(fig, f"{file_stub}_efficient_frontier.png", show=show, results_dir=results_dir))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(table["Risk"], table["Skewness"], color="green", alpha=0.75)
        ax.set_xlabel(_risk_label(model_name))
        ax.set_ylabel("Skewness")
        ax.set_title(f"Risk vs Skewness: {model_name} {title_suffix}".strip())
        ax.grid(True, alpha=0.3)
        generated.append(_save_and_optionally_show(fig, f"{file_stub}_risk_vs_skewness.png", show=show, results_dir=results_dir))

    if not portfolios.empty:
        value_cols = [c for c in portfolios.columns if c != "Rep"]
        matrix = portfolios[value_cols].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, aspect="auto")
        fig.colorbar(im, ax=ax, label="Weight")
        ax.set_xlabel("Assets")
        ax.set_ylabel("Portfolio Index")
        ax.set_title(f"Portfolio Allocations: {model_name} {title_suffix}".strip())
        generated.append(_save_and_optionally_show(fig, f"{file_stub}_portfolio_allocations.png", show=show, results_dir=results_dir))

    return generated


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
        stages = [
            {
                "table_path": results_dir / f"{model_name}_table.csv",
                "portfolios_path": results_dir / f"{model_name}_portfolios.csv",
                "file_stub": model_name,
                "title_suffix": "Representatives",
                "table_key": "representatives",
            },
            {
                "table_path": results_dir / f"{model_name}_global_pareto_table.csv",
                "portfolios_path": results_dir / f"{model_name}_global_pareto_portfolios.csv",
                "file_stub": f"{model_name}_global",
                "title_suffix": "Global Pareto",
                "table_key": "global",
            },
            {
                "table_path": results_dir / f"{model_name}_filtered_pareto_table.csv",
                "portfolios_path": results_dir / f"{model_name}_filtered_pareto_portfolios.csv",
                "file_stub": f"{model_name}_filtered",
                "title_suffix": "Filtered Pareto",
                "table_key": "filtered",
            },
        ]

        any_table_nonempty = False
        any_portfolios_nonempty = False
        tables_by_stage = {}

        for stage in stages:
            table = _read_csv_or_empty(stage["table_path"])
            portfolios = _read_csv_or_empty(stage["portfolios_path"])
            tables_by_stage[stage["table_key"]] = table

            if not table.empty:
                any_table_nonempty = True
            if not portfolios.empty:
                any_portfolios_nonempty = True

            if not portfolios.empty:
                generated.extend(
                    plot_table_set(
                        table=pd.DataFrame(),
                        portfolios=portfolios,
                        model_name=model_name,
                        file_stub=stage["file_stub"],
                        title_suffix=stage["title_suffix"],
                        show=show,
                        results_dir=results_dir,
                    )
                )

        if any_table_nonempty:
            generated.append(_plot_combined_pareto(tables_by_stage, model_name, show=show, results_dir=results_dir))
            generated.append(_plot_combined_frontier(tables_by_stage, model_name, show=show, results_dir=results_dir))
            generated.append(_plot_combined_skewness(tables_by_stage, model_name, show=show, results_dir=results_dir))

        if not any_table_nonempty:
            for file_name in _model_table_plot_names(model_name):
                _remove_stale_plot(file_name, results_dir=results_dir)
        else:
            for file_name in [
                f"{model_name}_global_pareto_front.png",
                f"{model_name}_global_efficient_frontier.png",
                f"{model_name}_global_risk_vs_skewness.png",
                f"{model_name}_filtered_pareto_front.png",
                f"{model_name}_filtered_efficient_frontier.png",
                f"{model_name}_filtered_risk_vs_skewness.png",
            ]:
                _remove_stale_plot(file_name, results_dir=results_dir)

        if not any_portfolios_nonempty:
            for file_name in _model_portfolio_plot_names(model_name):
                _remove_stale_plot(file_name, results_dir=results_dir)
    return generated


if __name__ == "__main__":
    output_paths = generate_plots_from_results(show=False)
    if not output_paths:
        print(f"No results tables found in {RESULTS_DIR}. Run scripts/run_tables.py first.")
    else:
        print("Saved plot files:")
        for path in output_paths:
            print(path)
