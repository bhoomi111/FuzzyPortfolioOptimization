import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.config import (
    CARDINALITY,
    GENERATIONS,
    LOWER_BOUND,
    MIN_EXPECTED_RETURN,
    MIN_SKEWNESS,
    POP_SIZE,
    REPRESENTATIVES,
    RUNS,
    UPPER_BOUND,
)
from src.clustering.kmedoids import k_medoids
from src.data.preprocessing import portfolio_returns
from src.fuzzy.triangular import CoherentTriangularFuzzy
from src.moments.mean import credibilistic_mean
from src.moments.skewness import skewness
from src.optimization.constraints import repair
from src.optimization.moga import MOGA
from src.optimization.pareto import non_dominated_sort


def run_model(
    returns,
    objective_function,
    cp_values,
    mp_values,
    n_assets,
    lower=None,
    upper=None,
    cardinality=CARDINALITY,
    pop_size=POP_SIZE,
    generations=GENERATIONS,
    runs=RUNS,
    representatives=REPRESENTATIVES,
    min_expected_return=MIN_EXPECTED_RETURN,
    min_skewness=MIN_SKEWNESS,
    verbose=True,
    detailed_logging=False,
    model_label="Model",
):
    all_solutions = []
    lower = np.full(n_assets, LOWER_BOUND) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(n_assets, UPPER_BOUND) if upper is None else np.asarray(upper, dtype=float)
    total_jobs = len(cp_values) * len(mp_values) * runs
    completed_jobs = 0
    def log(message):
        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

    def solution_signature(solution):
        fuzzy = solution["fuzzy"]
        rounded_weights = tuple(np.round(solution["weights"], 10))
        rounded_fuzzy = (
            round(float(fuzzy.b1), 10),
            round(float(fuzzy.b2), 10),
            round(float(fuzzy.b3), 10),
            round(float(fuzzy.k), 10),
        )
        return rounded_weights + rounded_fuzzy

    def full_evaluation(weights):
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)

        weights = repair(weights, lower, upper, cardinality)
        R = portfolio_returns(returns, weights)
        R = R[~np.isnan(R)]

        if len(R) < 10:
            return None

        try:
            fuzzy = CoherentTriangularFuzzy.fit_from_returns(R)
        except ValueError:
            return None

        return {
            "weights": weights,
            "fuzzy": fuzzy,
        }

    def objective_fn(weights):
        result = full_evaluation(weights)

        if result is None:
            return [1e6, 1e6, 1e6, 1e6]

        return objective_function(result["fuzzy"])

    progress_bar = tqdm(
        total=total_jobs * generations,
        desc=model_label,
        unit="gen",
        disable=not verbose,
        leave=True,
        dynamic_ncols=True,
    )

    for cp in cp_values:
        for mp in mp_values:
            for run in range(runs):
                completed_jobs += 1
                moga = MOGA(
                    pop_size=pop_size,
                    n_assets=n_assets,
                    lower=lower,
                    upper=upper,
                    k=cardinality,
                    cp=cp,
                    mp=mp,
                )

                def on_generation(generation_done, generation_total):
                    progress_bar.set_postfix_str(f"job={completed_jobs}/{total_jobs} cp={cp:.1f} mp={mp:.1f} run={run + 1}/{runs}")
                    progress_bar.update(1)

                pop = moga.run(objective_fn, generations=generations, progress_callback=on_generation)
                fitnesses = [objective_fn(p) for p in pop]
                fronts = non_dominated_sort(pop, fitnesses)
                pareto_indices = fronts[0] if fronts else range(len(pop))
                added_before = len(all_solutions)

                for idx in pareto_indices:
                    eval_result = full_evaluation(pop[idx])
                    if eval_result is None:
                        continue

                    e = credibilistic_mean(eval_result["fuzzy"])
                    s = skewness(eval_result["fuzzy"], e)

                    if e < min_expected_return or s < min_skewness:
                        continue

                    all_solutions.append({
                        "weights": eval_result["weights"],
                        "cp": cp,
                        "mp": mp,
                        "run": run + 1,
                        "fuzzy": eval_result["fuzzy"],
                    })

                if detailed_logging:
                    log(
                        f"{model_label} finished job {completed_jobs}/{total_jobs}; "
                        f"Pareto candidates kept after filters: {len(all_solutions) - added_before}"
                    )

    progress_bar.close()

    if not all_solutions:
        log("No feasible representative candidates survived the filtration criteria.")
        return []

    unique_solutions = []
    seen_signatures = set()
    for solution in all_solutions:
        signature = solution_signature(solution)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_solutions.append(solution)

    duplicate_count = len(all_solutions) - len(unique_solutions)
    if duplicate_count and detailed_logging:
        log(f"{model_label} removed {duplicate_count} duplicate Pareto candidates before clustering.")

    weights_only = np.array([x["weights"] for x in unique_solutions])
    if detailed_logging:
        log(f"{model_label} running k-medoids on {len(unique_solutions)} filtered Pareto candidates.")
    medoids = k_medoids(weights_only, k=min(representatives, len(unique_solutions)))
    total_pareto_solutions = len(unique_solutions)
    selected = []
    for medoid_idx in medoids:
        representative = dict(unique_solutions[medoid_idx])
        representative["center_position"] = medoid_idx + 1
        representative["total_pareto_solutions"] = total_pareto_solutions
        selected.append(representative)
    log(
        f"{model_label}: {len(selected)} representatives selected "
        f"from {total_pareto_solutions} unique Pareto candidates."
    )
    return selected
