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
    evaluation_cache = {}
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
        cache_key = tuple(np.round(weights, 12))
        cached = evaluation_cache.get(cache_key)
        if cached is not None:
            return cached
        R = portfolio_returns(returns, weights)
        R = R[~np.isnan(R)]

        if len(R) < 10:
            return None

        try:
            fuzzy = CoherentTriangularFuzzy.fit_from_returns(R)
        except ValueError:
            return None

        result = {
            "weights": weights,
            "fuzzy": fuzzy,
        }
        evaluation_cache[cache_key] = result
        return result

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
                for idx in pareto_indices:
                    eval_result = full_evaluation(pop[idx])
                    if eval_result is None:
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
                        f"Pareto candidates collected from this run: {len(pareto_indices)}"
                    )

    progress_bar.close()

    if not all_solutions:
        log("No feasible representative candidates survived the filtration criteria.")
        return {
            "representatives": [],
            "global_pareto": [],
            "filtered_pareto": [],
        }

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

    global_fitnesses = [objective_function(solution["fuzzy"]) for solution in unique_solutions]
    global_fronts = non_dominated_sort(unique_solutions, global_fitnesses)
    global_pareto = [unique_solutions[idx] for idx in (global_fronts[0] if global_fronts else range(len(unique_solutions)))]

    filtered_solutions = []
    for solution in global_pareto:
        e = credibilistic_mean(solution["fuzzy"])
        s = skewness(solution["fuzzy"], e)

        if min_expected_return is not None and e < min_expected_return:
            continue

        if min_skewness is not None and s < min_skewness:
            continue

        filtered_solutions.append(solution)

    if not filtered_solutions:
        log("No globally non-dominated candidates survived the filtration criteria.")
        return {
            "representatives": [],
            "global_pareto": global_pareto,
            "filtered_pareto": [],
        }

    weights_only = np.array([x["weights"] for x in filtered_solutions])
    if detailed_logging:
        log(
            f"{model_label} global Pareto union size={len(global_pareto)}; "
            f"filtered size={len(filtered_solutions)}. Running k-medoids."
        )
    medoids = k_medoids(weights_only, k=min(representatives, len(filtered_solutions)))
    total_pareto_solutions = len(filtered_solutions)
    selected = []
    for medoid_idx in medoids:
        representative = dict(filtered_solutions[medoid_idx])
        representative["center_position"] = medoid_idx + 1
        representative["total_pareto_solutions"] = total_pareto_solutions
        selected.append(representative)
    log(
        f"{model_label}: {len(selected)} representatives selected "
        f"from {total_pareto_solutions} unique Pareto candidates."
    )
    return {
        "representatives": selected,
        "global_pareto": global_pareto,
        "filtered_pareto": filtered_solutions,
    }
