import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from tqdm import tqdm

from src.config import (
    CARDINALITY,
    GENERATIONS,
    LOWER_BOUND,
    MIN_EXPECTED_RETURN,
    MIN_SKEWNESS,
    MOMENTUM_LOOKBACK,
    MOMENTUM_WEIGHT,
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
from src.optimization.nsga2 import NSGAII
from src.optimization.pareto import non_dominated_sort


_WORKER_RETURNS = None
_WORKER_OBJECTIVE_FUNCTION = None
_WORKER_LOWER = None
_WORKER_UPPER = None
_WORKER_CARDINALITY = None


def _init_worker(returns, objective_function, lower, upper, cardinality):
    global _WORKER_RETURNS, _WORKER_OBJECTIVE_FUNCTION, _WORKER_LOWER, _WORKER_UPPER, _WORKER_CARDINALITY
    _WORKER_RETURNS = returns
    _WORKER_OBJECTIVE_FUNCTION = objective_function
    _WORKER_LOWER = lower
    _WORKER_UPPER = upper
    _WORKER_CARDINALITY = cardinality


def _job_seed(model_label: str, cp: float, mp: float, run: int) -> int:
    payload = f"{model_label}|{cp:.12g}|{mp:.12g}|{run}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def _run_single_job(job_payload):
    (
        n_assets,
        pop_size,
        generations,
        cp,
        mp,
        run,
        model_label,
        momentum_weight,
        momentum_lookback,
        fuzzy_fit_method,
        fuzzy_window,
        fuzzy_quantiles,
        fuzzy_min_periods,
        optimizer,
        crossover_mode,
    ) = job_payload

    returns = _WORKER_RETURNS
    objective_function = _WORKER_OBJECTIVE_FUNCTION
    lower = _WORKER_LOWER
    upper = _WORKER_UPPER
    cardinality = _WORKER_CARDINALITY

    job_seed = _job_seed(model_label, cp, mp, run)
    np.random.seed(job_seed)

    evaluation_cache = {}
    objective_cache = {}

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
            fuzzy = CoherentTriangularFuzzy.fit_from_returns(
                R,
                method=fuzzy_fit_method,
                window=fuzzy_window,
                quantiles=fuzzy_quantiles,
                min_periods=fuzzy_min_periods,
            )
        except ValueError:
            return None

        recent_momentum = 0.0
        if momentum_weight and momentum_lookback > 0:
            recent_window = R[-momentum_lookback:]
            if len(recent_window) > 0:
                recent_momentum = float(np.mean(recent_window))

        result = {
            "weights": weights,
            "fuzzy": fuzzy,
            "recent_momentum": recent_momentum,
        }
        evaluation_cache[cache_key] = result
        return result

    def objective_fn(weights):
        result = full_evaluation(weights)

        if result is None:
            return [1e6, 1e6, 1e6, 1e6]

        cache_key = tuple(np.round(result["weights"], 12))
        cached = objective_cache.get(cache_key)
        if cached is not None:
            return cached

        objectives = list(objective_function(result["fuzzy"]))
        if any(not np.isfinite(value) for value in objectives):
            return [1e6, 1e6, 1e6, 1e6]
        if momentum_weight:
            objectives[0] -= momentum_weight * result.get("recent_momentum", 0.0)
        objective_cache[cache_key] = objectives
        return objectives

    optimizer_name = (optimizer or "moga").lower()
    if optimizer_name == "nsga2":
        solver = NSGAII(
            pop_size=pop_size,
            n_assets=n_assets,
            lower=lower,
            upper=upper,
            k=cardinality,
            cp=cp,
            mp=mp,
            crossover_mode=crossover_mode,
            random_state=job_seed,
        )
    else:
        solver = MOGA(
            pop_size=pop_size,
            n_assets=n_assets,
            lower=lower,
            upper=upper,
            k=cardinality,
            cp=cp,
            mp=mp,
        )

    pop = solver.run(objective_fn, generations=generations)
    fitnesses = [objective_fn(p) for p in pop]
    fronts = non_dominated_sort(pop, fitnesses)
    pareto_indices = fronts[0] if fronts else range(len(pop))

    all_solutions = []
    for idx in pareto_indices:
        eval_result = full_evaluation(pop[idx])
        if eval_result is None:
            continue

        all_solutions.append(
            {
                "weights": eval_result["weights"],
                "cp": cp,
                "mp": mp,
                "run": run + 1,
                "fuzzy": eval_result["fuzzy"],
            }
        )

    return {
        "cp": cp,
        "mp": mp,
        "run": run,
        "solutions": all_solutions,
        "pareto_count": len(pareto_indices),
    }


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
    parallel_jobs=True,
    max_workers=None,
    momentum_weight: float = MOMENTUM_WEIGHT,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    fuzzy_fit_method: str = "static",
    fuzzy_window: int = 30,
    fuzzy_quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
    fuzzy_min_periods: int | None = None,
    optimizer: str = "moga",
    crossover_mode: str = "sbx",
):
    all_solutions = []
    lower = np.full(n_assets, LOWER_BOUND) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(n_assets, UPPER_BOUND) if upper is None else np.asarray(upper, dtype=float)

    job_payloads = [
        (
            n_assets,
            pop_size,
            generations,
            cp,
            mp,
            run,
            model_label,
            momentum_weight,
            momentum_lookback,
            fuzzy_fit_method,
            fuzzy_window,
            fuzzy_quantiles,
            fuzzy_min_periods,
            optimizer,
            crossover_mode,
        )
        for cp in cp_values
        for mp in mp_values
        for run in range(runs)
    ]
    total_jobs = len(job_payloads)

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

    progress_bar = tqdm(
        total=total_jobs,
        desc=model_label,
        unit="job",
        disable=not verbose,
        leave=True,
    )

    if parallel_jobs and total_jobs > 1:
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)
        worker_count = max(1, min(worker_count, total_jobs))

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(returns, objective_function, lower, upper, cardinality),
        ) as executor:
            for job_result in executor.map(_run_single_job, job_payloads, chunksize=1):
                all_solutions.extend(job_result["solutions"])
                progress_bar.set_postfix_str(
                    f"job={job_result['run'] + 1}/{runs} cp={job_result['cp']:.1f} mp={job_result['mp']:.1f}"
                )
                progress_bar.update(1)

                if detailed_logging:
                    log(
                        f"{model_label} finished job cp={job_result['cp']:.1f} mp={job_result['mp']:.1f} run={job_result['run'] + 1}/{runs}; "
                        f"Pareto candidates collected from this run: {job_result['pareto_count']}"
                    )
    else:
        _init_worker(returns, objective_function, lower, upper, cardinality)
        for job_payload in job_payloads:
            job_result = _run_single_job(job_payload)
            all_solutions.extend(job_result["solutions"])
            progress_bar.set_postfix_str(
                f"job={job_result['run'] + 1}/{runs} cp={job_result['cp']:.1f} mp={job_result['mp']:.1f}"
            )
            progress_bar.update(1)

            if detailed_logging:
                log(
                    f"{model_label} finished job cp={job_result['cp']:.1f} mp={job_result['mp']:.1f} run={job_result['run'] + 1}/{runs}; "
                    f"Pareto candidates collected from this run: {job_result['pareto_count']}"
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
