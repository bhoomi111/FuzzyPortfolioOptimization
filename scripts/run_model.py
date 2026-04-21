import numpy as np
from src.fuzzy.triangular import CoherentTriangularFuzzy
from src.data.preprocessing import portfolio_returns
from src.optimization.moga import MOGA
from src.clustering.kmedoids import k_medoids


def run_model(returns, objective_function, cp_values, mp_values, n_assets):
    all_solutions = []

    def full_evaluation(weights):
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)

        weights = weights / np.sum(weights)

        R = portfolio_returns(returns, weights)
        R = R[~np.isnan(R)]

        if len(R) < 10:
            return None

        fuzzy = CoherentTriangularFuzzy.fit_from_returns(R)

        return {
            "weights": weights,
            "fuzzy": fuzzy
        }

    def objective_fn(weights):
        result = full_evaluation(weights)

        if result is None:
            return [1e6, 1e6, 1e6, 1e6]

        return objective_function(result["fuzzy"])

    # 🔥 Run GA
    for cp in cp_values:
        for mp in mp_values:

            moga = MOGA(
                pop_size=50,
                n_assets=n_assets,
                lower=np.zeros(n_assets),
                upper=np.ones(n_assets),
                k=5,
                cp=cp,
                mp=mp
            )

            pop = moga.run(objective_fn, generations=50)

            for p in pop:
                eval_result = full_evaluation(p)
                if eval_result is not None:
                    all_solutions.append({
                        "weights": p,
                        "cp": cp,
                        "mp": mp,
                        "fuzzy": eval_result["fuzzy"]
                    })

    # 🔥 K-medoids
    weights_only = np.array([x["weights"] for x in all_solutions])
    medoids = k_medoids(weights_only, k=25)

    representatives = [all_solutions[i] for i in medoids]

    return representatives