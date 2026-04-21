from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.preprocessing import portfolio_returns
from src.fuzzy.triangular import CoherentTriangularFuzzy

from src.models.objectives import compute_objectives, fitness_vector
from src.optimization.moga import MOGA
from src.clustering.kmedoids import k_medoids


# ✅ 1. Load RETURNS (paper uses returns directly)
returns = pd.read_csv("data/raw/monthly_returns.csv")

returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date")
returns = returns.astype(float)


# ✅ 2. Setup
n_assets = returns.shape[1]


# 🔥 FULL OBJECTIVE FUNCTION
def objective_fn(weights):
    # Ensure valid portfolio
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)

    weights = weights / np.sum(weights)

    R = portfolio_returns(returns, weights)
    R = R[~np.isnan(R)]

    # Safety
    if len(R) < 10:
        return [1e6, 1e6, 1e6, 1e6]

    fuzzy = CoherentTriangularFuzzy.fit_from_returns(R)

    obj = compute_objectives(fuzzy)

    return fitness_vector(obj)


# 🔥 PARAMETER GRID (from paper idea)
cp_values = [0.6, 0.7, 0.8]
mp_values = [0.2, 0.3, 0.4]

all_solutions = []


# 🔥 RUN MULTIPLE EXPERIMENTS
for cp in cp_values:
    for mp in mp_values:

        print(f"\nRunning MOGA with cp={cp}, mp={mp}")

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
            all_solutions.append((p, cp, mp))


# 🔥 EXTRACT REPRESENTATIVE SOLUTIONS (R1–R25)
weights_only = np.array([x[0] for x in all_solutions])

medoids = k_medoids(weights_only, k=25)

representatives = [all_solutions[i] for i in medoids]


print("\n===== REPRESENTATIVE SOLUTIONS (R1–R25) =====")

for i, (w, cp, mp) in enumerate(representatives):
    print(f"R{i+1}: cp={cp}, mp={mp}")