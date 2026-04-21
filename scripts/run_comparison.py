from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.model1 import model1_objectives
from src.models.model2 import model2_objectives
from src.models.model3 import model3_objectives
from scripts.run_model import run_model


# Load returns
returns = pd.read_csv("data/raw/monthly_returns.csv")
returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date")
returns = returns.astype(float)

n_assets = returns.shape[1]

cp_values = [0.6, 0.7, 0.8]
mp_values = [0.2, 0.3, 0.4]


print("\nRunning Model I...")
model1_results = run_model(returns, model1_objectives, cp_values, mp_values, n_assets)

print("\nRunning Model II...")
model2_results = run_model(returns, model2_objectives, cp_values, mp_values, n_assets)

print("\nRunning Model III...")
model3_results = run_model(returns, model3_objectives, cp_values, mp_values, n_assets)


print("\n================ COMPARISON TABLE ================")
print(f"{'Rep':<5} | {'Model I (cp,mp)':<20} | {'Model II (cp,mp)':<20} | {'Model III (cp,mp)':<20}")
print("-" * 75)

for i in range(25):
    r1 = model1_results[i]
    r2 = model2_results[i]
    r3 = model3_results[i]

    print(f"R{i+1:<3} | ({r1[1]:.1f},{r1[2]:.1f})        | ({r2[1]:.1f},{r2[2]:.1f})        | ({r3[1]:.1f},{r3[2]:.1f})")