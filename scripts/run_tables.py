from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "monthly_returns.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd

from src.models.model1 import model1_objectives
from src.models.model2 import model2_objectives
from src.models.model3 import model3_objectives

from scripts.run_model import run_model
from scripts.build_table import build_table


# Load data
returns = pd.read_csv(DATA_PATH)
returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date")
returns = returns.astype(float)

n_assets = returns.shape[1]

cp_values = [0.6, 0.7, 0.8]
mp_values = [0.2, 0.3, 0.4]


# 🔥 Run models
print("Running Model I...")
res1 = run_model(returns, model1_objectives, cp_values, mp_values, n_assets)

print("Running Model II...")
res2 = run_model(returns, model2_objectives, cp_values, mp_values, n_assets)

print("Running Model III...")
res3 = run_model(returns, model3_objectives, cp_values, mp_values, n_assets)


# 🔥 Build tables
table1 = build_table(res1, "model1")
table2 = build_table(res2, "model2")
table3 = build_table(res3, "model3")


# 🔥 Save results
table1.to_csv(RESULTS_DIR / "model1_table.csv", index=False)
table2.to_csv(RESULTS_DIR / "model2_table.csv", index=False)
table3.to_csv(RESULTS_DIR / "model3_table.csv", index=False)

def print_table(title, df):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Limit decimals for readability
    display_df = df.copy()

    for col in ["b1", "b2", "b3", "k", "Mean", "Risk", "Skewness", "Semikurtosis"]:
        display_df[col] = display_df[col].astype(float).map(lambda x: f"{x:.5e}")

    print(display_df.to_string(index=False))
print(f"\nTables saved in {RESULTS_DIR}")

print_table("MODEL I RESULTS", table1)
print_table("MODEL II RESULTS", table2)
print_table("MODEL III RESULTS", table3)

def save_portfolios(results, filename, columns):
    rows = []

    for i, r in enumerate(results):
        row = {"Rep": f"R{i+1}"}

        weights = r["weights"]

        for j, col in enumerate(columns):
            row[col] = weights[j]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

save_portfolios(res1, RESULTS_DIR / "model1_portfolios.csv", returns.columns)
save_portfolios(res2, RESULTS_DIR / "model2_portfolios.csv", returns.columns)
save_portfolios(res3, RESULTS_DIR / "model3_portfolios.csv", returns.columns)