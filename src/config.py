from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FULL_REPLICATION_DIR = PROJECT_ROOT / "results_full_replication"

MONTHLY_RETURNS_FILE = RAW_DATA_DIR / "monthly_returns.csv"
BENCHMARK_RETURNS_PATH = RAW_DATA_DIR / "nifty50_monthly_returns.csv"

CARDINALITY = 5
LOWER_BOUND = 0.08
UPPER_BOUND = 0.30

CVAR_CONFIDENCE = 0.95

MIN_EXPECTED_RETURN = 0.02
MIN_SKEWNESS = 0.0

CP_VALUES = [0.6, 0.7, 0.8]
MP_VALUES = [0.2, 0.3, 0.4]

# Optimization settings used by the main runners.
# Change these values in one place and rerun the scripts.
POP_SIZE = 150
GENERATIONS = 1000
RUNS = 10
REPRESENTATIVES = 25

TEST_MONTHS = 24

BENCHMARK_COLUMN = "NIFTY50"
