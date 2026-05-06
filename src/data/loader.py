import pandas as pd

from src.config import MONTHLY_RETURNS_FILE, RAW_DATA_DIR


def load_returns(path=MONTHLY_RETURNS_FILE):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df


def load_prices(path=RAW_DATA_DIR / "monthly_prices.csv"):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df
