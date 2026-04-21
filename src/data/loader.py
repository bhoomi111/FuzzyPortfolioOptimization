import pandas as pd


def load_returns(path="data/raw/monthly_returns.csv"):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df


def load_prices(path="data/raw/monthly_prices.csv"):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df