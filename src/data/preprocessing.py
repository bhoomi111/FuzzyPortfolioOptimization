import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns using:
    r_t = (p_{t+1} - p_t) / p_t
    """
    return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    weights = weights / np.sum(weights)  # ensure budget constraint
    return returns.values @ weights