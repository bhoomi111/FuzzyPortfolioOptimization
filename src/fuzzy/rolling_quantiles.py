from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _to_returns_frame(returns: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        return returns.astype(float)

    array = np.asarray(returns, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    columns = [f"asset_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def rolling_triangular_fuzzy_numbers(
    returns: np.ndarray | pd.DataFrame,
    window: int,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Convert historical returns into rolling triangular fuzzy numbers.

    Parameters
    ----------
    returns:
        Time series of asset returns with shape (time_steps, num_assets).
        Can be a pandas DataFrame or a numpy array.
    window:
        Rolling lookback window size.
    quantiles:
        Three quantiles used to define the triangular fuzzy number.
        The default is (Q10, Q50, Q90).
    min_periods:
        Minimum number of observations required inside each rolling window.
        Defaults to 1 so the beginning of the series is handled with partial windows.

    Returns
    -------
    np.ndarray
        Array with shape (time_steps, num_assets, 3), where the last axis is
        (a, b, c).
    """

    q_low, q_mid, q_high = tuple(quantiles)
    if not q_low <= q_mid <= q_high:
        raise ValueError("quantiles must satisfy q_low <= q_mid <= q_high")

    frame = _to_returns_frame(returns)
    rolling_kwargs = {"window": window, "min_periods": 1 if min_periods is None else min_periods}

    lower = frame.rolling(**rolling_kwargs).quantile(q_low).to_numpy(dtype=float)
    middle = frame.rolling(**rolling_kwargs).quantile(q_mid).to_numpy(dtype=float)
    upper = frame.rolling(**rolling_kwargs).quantile(q_high).to_numpy(dtype=float)

    fuzzy = np.stack([lower, middle, upper], axis=-1)

    if np.any(fuzzy[..., 0] > fuzzy[..., 1]) or np.any(fuzzy[..., 1] > fuzzy[..., 2]):
        raise ValueError("rolling quantiles must preserve a <= b <= c")

    return fuzzy


def latest_rolling_triangular_fuzzy(
    returns: np.ndarray | pd.DataFrame,
    window: int,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Return the latest rolling triangular fuzzy numbers.

    This is a convenience wrapper for callers that need a single fuzzy
    estimate after applying the rolling transform.
    """

    fuzzy = rolling_triangular_fuzzy_numbers(
        returns=returns,
        window=window,
        quantiles=quantiles,
        min_periods=min_periods,
    )

    valid_rows = np.where(~np.isnan(fuzzy[..., 1]).all(axis=1))[0]
    if len(valid_rows) == 0:
        raise ValueError("cannot derive a rolling fuzzy number from empty returns")

    return np.squeeze(fuzzy[valid_rows[-1]])
