import hashlib

import numpy as np

from src.fuzzy.rolling_quantiles import latest_rolling_triangular_fuzzy


class CoherentTriangularFuzzy:
    def __init__(self, b1, b2, b3, k):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.k = k

    @staticmethod
    def fit_from_returns(R, method: str = "static", window: int = 30, quantiles=(0.1, 0.5, 0.9), min_periods: int | None = None):
        """
        Fit a coherent triangular fuzzy number from historical returns.

        Parameters
        ----------
        R:
            One-dimensional historical return series.
        method:
            "static" keeps the original percentile-based fit.
            "rolling" derives the fuzzy number from the latest rolling window
            using quantiles.
        window:
            Rolling window size used when method="rolling".
        quantiles:
            Quantiles used for the triangular support when method="rolling".
        min_periods:
            Optional minimum number of observations for the rolling window.
        """
        if method not in {"static", "rolling"}:
            raise ValueError("method must be either 'static' or 'rolling'")

        if method == "rolling":
            return CoherentTriangularFuzzy.fit_from_rolling_returns(
                R,
                window=window,
                quantiles=quantiles,
                min_periods=min_periods,
            )

        R = np.asarray(R, dtype=float)
        R = R[~np.isnan(R)]
        if len(R) == 0:
            raise ValueError("cannot fit fuzzy number from empty returns")

        Q = np.percentile(R, range(1, 101))

        b1 = min(np.min(R), Q[2])   # Q3
        b2 = Q[49]                 # Q50
        b3 = Q[96]                 # Q97

        if not b1 < b2 < b3:
            raise ValueError("fuzzy fit requires b1 < b2 < b3")

        # The paper samples a uniform variate to choose the branch used for k.
        # We keep that step, but derive it deterministically from the return
        # series so the same portfolio is scored consistently across evaluations.
        rounded = np.round(R, 12)
        digest = hashlib.blake2b(rounded.tobytes(), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        u = np.random.default_rng(seed).uniform()

        if u < 0.5:
            ratio = (b2 - Q[19]) / (b2 - b1)
        else:
            ratio = (Q[79] - b2) / (b3 - b2)

        ratio = np.clip(ratio, 1e-12, 1 - 1e-12)
        k = np.log(0.5) / np.log(ratio)

        return CoherentTriangularFuzzy(b1, b2, b3, k)

    @staticmethod
    def fit_from_rolling_returns(R, window: int = 30, quantiles=(0.1, 0.5, 0.9), min_periods: int | None = None):
        """Fit a fuzzy number from the latest rolling quantile window."""

        latest = latest_rolling_triangular_fuzzy(
            returns=R,
            window=window,
            quantiles=quantiles,
            min_periods=min_periods,
        )
        latest = np.asarray(latest, dtype=float).squeeze()

        R = np.asarray(R, dtype=float)
        R = R[~np.isnan(R)]
        if len(R) == 0:
            raise ValueError("cannot fit fuzzy number from empty returns")

        if latest.ndim != 1 or len(latest) != 3:
            raise ValueError("rolling fit requires a single triangular quantile triple")

        b1, b2, b3 = map(float, latest)

        if b1 > b2:
            b1 = b2
        if b2 > b3:
            b3 = b2

        rolling_window = R[-window:] if window > 0 else R
        rolling_window = rolling_window[~np.isnan(rolling_window)]

        if len(rolling_window) == 0:
            raise ValueError("cannot fit fuzzy number from empty rolling window")

        q20, q80 = np.percentile(rolling_window, [20, 80])

        rounded = np.round(rolling_window, 12)
        digest = hashlib.blake2b(rounded.tobytes(), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        u = np.random.default_rng(seed).uniform()

        if u < 0.5:
            denominator = b2 - b1
            ratio = (b2 - q20) / denominator if abs(denominator) > 1e-12 else 1.0
        else:
            denominator = b3 - b2
            ratio = (q80 - b2) / denominator if abs(denominator) > 1e-12 else 1.0

        ratio = np.clip(ratio, 1e-12, 1 - 1e-12)
        k = np.log(0.5) / np.log(ratio)

        return CoherentTriangularFuzzy(b1, b2, b3, k)
