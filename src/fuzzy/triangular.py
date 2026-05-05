import hashlib

import numpy as np


class CoherentTriangularFuzzy:
    def __init__(self, b1, b2, b3, k):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.k = k

    @staticmethod
    def fit_from_returns(R):
        """
        Percentile-based fitting (paper Section 4.1)
        """
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
