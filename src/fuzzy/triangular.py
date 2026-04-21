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
        Q = np.percentile(R, range(1, 101))

        b1 = min(np.min(R), Q[2])   # Q3
        b2 = Q[49]                 # Q50
        b3 = Q[96]                 # Q97

        u = np.random.uniform()

        if u < 0.5:
            k = np.log(0.5) / np.log((b2 - Q[19]) / (b2 - b1))
        else:
            k = np.log(0.5) / np.log((Q[79] - b2) / (b3 - b2))

        return CoherentTriangularFuzzy(b1, b2, b3, k)