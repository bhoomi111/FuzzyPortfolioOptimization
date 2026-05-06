from src.config import CVAR_CONFIDENCE


def cvar(fuzzy, p=CVAR_CONFIDENCE):
    """
    Credibilistic CVaR for A = (b1, b2, b3)_k.

    This is Proposition 5 in the paper.
    """
    if not 0 < p < 1:
        raise ValueError("confidence level p must be in (0, 1)")

    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k
    alpha = b2 - b1
    beta = b3 - b2

    if p < 0.5:
        numerator = (2 * p * (1 - (2 * p) ** k) + k * (2 * p - 1)) * alpha + beta
        denominator = 2 * (k + 1) * (1 - p)
        return b2 + numerator / denominator

    return alpha + beta - (k * beta * (2 * (1 - p)) ** (1 / k)) / (k + 1)

