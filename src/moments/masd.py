def masd(fuzzy):
    """
    Credibilistic MASD for A = (b1, b2, b3)_k.

    This is Proposition 4 in the paper.
    """
    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k
    alpha = b2 - b1
    beta = b3 - b2

    if k * alpha >= beta:
        base = 1 + (beta - k * alpha) / (2 * alpha * (k + 1))
        return (k * alpha / (2 * (k + 1))) * base ** ((k + 1) / k)

    base = 1 + (k * alpha - beta) / (2 * beta * (k + 1))
    return (beta / (2 * (k + 1))) * base ** (k + 1)

