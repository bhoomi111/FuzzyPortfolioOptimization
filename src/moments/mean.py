from src.models.mean import credibilistic_mean
def credibilistic_mean(fuzzy):
    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k
    alpha = b2 - b1
    beta = b3 - b2

    return b2 + 0.5 * ((beta - k * alpha) / (k + 1))