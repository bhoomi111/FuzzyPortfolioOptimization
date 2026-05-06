import numpy as np


def _bounded_exponential_step(x, bound, diff, lamb, toward_lower, r):
    if diff <= 1e-12:
        return 0.0

    distance = x - bound if toward_lower else bound - x
    exponent = np.exp(-distance / (lamb * diff))

    if r <= 0.5:
        value = exponent + 2 * r * (1 - exponent)
        return lamb * np.log(max(value, 1e-12))

    value = 1 - (2 * r - 1) * (1 - exponent)
    return -lamb * np.log(max(value, 1e-12))


def crossover_pair(parent1, parent2, lower=None, upper=None, lamb=0.5):
    """
    CCBEX-style bounded exponential crossover from Appendix B.1.

    Returns both offspring described in the original pseudocode.
    """
    if lower is None or upper is None:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(parent1)):
        if parent1[i] <= 0 or parent2[i] <= 0:
            continue

        diff = abs(parent1[i] - parent2[i])
        if diff <= 1e-12:
            continue

        r = np.random.rand()
        step1 = _bounded_exponential_step(parent1[i], lower[i], diff, lamb, True, r)
        step2 = _bounded_exponential_step(parent2[i], lower[i], diff, lamb, True, r)

        if r > 0.5:
            step1 = _bounded_exponential_step(parent1[i], upper[i], diff, lamb, False, r)
            step2 = _bounded_exponential_step(parent2[i], upper[i], diff, lamb, False, r)

        child1[i] = parent1[i] + step1 * diff
        child2[i] = parent2[i] + step2 * diff

    return np.clip(child1, 0.0, upper), np.clip(child2, 0.0, upper)


def crossover(parent1, parent2, lower=None, upper=None, lamb=0.5):
    child1, child2 = crossover_pair(parent1, parent2, lower=lower, upper=upper, lamb=lamb)
    return child1 if np.random.rand() < 0.5 else child2


def swap_mutation(weights, lower, upper):
    w = weights.copy()
    active = np.flatnonzero(w > 0)
    inactive = np.flatnonzero(w == 0)

    if len(active) == 0 or len(inactive) == 0:
        return w

    i = np.random.choice(active)
    j = np.random.choice(inactive)
    denominator = upper[i] - lower[i]

    if denominator <= 0:
        return w

    w[j] = lower[j] + ((w[i] - lower[i]) / denominator) * (upper[j] - lower[j])
    w[i] = 0.0
    return w


def power_mutation(weights, lower, upper, power_index=10):
    w = weights.copy()
    active = np.flatnonzero(w > 0)

    if len(active) == 0:
        return w

    i = np.random.choice(active)
    span = upper[i] - lower[i]

    if span <= 0:
        return w

    rho = np.random.rand()
    sigma = np.random.rand()
    theta = (w[i] - lower[i]) / span
    chi = rho ** (1.0 / power_index)

    if theta < sigma:
        w[i] = w[i] - chi * (w[i] - lower[i])
    else:
        w[i] = w[i] + chi * (upper[i] - w[i])

    return w


def mutation(weights, rate=0.1, lower=None, upper=None, k=None):
    w = weights.copy()

    if lower is None or upper is None:
        if np.random.rand() >= rate:
            return w
        i, j = np.random.choice(len(w), 2, replace=False)
        w[i], w[j] = w[j], w[i]
        return w

    if np.random.rand() < rate:
        w = swap_mutation(w, lower, upper)

    if np.random.rand() < rate:
        w = power_mutation(w, lower, upper)

    return w
