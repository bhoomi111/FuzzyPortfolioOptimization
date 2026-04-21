import numpy as np


def enforce_bounds(weights, lower, upper):
    return np.clip(weights, lower, upper)


def enforce_budget(weights):
    total = np.sum(weights)
    if total == 0:
        return weights
    return weights / total


def enforce_cardinality(weights, k):
    """
    Keep top-k assets, zero out others
    """
    idx = np.argsort(weights)[-k:]
    new_w = np.zeros_like(weights)
    new_w[idx] = weights[idx]
    return new_w


def repair(weights, lower, upper, k):
    weights = enforce_bounds(weights, lower, upper)
    weights = enforce_cardinality(weights, k)
    weights = enforce_budget(weights)
    return weights