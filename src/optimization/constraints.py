import numpy as np


def _as_bounds(lower, upper, n):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.ndim == 0:
        lower = np.full(n, float(lower))
    if upper.ndim == 0:
        upper = np.full(n, float(upper))

    return lower, upper


def enforce_bounds(weights, lower, upper):
    lower, upper = _as_bounds(lower, upper, len(weights))
    return np.clip(weights, lower, upper)


def enforce_cardinality(weights, k):
    """
    Keep top-k assets, zero out others
    """
    idx = np.argsort(weights)[-k:]
    new_w = np.zeros_like(weights)
    new_w[idx] = weights[idx]
    return new_w


def repair_budget(weights, lower, upper, active):
    """
    Paper Appendix B.4 repair mechanism for the capital budget.
    """
    repaired = np.zeros_like(weights, dtype=float)
    lower_active = lower[active]
    upper_active = upper[active]
    current = weights[active]

    if lower_active.sum() > 1 + 1e-12 or upper_active.sum() < 1 - 1e-12:
        raise ValueError("selected active assets cannot satisfy budget with given bounds")

    total = current.sum()

    if np.isclose(total, 1.0):
        repaired[active] = current
    elif total > 1:
        denominator = np.sum(current - lower_active)
        if denominator <= 1e-12:
            span = upper_active - lower_active
            repaired[active] = lower_active + span * ((1 - lower_active.sum()) / span.sum())
        else:
            repaired[active] = lower_active + (
                (current - lower_active) / denominator
            ) * (1 - lower_active.sum())
    else:
        denominator = np.sum(upper_active - current)
        if denominator <= 1e-12:
            span = upper_active - lower_active
            repaired[active] = lower_active + span * ((1 - lower_active.sum()) / span.sum())
        else:
            repaired[active] = upper_active - (
                (upper_active - current) / denominator
            ) * (upper_active.sum() - 1)

    repaired[active] = np.clip(repaired[active], lower_active, upper_active)
    repaired[active] /= repaired[active].sum()
    return repaired


def repair(weights, lower, upper, k):
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    weights = np.maximum(weights, 0.0)
    n = len(weights)
    lower, upper = _as_bounds(lower, upper, n)

    if k <= 0 or k > n:
        raise ValueError("cardinality k must be between 1 and number of assets")

    active = np.argsort(weights)[-k:]
    clipped = np.zeros_like(weights, dtype=float)
    clipped[active] = np.clip(weights[active], lower[active], upper[active])

    return repair_budget(clipped, lower, upper, active)


def random_feasible_portfolio(n_assets, lower, upper, k):
    lower, upper = _as_bounds(lower, upper, n_assets)
    active = np.random.choice(n_assets, k, replace=False)

    weights = np.zeros(n_assets)
    weights[active] = np.random.uniform(lower[active], upper[active])

    return repair_budget(weights, lower, upper, active)
