import numpy as np


def crowding_distance(front, fitnesses):
    n = len(front)
    distances = [0.0] * n

    num_objectives = len(fitnesses[0])

    for m in range(num_objectives):
        values = [fitnesses[i][m] for i in front]
        sorted_idx = np.argsort(values)

        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        min_val = values[sorted_idx[0]]
        max_val = values[sorted_idx[-1]]

        if max_val == min_val:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                values[sorted_idx[i + 1]] - values[sorted_idx[i - 1]]
            ) / (max_val - min_val)

    return distances