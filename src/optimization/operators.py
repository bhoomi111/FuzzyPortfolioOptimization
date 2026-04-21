import numpy as np


def crossover(parent1, parent2):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child
def mutation(weights, rate=0.1):
    w = weights.copy()

    if np.random.rand() < rate:
        i, j = np.random.choice(len(w), 2, replace=False)
        w[i], w[j] = w[j], w[i]

    if np.random.rand() < rate:
        idx = np.random.randint(len(w))
        w[idx] += np.random.normal(0, 0.01)

    return w