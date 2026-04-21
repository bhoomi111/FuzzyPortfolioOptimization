import numpy as np


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(population, fitnesses):
    """
    Returns list of fronts: [[idx1, idx2], [...]]
    """
    S = [[] for _ in range(len(population))]
    n = [0] * len(population)
    rank = [0] * len(population)

    fronts = [[]]

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(fitnesses[p], fitnesses[q]):
                S[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return fronts[:-1]