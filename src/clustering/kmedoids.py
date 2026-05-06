import numpy as np


def euclidean(a, b):
    return np.linalg.norm(a - b)


def k_medoids(X, k, max_iter=100):
    X = np.asarray(X, dtype=float)
    n = len(X)

    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = euclidean(X[i], X[j])

    denominators = distances.sum(axis=1)
    denominators[denominators == 0] = 1.0
    scores = (distances / denominators[:, None]).sum(axis=0)
    medoids = np.argsort(scores)[:k]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        for i in range(n):
            distances = [euclidean(X[i], X[m]) for m in medoids]
            cluster = np.argmin(distances)
            clusters[cluster].append(i)

        new_medoids = []

        for cluster_idx, cluster in enumerate(clusters):
            if not cluster:
                new_medoids.append(medoids[cluster_idx])
                continue

            costs = []
            for i in cluster:
                cost = sum(euclidean(X[i], X[j]) for j in cluster)
                costs.append((cost, i))

            new_medoids.append(min(costs)[1])

        if np.array_equal(medoids, new_medoids):
            break

        medoids = np.array(new_medoids)

    return list(medoids)
