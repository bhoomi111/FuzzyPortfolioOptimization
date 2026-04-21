import numpy as np


def euclidean(a, b):
    return np.linalg.norm(a - b)


def k_medoids(X, k, max_iter=100):
    n = len(X)

    medoids = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        for i in range(n):
            distances = [euclidean(X[i], X[m]) for m in medoids]
            cluster = np.argmin(distances)
            clusters[cluster].append(i)

        new_medoids = []

        for cluster in clusters:
            if not cluster:
                continue

            costs = []
            for i in cluster:
                cost = sum(euclidean(X[i], X[j]) for j in cluster)
                costs.append((cost, i))

            new_medoids.append(min(costs)[1])

        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return medoids