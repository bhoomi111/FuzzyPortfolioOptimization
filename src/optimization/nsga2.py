import numpy as np

from src.optimization.constraints import random_feasible_portfolio, repair
from src.optimization.operators import crossover_pair, mutation


def dominates(a, b):
    """Return True when objective vector a Pareto-dominates b (minimization)."""
    return np.all(np.asarray(a) <= np.asarray(b)) and np.any(np.asarray(a) < np.asarray(b))


def non_dominated_sort(fitnesses):
    """Fast non-dominated sorting. Returns fronts and rank array."""
    n = len(fitnesses)
    domination_sets = [[] for _ in range(n)]
    dominated_counts = np.zeros(n, dtype=int)
    ranks = np.full(n, -1, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(fitnesses[p], fitnesses[q]):
                domination_sets[p].append(q)
                dominated_counts[q] += 1
            elif dominates(fitnesses[q], fitnesses[p]):
                domination_sets[q].append(p)
                dominated_counts[p] += 1

    first_front = np.where(dominated_counts == 0)[0].tolist()
    fronts[0] = first_front
    ranks[first_front] = 0

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                dominated_counts[q] -= 1
                if dominated_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1], ranks


def crowding_distance(front_indices, fitnesses):
    """Crowding distance with objective-wise normalization."""
    size = len(front_indices)
    if size == 0:
        return np.array([], dtype=float)
    if size <= 2:
        return np.full(size, np.inf, dtype=float)

    f = np.asarray([fitnesses[i] for i in front_indices], dtype=float)
    n_obj = f.shape[1]
    distances = np.zeros(size, dtype=float)

    for m in range(n_obj):
        order = np.argsort(f[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        f_min = f[order[0], m]
        f_max = f[order[-1], m]
        denom = f_max - f_min
        if denom <= 1e-12:
            continue

        normalized = (f[:, m] - f_min) / denom
        for i in range(1, size - 1):
            if np.isinf(distances[order[i]]):
                continue
            distances[order[i]] += normalized[order[i + 1]] - normalized[order[i - 1]]

    return distances


def binary_tournament_select(population, ranks, crowding, rng):
    """Binary tournament selection using (rank asc, crowding desc)."""
    i, j = rng.integers(0, len(population), size=2)

    if ranks[i] < ranks[j]:
        return population[i]
    if ranks[j] < ranks[i]:
        return population[j]
    if crowding[i] > crowding[j]:
        return population[i]
    if crowding[j] > crowding[i]:
        return population[j]

    return population[i] if rng.random() < 0.5 else population[j]


def convex_crossover(parent1, parent2, rng):
    """Simple convex crossover that preserves non-negativity and sum after repair."""
    alpha = rng.random()
    return alpha * parent1 + (1.0 - alpha) * parent2, alpha * parent2 + (1.0 - alpha) * parent1


def gaussian_mutation(child, sigma, rng):
    """Additive Gaussian mutation for portfolio vectors."""
    mutated = child + rng.normal(0.0, sigma, size=child.shape)
    return np.maximum(mutated, 0.0)


class NSGAII:
    """NSGA-II optimizer for constrained portfolio weights."""

    def __init__(
        self,
        pop_size,
        n_assets,
        lower,
        upper,
        k,
        cp,
        mp,
        eta=None,
        mutation_sigma=0.02,
        random_state=None,
        crossover_mode="sbx",
    ):
        self.pop_size = pop_size
        self.n_assets = n_assets
        self.lower = lower
        self.upper = upper
        self.k = k
        self.cp = cp
        self.mp = mp
        self.eta = eta
        self.mutation_sigma = mutation_sigma
        self.rng = np.random.default_rng(random_state)
        self.crossover_mode = crossover_mode

    def initialize(self):
        return [
            random_feasible_portfolio(self.n_assets, self.lower, self.upper, self.k)
            for _ in range(self.pop_size)
        ]

    def evaluate(self, population, objective_fn):
        return [objective_fn(ind) for ind in population]

    def _crossover(self, p1, p2):
        if self.crossover_mode == "convex":
            return convex_crossover(p1, p2, self.rng)
        return crossover_pair(p1, p2, self.lower, self.upper)

    def _mutate(self, child):
        if self.rng.random() < self.mp:
            child = gaussian_mutation(child, self.mutation_sigma, self.rng)
        child = mutation(child, rate=self.mp, lower=self.lower, upper=self.upper, k=self.k)
        child = repair(child, self.lower, self.upper, self.k)
        return child

    def _assign_rank_and_crowding(self, fitnesses):
        fronts, ranks = non_dominated_sort(fitnesses)
        crowding = np.zeros(len(fitnesses), dtype=float)
        for front in fronts:
            distances = crowding_distance(front, fitnesses)
            for local_idx, ind_idx in enumerate(front):
                crowding[ind_idx] = distances[local_idx]
        return fronts, ranks, crowding

    def _create_offspring(self, population, ranks, crowding):
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = binary_tournament_select(population, ranks, crowding, self.rng)
            p2 = binary_tournament_select(population, ranks, crowding, self.rng)

            if self.rng.random() < self.cp:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            offspring.append(self._mutate(c1))
            if len(offspring) < self.pop_size:
                offspring.append(self._mutate(c2))

        return offspring

    def _select_next_generation(self, combined, fitnesses):
        fronts, _, _ = self._assign_rank_and_crowding(fitnesses)
        next_population = []

        for front in fronts:
            if len(next_population) + len(front) <= self.pop_size:
                next_population.extend([combined[i] for i in front])
                continue

            distances = crowding_distance(front, fitnesses)
            order = np.argsort(-distances)
            remaining = self.pop_size - len(next_population)
            selected = [front[i] for i in order[:remaining]]
            next_population.extend([combined[i] for i in selected])
            break

        return next_population

    def run(self, objective_fn, generations=100, progress_callback=None):
        population = self.initialize()

        for generation_idx in range(generations):
            fitnesses = self.evaluate(population, objective_fn)
            _, ranks, crowding = self._assign_rank_and_crowding(fitnesses)
            offspring = self._create_offspring(population, ranks, crowding)

            combined = population + offspring
            combined_fitnesses = self.evaluate(combined, objective_fn)
            population = self._select_next_generation(combined, combined_fitnesses)

            if progress_callback is not None:
                progress_callback(generation_idx + 1, generations)

        return population
