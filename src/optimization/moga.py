import numpy as np
from src.optimization.operators import crossover, mutation
from src.optimization.constraints import random_feasible_portfolio, repair
from src.optimization.pareto import non_dominated_sort
from src.optimization.crowding import crowding_distance


class MOGA:
    def __init__(self, pop_size, n_assets, lower, upper, k, cp, mp):
        self.pop_size = pop_size
        self.n_assets = n_assets
        self.lower = lower
        self.upper = upper
        self.k = k
        self.cp = cp   # crossover probability
        self.mp = mp   # mutation probability

    def initialize(self):
        return [
            random_feasible_portfolio(self.n_assets, self.lower, self.upper, self.k)
            for _ in range(self.pop_size)
        ]

    def evaluate(self, population, objective_fn):
        return [objective_fn(ind) for ind in population]

    def evolve(self, population):
        new_pop = []

        for _ in range(len(population)):
            # Select parents
            idx = np.random.randint(len(population), size=2)
            p1, p2 = population[idx[0]], population[idx[1]]

            # Apply crossover probability
            if np.random.rand() < self.cp:
                child = crossover(p1, p2, self.lower, self.upper)
            else:
                child = p1.copy()

            # Apply mutation probability
            child = mutation(child, rate=self.mp, lower=self.lower, upper=self.upper, k=self.k)

            # Repair constraints
            child = repair(child, self.lower, self.upper, self.k)

            new_pop.append(child)

        return new_pop

    def run(self, objective_fn, generations=50, progress_callback=None):
        pop = self.initialize()

        for generation_idx in range(generations):
            offspring = self.evolve(pop)

            combined = pop + offspring

            fitnesses = [objective_fn(ind) for ind in combined]

            pop = self.select_next_generation(combined, fitnesses, self.pop_size)

            if progress_callback is not None:
                progress_callback(generation_idx + 1, generations)

        return pop

    def select_next_generation(self, population, fitnesses, pop_size):
        fronts = non_dominated_sort(population, fitnesses)

        new_population = []

        for front in fronts:
            if len(new_population) + len(front) > pop_size:
                distances = crowding_distance(front, fitnesses)

                sorted_front = sorted(
                    zip(front, distances),
                    key=lambda x: x[1],
                    reverse=True
                )

                for idx, _ in sorted_front:
                    if len(new_population) < pop_size:
                        new_population.append(population[idx])
                    else:
                        break
                break
            else:
                for idx in front:
                    new_population.append(population[idx])

        return new_population
