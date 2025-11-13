import numpy as np
import itertools

rng = np.random.default_rng()

# -------------------------------
# Environment
# -------------------------------
class Environment:
    _id_iter = itertools.count()

    def __init__(self, functions, dynamicity=0.0, M=5):
        """
        functions: list of functions that take an array of length M and return a scalar
        dynamicity: probability of rotating/changing the functions each generation
        """
        self.id = next(Environment._id_iter)
        self.M = M
        self.dynamicity = dynamicity
        self.functions = functions
        self.active_function = self.functions[0]
        self.inputs = self.sample_inputs()
        self.compute_targets()

    def sample_inputs(self):
        """Random inputs for genome -> output mapping."""
        return rng.uniform(3, 5, size=self.M)

    def compute_targets(self):
        """Compute target outputs for current inputs."""
        self.targets = np.array([f(self.inputs) for f in self.functions])

    def update(self):
        """If dynamic, possibly rotate functions and change inputs."""
        if rng.random() < self.dynamicity:
            self.active_function = self.functions[rng.integers(len(self.functions))]
            self.inputs = self.sample_inputs()
        self.compute_targets()


# -------------------------------
# Population
# -------------------------------
class Population:
    def __init__(self, N=20, genome_size=5, env_id=None):
        self.N = N
        self.genome_size = genome_size
        self.genome = rng.integers(0, 2, size=(N, genome_size))
        self.metadata = {
            'travel': self.genome[:, 0].copy(),
            'num_travelled': np.zeros(N, dtype=int),
            'env_id': np.full(N, env_id if env_id is not None else -1, dtype=int)
        }

    # --- GA methods ---
    def tournament_selection(self, fitnesses, k=2):
        N = self.genome.shape[0]
        idx = np.arange(N)
        rng.shuffle(idx)
        winners = []
        for i in range(0, N, k):
            competitors = idx[i:i+k]
            winner = competitors[np.argmax(fitnesses[competitors])]
            winners.append(winner)
        self.genome = self.genome[winners]

    def one_point_crossover(self, p_c=0.8):
        N, genome_size = self.genome.shape
        idx = np.arange(N)
        rng.shuffle(idx)
        new_population = self.genome.copy()
        for i in range(0, N-1, 2):
            if rng.random() < p_c:
                point = rng.integers(1, genome_size)
                p1, p2 = new_population[idx[i]].copy(), new_population[idx[i+1]].copy()
                new_population[idx[i], point:] = p2[point:]
                new_population[idx[i+1], point:] = p1[point:]
        self.genome = np.vstack((self.genome, new_population))

    def mutate(self, p_m=0.01):
        mutation_mask = rng.random(self.genome.shape) < p_m
        self.genome[mutation_mask] = 1 - self.genome[mutation_mask]

    # --- Population methods ---
    def sync_metadata(self):
        """Ensure travel gene matches genome."""
        self.metadata['travel'] = self.genome[:, 0]

    def compute_outputs(self, inputs):
        """Outputs = sum over genome traits weighted by inputs^power (excluding travel gene)."""
        powers = np.arange(self.genome_size - 1)
        return self.genome[:, 1:] @ (inputs ** powers)

    def compute_fitness(self, targets, inputs):
        outputs = self.compute_outputs(inputs)
        targets = np.atleast_2d(targets)
        # return -np.mean(np.abs(outputs[:, None] - targets), axis=1)
        return - np.abs(outputs - env1.targets)

    def migrate_to(self, other_population, p_migrate=0.3):
        """
        Move individuals with travel=1 to other population with probability p_migrate.
        """
        self.sync_metadata()
        travel_mask = self.metadata['travel'] == 1
        if not np.any(travel_mask):
            return

        # Apply probabilistic migration among travel=1 individuals
        migrate_mask = travel_mask.copy()
        migrate_mask[travel_mask] = rng.random(np.sum(travel_mask)) < p_migrate

        if not np.any(migrate_mask):
            return  # no one migrated this round

        # Select genomes and metadata to move
        moving_genomes = self.genome[migrate_mask]
        moving_metadata = {k: v[migrate_mask].copy() for k, v in self.metadata.items()}

        # Remove from current population
        keep_mask = ~migrate_mask
        self.genome = self.genome[keep_mask]
        self.metadata = {k: v[keep_mask].copy() for k, v in self.metadata.items()}

        # Add to other population
        other_population.genome = np.vstack([other_population.genome, moving_genomes])
        for k, v in moving_metadata.items():
            if k in other_population.metadata:
                other_population.metadata[k] = np.concatenate([other_population.metadata[k], v])
            else:
                other_population.metadata[k] = v

        # Update environment and travel stats for the moved individuals
        moved_n = len(moving_genomes)
        other_population.metadata['env_id'][-moved_n:] = other_population.metadata['env_id'][-moved_n:]
        other_population.metadata['num_travelled'][-moved_n:] += 1
        other_population.sync_metadata()



# -------------------------------
# Simulation
# -------------------------------
def run_generation(populations, environments, p_c=0.8, p_m=0.05):
    """
    Run GA for each population in its environment independently,
    then migrate traveling individuals between populations.
    """
    # GA per population
    for pop, env in zip(populations, environments):
        fitnesses = pop.compute_fitness(env.targets, env.inputs)
        pop.tournament_selection(fitnesses)
        pop.one_point_crossover(p_c)
        pop.mutate(p_m)
        pop.sync_metadata()

    # Migration between populations
    if len(populations) == 2:
        populations[0].migrate_to(populations[1])
        populations[1].migrate_to(populations[0])

    # Update environments if dynamic
    for env in environments:
        env.update()


# -------------------------------
# Example Experiment
# -------------------------------
if __name__ == "__main__":
    # Define two different functions for the two islands
    def f1(x): return np.sum(x**2)
    def f2(x): return np.prod(x)

    env1 = Environment([f1], dynamicity=0.0)
    env2 = Environment([f2], dynamicity=0.0)

    pop1 = Population(N=20, genome_size=5, env_id=env1.id)
    pop2 = Population(N=20, genome_size=5, env_id=env2.id)

    generations = 20
    for g in range(generations):
        print(f"\n--- Generation {g+1} ---")
        run_generation([pop1, pop2], [env1, env2])

        # Track travel genes and population sizes
        t1 = np.sum(pop1.metadata['travel'])
        t2 = np.sum(pop2.metadata['travel'])
        print(f"Pop1 size: {len(pop1.genome)}, travel sum: {t1}")
        print(f"Pop2 size: {len(pop2.genome)}, travel sum: {t2}")
        print(f"Pop1 num_travelled: {pop1.metadata['num_travelled'].sum()}, Pop2 num_travelled: {pop2.metadata['num_travelled'].sum()}")
