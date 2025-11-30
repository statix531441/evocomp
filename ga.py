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
        self.targets = np.array([self.active_function(self.inputs)])

    def update(self):
        """If dynamic, possibly rotate functions and change inputs."""
        if rng.random() < self.dynamicity:
            self.active_function = self.functions[rng.integers(len(self.functions))]
            self.inputs = self.sample_inputs()
        self.compute_targets()

    def get_information_content(self):
        """
        Estimate the 'information content' of the environment.
        Based on the number of available functions (complexity).
        """
        if not self.functions:
            return 0.0
        return np.log2(len(self.functions))


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
    def tournament_selection(self, fitnesses, k=2, elitism_rate=0.1):
        # Select self.N individuals to maintain population size
        # We sample with replacement to allow best individuals to be selected multiple times
        # Added elitism to preserve top performers and prevent extinction
        
        current_size = self.genome.shape[0]
        
        # Elitism: automatically keep top performers
        n_elite = max(2, int(self.N * elitism_rate))  # At least 2 elites
        elite_indices = np.argsort(fitnesses)[-n_elite:]  # Top n_elite individuals
        
        # Tournament selection for remaining slots
        new_genome_indices = list(elite_indices)
        
        for _ in range(self.N - n_elite):
            # Select k random individuals
            competitors_idx = rng.integers(0, current_size, size=k)
            # Find the best among them
            winner_idx = competitors_idx[np.argmax(fitnesses[competitors_idx])]
            new_genome_indices.append(winner_idx)
            
        new_genome_indices = np.array(new_genome_indices)
        self.genome = self.genome[new_genome_indices]
        
        # Update metadata to match winners
        for key in self.metadata:
            self.metadata[key] = self.metadata[key][new_genome_indices]

    def one_point_crossover(self, p_c=0.8):
        N, genome_size = self.genome.shape
        if N < 2: return
        
        idx = np.arange(N)
        rng.shuffle(idx)
        new_population = []
        
        for i in range(0, N-1, 2):
            if rng.random() < p_c:
                point = rng.integers(1, genome_size)
                p1, p2 = self.genome[idx[i]].copy(), self.genome[idx[i+1]].copy()
                
                # Create offspring
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
                
                new_population.append(c1)
                new_population.append(c2)
        
        if new_population:
            self.genome = np.vstack((self.genome, np.array(new_population)))
            self.resize_metadata()

    def mutate(self, p_m=0.01):
        mutation_mask = rng.random(self.genome.shape) < p_m
        self.genome[mutation_mask] = 1 - self.genome[mutation_mask]

    # --- Population methods ---
    def sync_metadata(self):
        """Ensure travel gene matches genome."""
        if len(self.genome) != len(self.metadata['travel']):
            self.resize_metadata()
        self.metadata['travel'] = self.genome[:, 0]

    def resize_metadata(self):
        """Ensure all metadata arrays match the current genome size."""
        current_N = len(self.genome)
        old_N = len(self.metadata['travel'])
        
        if current_N == old_N:
            return

        if current_N > old_N:
            # Expand (happens after crossover)
            diff = current_N - old_N
            self.metadata['travel'] = np.concatenate([self.metadata['travel'], np.zeros(diff, dtype=int)])
            self.metadata['num_travelled'] = np.concatenate([self.metadata['num_travelled'], np.zeros(diff, dtype=int)])
            current_env_id = self.metadata['env_id'][0] if old_N > 0 else -1
            self.metadata['env_id'] = np.concatenate([self.metadata['env_id'], np.full(diff, current_env_id, dtype=int)])
        elif current_N < old_N:
             # Should be handled by selection logic directly updating metadata, but as a fallback:
             for key in self.metadata:
                 self.metadata[key] = self.metadata[key][:current_N]

    def compute_outputs(self, inputs):
        """Outputs = sum over genome traits weighted by inputs^power (excluding travel gene)."""
        powers = np.arange(self.genome_size - 1)
        return self.genome[:, 1:] @ (inputs ** powers)

    def compute_fitness(self, targets, inputs):
        outputs = self.compute_outputs(inputs)
        targets = np.atleast_2d(targets)
        return -np.abs(outputs - targets.flatten()[0])

    def migrate_to(self, other_population, p_migrate=0.3, min_population=30):
        """
        Move individuals with travel=1 to other population with probability p_migrate.
        Ensures at least min_population individuals remain to prevent extinction.
        """
        self.sync_metadata()
        travel_mask = self.metadata['travel'] == 1
        if not np.any(travel_mask):
            return

        # Apply probabilistic migration among travel=1 individuals
        indices = np.where(travel_mask)[0]
        migrating_indices = []
        for idx in indices:
            if rng.random() < p_migrate:
                migrating_indices.append(idx)
        
        if not migrating_indices:
            return

        migrating_indices = np.array(migrating_indices)
        
        # Prevent population extinction - keep minimum population size
        current_pop_size = len(self.genome)
        max_migrants = max(0, current_pop_size - min_population)
        if len(migrating_indices) > max_migrants:
            # Randomly select subset of migrants to preserve minimum population
            rng.shuffle(migrating_indices)
            migrating_indices = migrating_indices[:max_migrants]
        
        if len(migrating_indices) == 0:
            return
        
        # Select genomes and metadata to move
        moving_genomes = self.genome[migrating_indices]
        moving_metadata = {k: v[migrating_indices].copy() for k, v in self.metadata.items()}

        # Remove from current population
        keep_mask = np.ones(len(self.genome), dtype=bool)
        keep_mask[migrating_indices] = False
        
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
        other_population.metadata['num_travelled'][-moved_n:] += 1
        other_population.sync_metadata()

    def get_stats(self):
        """Return dictionary with current population statistics."""
        if len(self.genome) == 0:
            return {
                'size': 0,
                'mean_genome': np.zeros(self.genome_size),
                'travel_freq': 0.0
            }
        return {
            'size': len(self.genome),
            'mean_genome': np.mean(self.genome, axis=0),
            'travel_freq': np.mean(self.genome[:, 0])
        }
        

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
