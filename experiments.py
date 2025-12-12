"""
Experiment configurations and runners for Phase 1 and Phase 2.
"""
import numpy as np
from ga import Environment, Population, run_generation
from data_logger import EvolutionLogger

# Define some standard functions for the environment
def f_sum_sq(x):
    return np.sum(x**2)

def f_prod(x):
    return np.prod(np.abs(x)) # Use abs to avoid complex numbers if inputs are negative (though inputs are 3-5)

def f_sum(x):
    return np.sum(x)

def f_max(x):
    return np.max(x)

def f_min(x):
    return np.min(x)

def get_functions_pool():
    return [f_sum_sq, f_prod, f_sum, f_max, f_min]

def run_phase1_static(generations=200, N=100, genome_size=8, p_c=0.7, p_m=0.05, shock_generation=150):
    """
    Run Phase 1 Static Experiment.
    """
    print(f"Starting Phase 1 Static (N={N}, G={generations}, Shock={shock_generation})")
    
    # Setup Environment with a single function initially
    # We'll use f_sum_sq as the primary, and f_prod as the shock
    functions = [f_sum_sq]
    env = Environment(functions, dynamicity=0.0, M=genome_size-1)
    
    pop = Population(N=N, genome_size=genome_size, env_id=env.id)
    logger = EvolutionLogger("Phase1_Static")
    
    shock_applied = False
    
    for g in range(generations):
        # Apply shock
        if g == shock_generation and not shock_applied:
            print(f"Applying environmental shock at generation {g}")
            # Change the active function to something else
            # We can modify the environment's functions list and active function
            env.functions = [f_prod]
            env.active_function = f_prod
            env.compute_targets() # Recompute targets for new function
            shock_applied = True
            
        run_generation([pop], [env], p_c=p_c, p_m=p_m)
        logger.log_generation(g, pop, env)
        
    return logger

def run_phase1_dynamic(generations=200, N=100, genome_size=8, p_c=0.7, p_m=0.05, dynamicity=0.1):
    """
    Run Phase 1 Dynamic Experiment.
    """
    print(f"Starting Phase 1 Dynamic (N={N}, G={generations}, Dynamicity={dynamicity})")
    
    functions = get_functions_pool()
    env = Environment(functions, dynamicity=dynamicity, M=genome_size-1)
    
    pop = Population(N=N, genome_size=genome_size, env_id=env.id)
    logger = EvolutionLogger("Phase1_Dynamic")
    
    for g in range(generations):
        run_generation([pop], [env], p_c=p_c, p_m=p_m)
        logger.log_generation(g, pop, env)
        
    return logger

def compare_phase1_results(static_logger, dynamic_logger):
    """
    Compare results from static and dynamic experiments.
    """
    print("\n--- Phase 1 Comparison ---")
    
    static_entropy = static_logger.get_metric_series('entropy')[-1]
    dynamic_entropy = dynamic_logger.get_metric_series('entropy')[-1]
    
    print(f"Final Entropy - Static: {static_entropy:.4f}, Dynamic: {dynamic_entropy:.4f}")
    
    # Check survival in static after shock
    # Assuming shock was at 150 and total is 200
    # We check if population size dropped or fitness collapsed
    # Since we don't have death in this GA (fixed N), we check fitness
    static_fitness = static_logger.get_metric_series('mean_fitness')
    shock_gen = 150 # Hardcoded assumption based on default, ideally passed or detected
    if len(static_fitness) > shock_gen + 10:
        pre_shock = np.mean(static_fitness[shock_gen-10:shock_gen])
        post_shock = np.mean(static_fitness[shock_gen:shock_gen+10])
        print(f"Static Fitness Pre-Shock: {pre_shock:.4f}, Post-Shock: {post_shock:.4f}")
        drop = (pre_shock - post_shock) / abs(pre_shock) if pre_shock != 0 else 0
        print(f"Fitness Drop: {drop*100:.1f}%")
    
    # Mutual Information
    static_mi = np.mean(static_logger.get_metric_series('mutual_information')[-20:])
    dynamic_mi = np.mean(dynamic_logger.get_metric_series('mutual_information')[-20:])
    print(f"Avg Final MI - Static: {static_mi:.4f}, Dynamic: {dynamic_mi:.4f}")
    
    return {
        'static_entropy': static_entropy,
        'dynamic_entropy': dynamic_entropy,
        'static_mi': static_mi,
        'dynamic_mi': dynamic_mi
    }

def run_phase2_two_islands(generations=300, N=100, genome_size=8, p_c=0.7, p_m=0.05, dynamicity1=0.0, dynamicity2=0.3, p_migrate=0.15):
    """
    Run Phase 2 Two Islands Experiment.
    """
    print(f"Starting Phase 2 Two Islands (N={N}, G={generations}, D1={dynamicity1}, D2={dynamicity2})")
    
    # Island 1: Static (or low dynamic)
    functions1 = [f_sum_sq] # Mostly static
    if dynamicity1 > 0: functions1 = get_functions_pool()
    env1 = Environment(functions1, dynamicity=dynamicity1, M=genome_size-1)
    
    # Island 2: Dynamic
    functions2 = get_functions_pool()
    env2 = Environment(functions2, dynamicity=dynamicity2, M=genome_size-1)
    
    pop1 = Population(N=N, genome_size=genome_size, env_id=env1.id)
    pop2 = Population(N=N, genome_size=genome_size, env_id=env2.id)
    
    logger1 = EvolutionLogger("Phase2_Island1")
    logger2 = EvolutionLogger("Phase2_Island2")
    
    for g in range(generations):
        
        # 1. Compute Fitness & Selection & Crossover & Mutation
        fitnesses1 = pop1.compute_fitness(env1.targets, env1.inputs)
        pop1.tournament_selection(fitnesses1)
        pop1.one_point_crossover(p_c)
        pop1.mutate(p_m)
        pop1.sync_metadata()
        
        fitnesses2 = pop2.compute_fitness(env2.targets, env2.inputs)
        pop2.tournament_selection(fitnesses2)
        pop2.one_point_crossover(p_c)
        pop2.mutate(p_m)
        pop2.sync_metadata()
        
        # 2. Migration
        pop1.migrate_to(pop2, p_migrate=p_migrate)
        pop2.migrate_to(pop1, p_migrate=p_migrate)
        
        # 3. Environment Update
        env1.update()
        env2.update()
        
        # Log
        logger1.log_generation(g, pop1, env1)
        logger2.log_generation(g, pop2, env2)
        
    return logger1, logger2

def analyze_phase2_travel_evolution(logger1, logger2):
    """
    Analyze travel gene evolution in Phase 2.
    """
    print("\n--- Phase 2 Analysis ---")
    
    travel1 = logger1.get_metric_series('travel_gene_freq')
    travel2 = logger2.get_metric_series('travel_gene_freq')
    
    print(f"Final Travel Freq - Island 1: {travel1[-1]:.4f}")
    print(f"Final Travel Freq - Island 2: {travel2[-1]:.4f}")
    
    # Check if dynamic environment promoted migration
    if travel2[-1] > travel1[-1]:
        print("Hypothesis Supported: Dynamic environment promoted higher migration.")
    else:
        print("Hypothesis Not Supported: Dynamic environment did not promote higher migration.")


def experiment_dynamicity_sweep(generations, output_dir):
    dynamicities = [0.0,  0.1, 0.5]
    population_size = 100
    mutation_rate = 0.05
    crossover_rate = 0.7
    genome_size = 10 
    functions = get_functions_pool()

    for d in dynamicities:
        exp_name = f"dynamicity_{d}"
        print(f"\n=== Running dynamicity={d} ===")

        env = Environment(functions, dynamicity=d, M=genome_size-1)
        pop = Population(N=population_size, genome_size=genome_size, env_id=env.id)
        logger = EvolutionLogger(exp_name)

        for gen in range(generations):
            run_generation([pop], [env], p_c=crossover_rate, p_m=mutation_rate)
            logger.log_generation(gen, pop, env)

        logger.save_csv(f"{output_dir}/{exp_name}.csv")
        print(f"Saved results for dynamicity={d}")

def experiment_mutation_sweep(generations, output_dir):
    mutation_rates = [0.001, 0.1, 0.2]
    population_size = 100
    crossover_rate = 0.7
    genome_size = 10
    functions = get_functions_pool()
    dynamicity = 0.1   # moderate environment

    for m in mutation_rates:
        exp_name = f"mutation_{m}"
        print(f"\n=== Running mutation_rate={m} ===")

        env = Environment(functions, dynamicity=dynamicity, M=genome_size-1)
        pop = Population(N=population_size, genome_size=genome_size, env_id=env.id)
        logger = EvolutionLogger(exp_name)

        for gen in range(generations):
            run_generation([pop], [env], p_c=crossover_rate, p_m=m)
            logger.log_generation(gen, pop, env)

        logger.save_csv(f"{output_dir}/{exp_name}.csv")
        print(f"Saved results for mutation={m}")
