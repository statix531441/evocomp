"""
Visualization functions for evolutionary simulation results.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_over_time(logger, save_path=None):
    """
    Plot mean and max fitness over generations.
    """
    generations = logger.get_metric_series('generation')
    mean_fitness = logger.get_metric_series('mean_fitness')
    max_fitness = logger.get_metric_series('max_fitness')
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_fitness, label='Mean Fitness')
    plt.plot(generations, max_fitness, label='Max Fitness', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness over Time - {logger.experiment_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_entropy_over_time(logger, save_path=None):
    """
    Plot population entropy over time.
    """
    generations = logger.get_metric_series('generation')
    entropy = logger.get_metric_series('entropy')
    
    # Assuming genome size is constant, we can estimate max entropy
    # But we don't have genome size here easily unless we pass it or infer it.
    # Let's just plot the entropy.
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, entropy, label='Entropy', color='purple')
    plt.xlabel('Generation')
    plt.ylabel('Entropy (bits)')
    plt.title(f'Population Entropy - {logger.experiment_name}')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_mutual_information(logger, save_path=None):
    """
    Plot mutual information trajectory.
    """
    generations = logger.get_metric_series('generation')
    mi = logger.get_metric_series('mutual_information')
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mi, label='Mutual Information', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Mutual Information (bits)')
    plt.title(f'Mutual Information with Environment - {logger.experiment_name}')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_population_dynamics(loggers, labels, save_path=None):
    """
    Plot population sizes over time for multiple populations.
    """
    plt.figure(figsize=(10, 6))
    
    for logger, label in zip(loggers, labels):
        generations = logger.get_metric_series('generation')
        pop_size = logger.get_metric_series('population_size')
        plt.plot(generations, pop_size, label=label)
        
    plt.xlabel('Generation')
    plt.ylabel('Population Size')
    plt.title('Population Dynamics')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_travel_gene_frequency(logger, save_path=None):
    """
    Plot travel gene frequency over generations.
    """
    generations = logger.get_metric_series('generation')
    travel_freq = logger.get_metric_series('travel_gene_freq')
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, travel_freq, label='Travel Gene Freq', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Frequency')
    plt.title(f'Travel Gene Frequency - {logger.experiment_name}')
    plt.ylim(0, 1)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_summary_figure(logger, save_path=None):
    """
    Create a 2x2 subplot figure showing key metrics.
    """
    generations = logger.get_metric_series('generation')
    mean_fitness = logger.get_metric_series('mean_fitness')
    entropy = logger.get_metric_series('entropy')
    mi = logger.get_metric_series('mutual_information')
    travel_freq = logger.get_metric_series('travel_gene_freq')
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment Summary - {logger.experiment_name}')
    
    # Fitness
    axs[0, 0].plot(generations, mean_fitness, 'b-')
    axs[0, 0].set_title('Mean Fitness')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].grid(True)
    
    # Entropy
    axs[0, 1].plot(generations, entropy, 'm-')
    axs[0, 1].set_title('Entropy')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].grid(True)
    
    # Mutual Information
    axs[1, 0].plot(generations, mi, 'g-')
    axs[1, 0].set_title('Mutual Information')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].grid(True)
    
    # Travel Gene
    axs[1, 1].plot(generations, travel_freq, 'orange')
    axs[1, 1].set_title('Travel Gene Frequency')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
