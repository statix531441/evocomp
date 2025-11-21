"""
Data logging utilities for tracking evolutionary metrics across generations.
"""
import pandas as pd
import numpy as np
from information_metrics import compute_entropy, compute_mutual_information

class EvolutionLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.records = []

    def log_generation(self, generation, population, environment, additional_metrics=None):
        """
        Compute and store all metrics for a given generation.
        Handles empty populations gracefully.
        """
        # Handle empty population
        if len(population.genome) == 0:
            record = {
                'generation': generation,
                'population_size': 0,
                'mean_fitness': np.nan,
                'max_fitness': np.nan,
                'entropy': 0.0,
                'mutual_information': 0.0,
                'travel_gene_freq': 0.0,
                'num_migrations': 0
            }
            if additional_metrics:
                record.update(additional_metrics)
            self.records.append(record)
            return
        
        # Basic stats
        fitnesses = population.compute_fitness(environment.targets, environment.inputs)
        mean_fitness = np.mean(fitnesses) if len(fitnesses) > 0 else np.nan
        max_fitness = np.max(fitnesses) if len(fitnesses) > 0 else np.nan
        
        # Information metrics
        entropy = compute_entropy(population.genome)
        mutual_info = compute_mutual_information(population.genome, environment)
        
        # Travel gene frequency
        travel_freq = population.get_stats()['travel_freq']
        
        record = {
            'generation': generation,
            'population_size': len(population.genome),
            'mean_fitness': mean_fitness,
            'max_fitness': max_fitness,
            'entropy': entropy,
            'mutual_information': mutual_info,
            'travel_gene_freq': travel_freq,
            'num_migrations': np.sum(population.metadata['num_travelled']) if len(population.metadata['num_travelled']) > 0 else 0
        }
        
        if additional_metrics:
            record.update(additional_metrics)
            
        self.records.append(record)

    def to_dataframe(self):
        """Convert records list to a pandas DataFrame."""
        return pd.DataFrame(self.records)

    def save_csv(self, filepath):
        """Call to_dataframe and save to the specified path."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def get_metric_series(self, metric_name):
        """Return a list of values for the specified metric across all logged generations."""
        return [r.get(metric_name) for r in self.records]
