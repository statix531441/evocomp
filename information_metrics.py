"""
Information-theoretic metrics for analyzing evolutionary dynamics.
"""
import numpy as np

def compute_entropy(population_genome):
    """
    Calculate Shannon entropy of the population genome.
    
    Input: A 2D numpy array of shape (N, genome_size) containing binary genomes.
    Output: A scalar representing the total entropy of the population.
    """
    if len(population_genome) == 0:
        return 0.0
        
    # Calculate frequency of 1s at each locus
    p = np.mean(population_genome, axis=0)
    
    # Calculate entropy at each locus: -p*log2(p) - (1-p)*log2(1-p)
    # Handle edge cases where p=0 or p=1 (log2(0) is undefined, but limit is 0)
    entropy_per_locus = np.zeros_like(p)
    
    # Mask for p not 0 and not 1
    mask = (p > 0) & (p < 1)
    
    if np.any(mask):
        p_valid = p[mask]
        entropy_per_locus[mask] = -p_valid * np.log2(p_valid) - (1 - p_valid) * np.log2(1 - p_valid)
        
    # Sum across all loci
    return np.sum(entropy_per_locus)

def compute_mutual_information(population_genome, environment, n_bins=10):
    """
    Estimate mutual information between genome configurations and environmental targets.
    
    Input: Population genome array, an Environment object, and number of bins.
    Output: A scalar representing mutual information in bits.
    """
    if len(population_genome) == 0:
        return 0.0

    # We use the Adami complexity approximation: MI = H_max - H_observed.
    # This represents the information stored in the genome about the environment (constraints).
    # H_max = genome_size (assuming binary loci with max entropy of 1 bit each).
    # H_observed = compute_entropy(population).
    
    max_entropy = population_genome.shape[1] # 1 bit per locus
    pop_entropy = compute_entropy(population_genome)
    return max_entropy - pop_entropy

def compute_allele_frequencies(population_genome):
    """
    Compute the proportion of individuals carrying allele 1 at each locus.
    
    Input: Population genome array.
    Output: A 1D array of length genome_size containing allele frequencies.
    """
    if len(population_genome) == 0:
        return np.array([])
    return np.mean(population_genome, axis=0)

def compute_genotype_diversity(population_genome):
    """
    Calculate the ratio of unique genotypes to population size.
    
    Input: Population genome array.
    Output: A scalar between 0 and 1.
    """
    if len(population_genome) == 0:
        return 0.0
    
    # Count unique rows
    unique_rows = np.unique(population_genome, axis=0)
    return len(unique_rows) / len(population_genome)
