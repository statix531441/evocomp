# Emergence of Complexity in Genetic Algorithms: An Information-Theoretic Approach

## Project Structure

```
evocomp/
├── ga.py                    # Core genetic algorithm, Environment and Population classes
├── information_metrics.py   # Information-theoretic metrics (entropy, mutual information)
├── data_logger.py          # EvolutionLogger for tracking metrics across generations
├── visualization.py        # Plotting functions for evolutionary dynamics
├── experiments.py          # Phase 1 and Phase 2 experiment configurations
├── main.py                 # CLI entry point for running experiments
└── scratch.ipynb          # Exploratory notebook
```

## Implementation Details

### Genome Encoding
- **Genome structure:** `[travel_gene, gene_1, gene_2, ..., gene_n]`
- **Travel gene (position 0):** Binary flag controlling migration between islands
- **Functional genes (positions 1-n):** Encode polynomial coefficients
- **Phenotype mapping:** `output = Σ(genome[i] × input^(i-1))` for i from 1 to genome_size-1

### Environment
- Supports both **static** (fixed fitness function) and **dynamic** (randomly changing) environments
- Configurable dynamicity parameter controls probability of environment change per generation
- Multiple fitness functions available: sum of squares, product, sum, max, min

### Information Metrics
- **Shannon Entropy:** Measures genetic diversity within population
- **Mutual Information:** Quantifies information the population has learned about environment
- **Physical Complexity:** Approximated as mutual information between genome and environment (Adami framework)

## Usage

### Running Experiments

The project implements two experimental phases as outlined in the proposal:

#### Phase 1: Single-Island Environment

**Phase 1a: Static Environment with Shock**
```bash
python main.py --experiment phase1_static --generations 200 --population_size 100
```
- Evolves population in static environment (x²)
- Applies environmental shock at generation 150 (switches to product function)
- Tests extinction risk in low-diversity populations

**Phase 1b: Dynamic Environment**
```bash
python main.py --experiment phase1_dynamic --generations 200 --population_size 100
```
- Evolves population with continuously changing environment
- Maintains higher diversity throughout evolution

**Phase 1c: Comparison**
```bash
python main.py --experiment phase1_compare --generations 200 --population_size 100
```
- Runs both static and dynamic experiments
- Generates comparative analysis of entropy, mutual information, and fitness dynamics

#### Phase 2: Two-Island Environment

```bash
python main.py --experiment phase2 --generations 300 --population_size 100
```
- Two populations on separate islands with different dynamicity levels
- Island 1: Static environment (dynamicity = 0.0)
- Island 2: Dynamic environment (dynamicity = 0.3)
- Individuals can migrate between islands based on travel gene
- Observes emergence of generalist genomes and travel behavior

### Command-Line Options

```bash
python main.py --help

Options:
  --experiment {phase1_static,phase1_dynamic,phase1_compare,phase2}
  --generations INT        Number of generations (default: 200)
  --population_size INT    Population size (default: 100)
  --mutation_rate FLOAT    Mutation probability per bit (default: 0.02)
  --crossover_rate FLOAT   Crossover probability (default: 0.7)
  --output_dir PATH        Directory for results (default: results)
```

### Output Files

All results are saved to the `--output_dir` (default: `results/`):

**CSV Files:**
- `phase1_static.csv` - Generation-by-generation metrics for static experiment
- `phase1_dynamic.csv` - Metrics for dynamic experiment
- `phase2_island1.csv`, `phase2_island2.csv` - Metrics for each island

**Plots:**
- Summary figures showing fitness, entropy, mutual information, and travel gene frequency
- Comparative plots for Phase 1 experiments
- Population dynamics and travel gene evolution for Phase 2

## References

1. Adami, C. (2002). What is complexity? *BioEssays*, 24(12), 1085–1094.
2. Hidalgo, J., et al. (2014). Information-based fitness and criticality in living systems. *PNAS*, 111(28), 10095–10100.
3. Krakauer, D. C. (2011). Darwinian demons, evolutionary complexity, and information maximization. *Chaos*, 21(3), 037110.
4. Lenski, R. E., et al. (2003). The evolutionary origin of complex features. *Nature*, 423, 139–144.
5. Szathmáry, E., & Smith, J. M. (1995). The major evolutionary transitions. *Nature*, 374, 227–232.
6. Walker, S. I., et al. (2016). *From matter to life: Information and causality*. Cambridge University Press.