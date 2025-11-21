"""
Main entry point for running evolution experiments.
"""
import argparse
import os
import matplotlib.pyplot as plt
from experiments import (
    run_phase1_static, 
    run_phase1_dynamic, 
    compare_phase1_results, 
    run_phase2_two_islands, 
    analyze_phase2_travel_evolution
)
from visualization import (
    plot_fitness_over_time, 
    plot_entropy_over_time, 
    plot_mutual_information, 
    create_summary_figure,
    plot_population_dynamics,
    plot_travel_gene_frequency
)

def main():
    parser = argparse.ArgumentParser(description="Evolutionary Computation Project")
    parser.add_argument('--experiment', type=str, required=True, 
                        choices=['phase1_static', 'phase1_dynamic', 'phase1_compare', 'phase2'],
                        help='Experiment to run')
    parser.add_argument('--generations', type=int, default=200, help='Number of generations')
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--mutation_rate', type=float, default=0.05, help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Running {args.experiment} with settings: {vars(args)}")
    
    if args.experiment == 'phase1_static':
        logger = run_phase1_static(
            generations=args.generations,
            N=args.population_size,
            p_c=args.crossover_rate,
            p_m=args.mutation_rate
        )
        logger.save_csv(os.path.join(args.output_dir, 'phase1_static.csv'))
        create_summary_figure(logger, os.path.join(args.output_dir, 'phase1_static_summary.png'))
        
    elif args.experiment == 'phase1_dynamic':
        logger = run_phase1_dynamic(
            generations=args.generations,
            N=args.population_size,
            p_c=args.crossover_rate,
            p_m=args.mutation_rate
        )
        logger.save_csv(os.path.join(args.output_dir, 'phase1_dynamic.csv'))
        create_summary_figure(logger, os.path.join(args.output_dir, 'phase1_dynamic_summary.png'))
        
    elif args.experiment == 'phase1_compare':
        # Run both
        static_logger = run_phase1_static(
            generations=args.generations,
            N=args.population_size,
            p_c=args.crossover_rate,
            p_m=args.mutation_rate
        )
        dynamic_logger = run_phase1_dynamic(
            generations=args.generations,
            N=args.population_size,
            p_c=args.crossover_rate,
            p_m=args.mutation_rate
        )
        
        compare_phase1_results(static_logger, dynamic_logger)
        
        # Save results
        static_logger.save_csv(os.path.join(args.output_dir, 'phase1_static.csv'))
        dynamic_logger.save_csv(os.path.join(args.output_dir, 'phase1_dynamic.csv'))
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(static_logger.get_metric_series('entropy'), label='Static Entropy')
        plt.plot(dynamic_logger.get_metric_series('entropy'), label='Dynamic Entropy')
        plt.xlabel('Generation')
        plt.ylabel('Entropy')
        plt.title('Entropy Comparison: Static vs Dynamic')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'phase1_entropy_comparison.png'))
        
    elif args.experiment == 'phase2':
        logger1, logger2 = run_phase2_two_islands(
            generations=args.generations,
            N=args.population_size,
            p_c=args.crossover_rate,
            p_m=args.mutation_rate
        )
        
        analyze_phase2_travel_evolution(logger1, logger2)
        
        logger1.save_csv(os.path.join(args.output_dir, 'phase2_island1.csv'))
        logger2.save_csv(os.path.join(args.output_dir, 'phase2_island2.csv'))
        
        # Plot population dynamics
        plot_population_dynamics([logger1, logger2], ['Island 1 (Static)', 'Island 2 (Dynamic)'], 
                                 os.path.join(args.output_dir, 'phase2_pop_dynamics.png'))
                                 
        # Plot travel gene freq
        plt.figure(figsize=(10, 6))
        plt.plot(logger1.get_metric_series('travel_gene_freq'), label='Island 1 (Static)')
        plt.plot(logger2.get_metric_series('travel_gene_freq'), label='Island 2 (Dynamic)')
        plt.xlabel('Generation')
        plt.ylabel('Travel Gene Frequency')
        plt.title('Travel Gene Evolution')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'phase2_travel_evolution.png'))

if __name__ == "__main__":
    main()
