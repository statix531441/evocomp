import os
import pandas as pd
import matplotlib.pyplot as plt

def load_dynamicity_results(results_dir):
    data = {}

    for filename in os.listdir(results_dir):
        if filename.startswith("dynamicity_") and filename.endswith(".csv"):
            try:
                value = float(filename.replace("dynamicity_", "").replace(".csv", ""))
            except ValueError:
                continue 

            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            data[value] = df

    if not data:
        print("No dynamicity sweep CSV files found in:", results_dir)
        exit(1)

    return dict(sorted(data.items())) 


def plot_metric_over_dynamicities(data, metric, output_path):

    plt.figure(figsize=(10, 6))

    for d, df in data.items():
        if metric not in df.columns:
            print(f"Warning: {metric} not found in CSV for dynamicity {d}")
            continue

        plt.plot(df[metric], label=f"d={d}")

    plt.xlabel("Generation")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Across Dynamicity Levels")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=250)
    print("Saved:", output_path)
    plt.close()


def main():
    results_dir = "results"
    output_dir = os.path.join(results_dir, "dynamicity_plots")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading CSV results from:", results_dir)

    data = load_dynamicity_results(results_dir)

    plot_metric_over_dynamicities(
        data, "mean_fitness", os.path.join(output_dir, "fitness_over_time.png")
    )

    plot_metric_over_dynamicities(
        data, "entropy", os.path.join(output_dir, "entropy_over_time.png")
    )

    plot_metric_over_dynamicities(
        data, "mutual_information", os.path.join(output_dir, "mutual_information_over_time.png")
    )

    print("\nAll plots saved in:", output_dir)


if __name__ == "__main__":
    main()
