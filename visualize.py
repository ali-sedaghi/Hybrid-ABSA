import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(csv_path):
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # Prepare output path
    output_dir = os.path.dirname(csv_path)
    output_file = os.path.join(output_dir, "results/comparison_plot.png")

    # Metrics to plot
    metrics = ["ATE_F1", "ASC_Accuracy", "ASC_F1"]
    models = df["Model"].tolist()

    # Setup Plot
    x = np.arange(len(metrics))
    width = 0.25  # Width of bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for distinct models
    colors = ["#e74c3c", "#3498db", "#2ecc71"]  # Red, Blue, Green

    # Plot bars for each model
    for i, model_name in enumerate(models):
        # Extract values for this model, handling "N/A" for baseline
        values = []
        row = df[df["Model"] == model_name].iloc[0]

        for m in metrics:
            val = row[m]
            if val == "N/A" or pd.isna(val):
                values.append(0)
            else:
                values.append(float(val))

        # Calculate offset for grouped bars
        offset = (i - len(models) / 2) * width + width / 2
        rects = ax.bar(
            x + offset, values, width, label=model_name, color=colors[i % len(colors)]
        )

        # Add labels on top of bars
        ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=8)

    # Styling
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Model Comparison: Baseline vs Instruct-DeBERTa variants")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)  # Add headroom for labels

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ABSA Experiment Results")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the results CSV file"
    )
    args = parser.parse_args()

    plot_results(args.input)
