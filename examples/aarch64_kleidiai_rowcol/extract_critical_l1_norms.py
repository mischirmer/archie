#!/usr/bin/env python3
"""
Script to extract L1 Norm values for Critical experiments from output.txt

Critical experiments are those where:
- Bytes differ but ABFT not triggered (marked as "ERROR ABFT: experimentXXXX - CRITICAL!")
"""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_critical_l1_norms(file_path):
    """
    Extract L1 Norm values from Critical experiments in the output file.
    
    Args:
        file_path (str): Path to the output.txt file
        
    Returns:
        list: List of tuples containing (experiment_id, l1_norm_value)
    """
    critical_experiments = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for Critical experiment markers
            if "ERROR ABFT:" in line and "CRITICAL!" in line:
                # Extract experiment ID
                experiment_match = re.search(r'experiment(\d+)', line)
                if experiment_match:
                    experiment_id = experiment_match.group(1)
                    
                    # Look for L1 Norm in the following lines
                    for j in range(i + 1, min(i + 10, len(lines))):
                        l1_norm_line = lines[j].strip()
                        if "L1 Norm:" in l1_norm_line:
                            # Extract L1 Norm value
                            l1_norm_match = re.search(r'L1 Norm:\s*([0-9.]+)', l1_norm_line)
                            if l1_norm_match:
                                l1_norm_value = float(l1_norm_match.group(1))
                                critical_experiments.append((experiment_id, l1_norm_value))
                            break
            i += 1
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return critical_experiments


def plot_l1_norms(critical_l1_norms, save_plots=True):
    """
    Create visualizations of the L1 Norm values.
    
    Args:
        critical_l1_norms (list): List of tuples containing (experiment_id, l1_norm_value)
        save_plots (bool): Whether to save plots to files
    """
    if not critical_l1_norms:
        print("No data to plot.")
        return
    
    experiment_ids = [int(exp_id) for exp_id, _ in critical_l1_norms]
    l1_values = [norm for _, norm in critical_l1_norms]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Critical Experiments L1 Norm Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot of L1 Norm values by experiment ID
    ax1.scatter(experiment_ids, l1_values, alpha=0.6, s=20, color='red')
    ax1.set_xlabel('Experiment ID')
    ax1.set_ylabel('L1 Norm Value')
    ax1.set_title('L1 Norm Values vs Experiment ID')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of L1 Norm values
    ax2.hist(l1_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('L1 Norm Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of L1 Norm Values')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    ax3.boxplot(l1_values, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax3.set_ylabel('L1 Norm Value')
    ax3.set_title('L1 Norm Value Distribution (Box Plot)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution
    sorted_values = np.sort(l1_values)
    cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax4.plot(sorted_values, cumulative_prob, marker='o', markersize=3, color='purple')
    ax4.set_xlabel('L1 Norm Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of L1 Norm Values')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('critical_l1_norms_analysis.png', dpi=300, bbox_inches='tight')
        print("Plot saved as: critical_l1_norms_analysis.png")
    
    plt.show()
    
    # Create a separate detailed histogram
    plt.figure(figsize=(12, 8))
    unique_values, counts = np.unique(l1_values, return_counts=True)
    
    plt.bar(range(len(unique_values)), counts, tick_label=[f'{val:.6f}' for val in unique_values])
    plt.xlabel('L1 Norm Value')
    plt.ylabel('Count')
    plt.title('Detailed Count of Each L1 Norm Value\n(Critical Experiments)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('critical_l1_norms_detailed_histogram.png', dpi=300, bbox_inches='tight')
        print("Detailed histogram saved as: critical_l1_norms_detailed_histogram.png")
    
    plt.show()
    
    # Print summary statistics
    print("\nDetailed L1 Norm Statistics:")
    print("=" * 40)
    for value, count in zip(unique_values, counts):
        percentage = (count / len(l1_values)) * 100
        print(f"L1 Norm {value:.6f}: {count:3d} experiments ({percentage:5.1f}%)")


def main():
    # Default file path
    default_file = "output.txt"
    
    # Allow file path as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' does not exist.")
        print(f"Usage: python {sys.argv[0]} [path_to_output.txt]")
        sys.exit(1)
    
    # Extract Critical L1 Norm values
    critical_l1_norms = extract_critical_l1_norms(file_path)
    
    if not critical_l1_norms:
        print("No Critical experiments found or no L1 Norm values extracted.")
        return
    
    # Display results
    print("Critical Experiments (Bytes different but ABFT not triggered)")
    print("=" * 60)
    print(f"{'Experiment ID':<15} {'L1 Norm':<15}")
    print("-" * 30)
    
    for experiment_id, l1_norm in critical_l1_norms:
        print(f"experiment{experiment_id:<7} {l1_norm:<15}")
    
    print("-" * 30)
    print(f"Total Critical experiments found: {len(critical_l1_norms)}")
    
    # Calculate statistics
    if critical_l1_norms:
        l1_values = [norm for _, norm in critical_l1_norms]
        print(f"Min L1 Norm: {min(l1_values)}")
        print(f"Max L1 Norm: {max(l1_values)}")
        print(f"Average L1 Norm: {sum(l1_values) / len(l1_values):.6f}")
    
    # Optionally save to file
    output_file = "critical_l1_norms.txt"
    with open(output_file, 'w') as f:
        f.write("Critical Experiments L1 Norm Values\n")
        f.write("===================================\n\n")
        for experiment_id, l1_norm in critical_l1_norms:
            f.write(f"experiment{experiment_id}: {l1_norm}\n")
    
    print(f"\nResults also saved to: {output_file}")
    
    # Create plots
    try:
        plot_l1_norms(critical_l1_norms)
    except ImportError:
        print("\nWarning: matplotlib not available. Skipping plots.")
        print("To install matplotlib, run: pip install matplotlib")
    except Exception as e:
        print(f"\nError creating plots: {e}")


if __name__ == "__main__":
    main()
    # Default file path
    default_file = "output.txt"
    
    # Allow file path as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' does not exist.")
        print(f"Usage: python {sys.argv[0]} [path_to_output.txt]")
        sys.exit(1)
    
    # Extract Critical L1 Norm values
    critical_l1_norms = extract_critical_l1_norms(file_path)
    
    if not critical_l1_norms:
        print("No Critical experiments found or no L1 Norm values extracted.")
        return
    
    # Display results
    print("Critical Experiments (Bytes different but ABFT not triggered)")
    print("=" * 60)
    print(f"{'Experiment ID':<15} {'L1 Norm':<15}")
    print("-" * 30)
    
    for experiment_id, l1_norm in critical_l1_norms:
        print(f"experiment{experiment_id:<7} {l1_norm:<15}")
    
    print("-" * 30)
    print(f"Total Critical experiments found: {len(critical_l1_norms)}")
    
    # Calculate statistics
    if critical_l1_norms:
        l1_values = [norm for _, norm in critical_l1_norms]
        print(f"Min L1 Norm: {min(l1_values)}")
        print(f"Max L1 Norm: {max(l1_values)}")
        print(f"Average L1 Norm: {sum(l1_values) / len(l1_values):.6f}")
    
    # Optionally save to file
    output_file = "critical_l1_norms.txt"
    with open(output_file, 'w') as f:
        f.write("Critical Experiments L1 Norm Values\n")
        f.write("===================================\n\n")
        for experiment_id, l1_norm in critical_l1_norms:
            f.write(f"experiment{experiment_id}: {l1_norm}\n")
    
    print(f"\nResults also saved to: {output_file}")


if __name__ == "__main__":
    main()
