#!/usr/bin/env python3
"""
Script to extract L1 Norm values for Critical experiments from output.txt

Critical experiments are those where:
- Bytes differ but ABFT not triggered (marked as "ERROR ABFT: experimentXXXX - CRITICAL!")
"""

import re
import sys
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
    
    # Create single histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(l1_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Absolute Difference in Checksums')
    plt.ylabel('Frequency')
    plt.title('Distribution of absolute checksum differences (Critical Experiments FN)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('critical_l1_norms_histogram.png', dpi=300, bbox_inches='tight')
        print("Plot saved as: critical_l1_norms_histogram.png")
    
    plt.show()
    
    # Print summary statistics
    unique_values, counts = np.unique(l1_values, return_counts=True)
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
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        try:
            plot_l1_norms(critical_l1_norms)
        except Exception as e:
            print(f"Error creating plots: {e}")
    else:
        print("\nWarning: matplotlib not available. Skipping plots.")
        print("To install matplotlib, run: pip install matplotlib")


if __name__ == "__main__":
    main()
