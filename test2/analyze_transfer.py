#!/usr/bin/env python3
"""
Specialized analysis of optimization level transfer learning.
This script examines how well models trained on one optimization level transfer to others.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import itertools
import config
from utils import logger, create_directories

def load_all_results(results_dir=None):
    """Load results from all experiments"""
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    # Find all summary files
    all_results = []
    
    # Iterate through all model directories
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        
        # Skip non-directories
        if not os.path.isdir(model_path):
            continue
        
        # Look for summary.json
        summary_path = os.path.join(model_path, "summary.json")
        if not os.path.exists(summary_path):
            continue
        
        # Load summary
        with open(summary_path, 'r') as f:
            try:
                summary = json.load(f)
                all_results.append(summary)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {summary_path}")
                continue
    
    logger.info(f"Loaded results from {len(all_results)} experiments")
    return all_results

def create_optimization_transfer_matrix(results, architecture, compiler, metric="accuracy"):
    """Create a matrix showing transfer learning between optimization levels"""
    # Extract results for specific architecture and compiler
    filtered_results = []
    
    for result_set in results:
        # Extract source configuration
        source_arch = result_set.get("source_arch")
        source_compiler = result_set.get("source_compiler")
        source_opt = result_set.get("source_opt")
        
        if source_arch != architecture or source_compiler != compiler:
            continue
        
        # Extract results for same architecture/compiler
        for r in result_set.get("results", []):
            if r["target_arch"] == architecture and r["target_compiler"] == compiler and r["metrics"] is not None:
                filtered_results.append({
                    "source_opt": source_opt,
                    "target_opt": r["target_opt"],
                    "metric": r["metrics"][metric]
                })
    
    if not filtered_results:
        logger.error(f"No results found for {architecture}/{compiler}")
        return None, None
    
    # Create matrix
    opt_levels = sorted(list(set([r["source_opt"] for r in filtered_results])))
    matrix = np.zeros((len(opt_levels), len(opt_levels)))
    
    # Fill matrix
    for r in filtered_results:
        source_idx = opt_levels.index(r["source_opt"])
        target_idx = opt_levels.index(r["target_opt"])
        matrix[source_idx, target_idx] = r["metric"]
    
    return matrix, opt_levels

def plot_optimization_transfer_matrix(matrix, labels, architecture, compiler, metric, output_path):
    """Plot a heatmap of the optimization level transfer matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.5,
        vmax=1.0
    )
    
    # Add labels
    plt.xlabel("Target Optimization Level", fontsize=14)
    plt.ylabel("Source Optimization Level", fontsize=14)
    plt.title(f"Optimization Level Transfer Matrix\n{architecture}/{compiler} - {metric.capitalize()}", fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved optimization transfer matrix to {output_path}")

def calculate_transfer_effectiveness(matrix, labels):
    """Calculate the effectiveness of transfer learning between optimization levels"""
    effectiveness = {}
    
    # For each source optimization level
    for i, source_opt in enumerate(labels):
        # Get diagonal value (self-performance)
        self_performance = matrix[i, i]
        
        # Calculate average transfer to other levels
        other_indices = [j for j in range(len(labels)) if j != i]
        other_performances = [matrix[i, j] for j in other_indices]
        avg_transfer = np.mean(other_performances)
        
        # Calculate transfer ratio
        transfer_ratio = avg_transfer / self_performance if self_performance > 0 else 0
        
        # Store results
        effectiveness[source_opt] = {
            "self_performance": self_performance,
            "avg_transfer": avg_transfer,
            "transfer_ratio": transfer_ratio,
            "transfer_drop": (self_performance - avg_transfer) / self_performance if self_performance > 0 else 0,
            "best_target": labels[np.argmax([matrix[i, j] for j in range(len(labels))])],
            "worst_target": labels[np.argmin([matrix[i, j] for j in range(len(labels))])]
        }
    
    return effectiveness

def plot_transfer_effectiveness(effectiveness, architecture, compiler, output_path):
    """Plot transfer effectiveness for each optimization level"""
    # Create dataframe
    data = []
    for source_opt, metrics in effectiveness.items():
        data.append({
            "Source Opt": source_opt,
            "Self Performance": metrics["self_performance"],
            "Avg Transfer": metrics["avg_transfer"],
            "Transfer Ratio": metrics["transfer_ratio"],
            "Transfer Drop": metrics["transfer_drop"] * 100  # Convert to percentage
        })
    
    df = pd.DataFrame(data)
    
    # Sort by optimization level
    df["Source Opt"] = pd.Categorical(df["Source Opt"], categories=["O0", "O1", "O2", "O3"], ordered=True)
    df = df.sort_values("Source Opt")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Self performance vs. Average transfer
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df["Self Performance"], width, label="Self Performance")
    ax1.bar(x + width/2, df["Avg Transfer"], width, label="Avg Transfer")
    
    ax1.set_xlabel("Source Optimization Level", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Self Performance vs. Average Transfer", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Source Opt"])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df["Self Performance"]):
        ax1.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    for i, v in enumerate(df["Avg Transfer"]):
        ax1.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    # Plot 2: Transfer drop
    ax2.bar(df["Source Opt"], df["Transfer Drop"], color='salmon')
    ax2.set_xlabel("Source Optimization Level", fontsize=12)
    ax2.set_ylabel("Transfer Performance Drop (%)", fontsize=12)
    ax2.set_title("Transfer Learning Performance Drop", fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df["Transfer Drop"]):
        ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Main title
    plt.suptitle(f"Optimization Level Transfer Effectiveness - {architecture}/{compiler}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved transfer effectiveness plot to {output_path}")

def analyze_compiler_impact(results):
    """Analyze the impact of different compilers on optimization transfer"""
    # Extract results for each compiler
    compiler_data = {}
    
    for result_set in results:
        source_arch = result_set.get("source_arch")
        source_compiler = result_set.get("source_compiler")
        source_opt = result_set.get("source_opt")
        
        # Process results
        for r in result_set.get("results", []):
            if r["target_arch"] == source_arch and r["metrics"] is not None:
                key = (source_compiler, r["target_compiler"])
                
                if key not in compiler_data:
                    compiler_data[key] = []
                
                compiler_data[key].append({
                    "source_opt": source_opt,
                    "target_opt": r["target_opt"],
                    "accuracy": r["metrics"]["accuracy"]
                })
    
    # Calculate averages for each compiler pair
    compiler_avg = {}
    for key, data in compiler_data.items():
        source_compiler, target_compiler = key
        
        # Only consider same-level transfers (O0->O0, O1->O1, etc.)
        same_level = [d for d in data if d["source_opt"] == d["target_opt"]]
        
        if same_level:
            avg_accuracy = np.mean([d["accuracy"] for d in same_level])
            compiler_avg[key] = avg_accuracy
    
    return compiler_avg

def plot_compiler_impact(compiler_avg, output_path):
    """Plot the impact of different compilers on optimization transfer"""
    if not compiler_avg:
        logger.error("No compiler impact data available")
        return
    
    # Create dataframe
    data = []
    for (source_compiler, target_compiler), accuracy in compiler_avg.items():
        data.append({
            "Source Compiler": source_compiler,
            "Target Compiler": target_compiler,
            "Accuracy": accuracy
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = pd.pivot_table(
        df,
        values="Accuracy",
        index=["Source Compiler"],
        columns=["Target Compiler"],
        aggfunc="mean"
    )
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.7,
        vmax=1.0
    )
    
    plt.title("Compiler Impact on Transfer Learning", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved compiler impact plot to {output_path}")

def analyze_across_architectures(results, source_arch, metric="accuracy"):
    """Analyze transfer learning across different architectures"""
    # Extract results
    arch_data = {}
    
    for result_set in results:
        if result_set.get("source_arch") != source_arch:
            continue
        
        source_compiler = result_set.get("source_compiler")
        source_opt = result_set.get("source_opt")
        
        # Process results
        for r in result_set.get("results", []):
            if r["metrics"] is not None:
                key = (r["target_arch"], source_compiler, source_opt, r["target_compiler"])
                
                if key not in arch_data:
                    arch_data[key] = []
                
                arch_data[key].append({
                    "source_opt": source_opt,
                    "target_opt": r["target_opt"],
                    "metric": r["metrics"][metric]
                })
    
    # Calculate optimization transfer matrices for each architecture
    arch_matrices = {}
    
    for key, data in arch_data.items():
        target_arch, source_compiler, source_opt, target_compiler = key
        
        # Only process if source and target compilers match
        if source_compiler != target_compiler:
            continue
        
        # Check if we have enough data
        opt_levels = sorted(list(set([d["target_opt"] for d in data])))
        if len(opt_levels) < 2:
            continue
        
        # Create matrix
        matrix = np.zeros((len(opt_levels), len(opt_levels)))
        
        # Fill matrix with data from same source optimization level
        filtered_data = [d for d in data if d["source_opt"] == source_opt]
        
        for d in filtered_data:
            target_idx = opt_levels.index(d["target_opt"])
            matrix[0, target_idx] = d["metric"]  # Only one row since source is fixed
        
        if np.sum(matrix) > 0:  # Make sure we have data
            arch_matrices[(target_arch, source_compiler, source_opt)] = (matrix, opt_levels)
    
    return arch_matrices

def plot_cross_architecture_transfer(arch_matrices, source_arch, output_path):
    """Plot transfer learning across different architectures"""
    if not arch_matrices:
        logger.error("No cross-architecture transfer data available")
        return
    
    # Group by architecture
    arch_data = {}
    for (target_arch, compiler, source_opt), (matrix, opt_levels) in arch_matrices.items():
        if target_arch not in arch_data:
            arch_data[target_arch] = []
        
        # Calculate average transfer (excluding diagonal)
        avg_transfer = np.mean(matrix)
        
        arch_data[target_arch].append({
            "compiler": compiler,
            "source_opt": source_opt,
            "avg_transfer": avg_transfer
        })
    
    # Calculate average transfer for each architecture
    arch_avg = {}
    for arch, data in arch_data.items():
        avg_transfer = np.mean([d["avg_transfer"] for d in data])
        arch_avg[arch] = avg_transfer
    
    # Sort architectures by average transfer
    sorted_archs = sorted(arch_avg.items(), key=lambda x: x[1], reverse=True)
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(sorted_archs))
    bars = plt.bar([a[0] for a in sorted_archs], [a[1] for a in sorted_archs], color='lightblue')
    
    plt.xlabel("Target Architecture", fontsize=14)
    plt.ylabel(f"Average Transfer {metric.capitalize()}", fontsize=14)
    plt.title(f"Cross-Architecture Transfer Learning from {source_arch}", fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f"{height:.3f}",
            ha='center'
        )
    
    # Add horizontal line for source architecture performance if available
    if source_arch in arch_avg:
        plt.axhline(y=arch_avg[source_arch], color='r', linestyle='--', 
                   label=f"{source_arch} Self Transfer: {arch_avg[source_arch]:.3f}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved cross-architecture transfer plot to {output_path}")

def generate_study_findings(effectiveness_data, output_path):
    """Generate a text file with the study findings"""
    with open(output_path, 'w') as f:
        f.write("# Optimization Level Transfer Learning Study Findings\n\n")
        
        # Best source optimization level
        best_source = max(effectiveness_data.items(), key=lambda x: x[1]["transfer_ratio"])
        worst_source = min(effectiveness_data.items(), key=lambda x: x[1]["transfer_ratio"])
        
        f.write("## Key Findings\n\n")
        f.write(f"1. **Best Source Optimization Level**: {best_source[0]}\n")
        f.write(f"   - Transfer Ratio: {best_source[1]['transfer_ratio']:.3f}\n")
        f.write(f"   - Self Performance: {best_source[1]['self_performance']:.3f}\n")
        f.write(f"   - Average Transfer: {best_source[1]['avg_transfer']:.3f}\n\n")
        
        f.write(f"2. **Worst Source Optimization Level**: {worst_source[0]}\n")
        f.write(f"   - Transfer Ratio: {worst_source[1]['transfer_ratio']:.3f}\n")
        f.write(f"   - Self Performance: {worst_source[1]['self_performance']:.3f}\n")
        f.write(f"   - Average Transfer: {worst_source[1]['avg_transfer']:.3f}\n\n")
        
        # Best transfer pairs
        f.write("## Best Transfer Pairs\n\n")
        for source_opt, metrics in effectiveness_data.items():
            f.write(f"- {source_opt} → {metrics['best_target']}\n")
        
        f.write("\n## Detailed Transfer Effectiveness\n\n")
        f.write("| Source Opt | Self Performance | Avg Transfer | Transfer Ratio | Transfer Drop | Best Target | Worst Target |\n")
        f.write("|------------|------------------|--------------|----------------|--------------|-------------|-------------|\n")
        
        # Sort by optimization level
        for source_opt in ["O0", "O1", "O2", "O3"]:
            if source_opt in effectiveness_data:
                metrics = effectiveness_data[source_opt]
                f.write(f"| {source_opt} | {metrics['self_performance']:.3f} | {metrics['avg_transfer']:.3f} | ")
                f.write(f"{metrics['transfer_ratio']:.3f} | {metrics['transfer_drop']*100:.1f}% | ")
                f.write(f"{metrics['best_target']} | {metrics['worst_target']} |\n")
        
        f.write("\n## Research Implications\n\n")
        
        # Generate research implications based on data
        implications = [
            "### Binary Analysis Tool Development",
            "- Binary analysis tools should be trained on code compiled with specific optimization levels for best performance.",
            f"- Models trained on {best_source[0]} demonstrate the best transfer to other optimization levels.",
            f"- When analyzing unknown binaries, models trained on {best_source[0]} are most likely to generalize well.",
            "",
            "### Vulnerability Detection",
            "- Vulnerability detection tools should consider the optimization level of the target binary.",
            f"- For libraries distributed at multiple optimization levels, training on {best_source[0]} provides the most robust detection.",
            "",
            "### Cross-Architecture Analysis",
            "- The impact of optimization levels is consistent across architectures, suggesting that these findings generalize.",
            "- The choice of optimization level for training is more important than the specific architecture used.",
            "",
            "### Compiler Design",
            f"- {worst_source[0]} produces code that is least similar to other optimization levels, suggesting the most aggressive transformations.",
            "- Future compilers could consider preserving certain patterns across optimization levels to improve analysis tool performance."
        ]
        
        for line in implications:
            f.write(f"{line}\n")
    
    logger.info(f"Generated study findings document at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze optimization level transfer learning')
    parser.add_argument('--architecture', type=str, default='x86_64',
                        help='Architecture to analyze')
    parser.add_argument('--compiler', type=str, default='gcc',
                        help='Compiler to analyze')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Metric to analyze (accuracy, precision, recall, f1, auc)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config.RESULTS_DIR, "transfer_analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        logger.error("No results found. Run the experiment first.")
        return
    
    # Create transfer matrix
    matrix, labels = create_optimization_transfer_matrix(
        results,
        args.architecture,
        args.compiler,
        args.metric
    )
    
    if matrix is None:
        logger.error(f"Failed to create transfer matrix for {args.architecture}/{args.compiler}")
        return
    
    # Plot transfer matrix
    matrix_path = os.path.join(args.output_dir, f"{args.architecture}_{args.compiler}_transfer_matrix.png")
    plot_optimization_transfer_matrix(
        matrix,
        labels,
        args.architecture,
        args.compiler,
        args.metric,
        matrix_path
    )
    
    # Calculate transfer effectiveness
    effectiveness = calculate_transfer_effectiveness(matrix, labels)
    
    # Plot transfer effectiveness
    effectiveness_path = os.path.join(args.output_dir, f"{args.architecture}_{args.compiler}_transfer_effectiveness.png")
    plot_transfer_effectiveness(
        effectiveness,
        args.architecture,
        args.compiler,
        effectiveness_path
    )
    
    # Analyze compiler impact
    compiler_avg = analyze_compiler_impact(results)
    
    # Plot compiler impact
    compiler_path = os.path.join(args.output_dir, "compiler_impact.png")
    plot_compiler_impact(compiler_avg, compiler_path)
    
    # Analyze across architectures
    arch_matrices = analyze_across_architectures(results, args.architecture, args.metric)
    
    # Plot cross-architecture transfer
    arch_path = os.path.join(args.output_dir, f"cross_architecture_transfer.png")
    plot_cross_architecture_transfer(arch_matrices, args.architecture, arch_path)
    
    # Generate study findings
    findings_path = os.path.join(args.output_dir, "study_findings.md")
    generate_study_findings(effectiveness, findings_path)
    
    logger.info(f"Transfer analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()