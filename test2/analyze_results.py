#!/usr/bin/env python3
"""
Analyze and visualize experiment results.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import config
from utils import logger, create_directories

def load_results(model_id):
    """Load results for a specific model"""
    results_dir = os.path.join(config.RESULTS_DIR, model_id)
    summary_path = os.path.join(results_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        logger.error(f"Summary file not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary

def create_data_frame(results):
    """Create a DataFrame from results"""
    data = []
    
    for result in results:
        if result["metrics"] is not None:
            data.append({
                "Source Architecture": result["source_arch"],
                "Source Compiler": result["source_compiler"],
                "Source Optimization": result["source_opt"],
                "Target Architecture": result["target_arch"],
                "Target Compiler": result["target_compiler"],
                "Target Optimization": result["target_opt"],
                "Accuracy": result["metrics"]["accuracy"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1": result["metrics"]["f1"],
                "AUC": result["metrics"]["auc"]
            })
    
    return pd.DataFrame(data)

def plot_optimization_matrix(df, output_path, source_arch, source_compiler, source_opt):
    """Plot optimization level impact matrix"""
    # Pivot table for optimization levels by architecture and compiler
    pivot = pd.pivot_table(
        df,
        values="Accuracy",
        index=["Target Architecture", "Target Compiler"],
        columns=["Target Optimization"],
        aggfunc="mean"
    )
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy"}
    )
    
    plt.title(f"Impact of Optimization Levels on Accuracy\nSource: {source_arch}/{source_compiler}/{source_opt}", fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_architecture_transfer(df, output_path, source_arch):
    """Plot architecture transfer performance"""
    # Group by target architecture
    arch_data = df.groupby("Target Architecture").mean()
    
    # Metrics to plot
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    x = np.arange(len(arch_data.index))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(
            x + (i - len(metrics)/2 + 0.5) * width,
            arch_data[metric],
            width=width,
            label=metric
        )
    
    plt.xlabel("Target Architecture", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(f"Knowledge Transfer from {source_arch} to Other Architectures", fontsize=16)
    plt.xticks(x, arch_data.index, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Add source architecture line
    if source_arch in arch_data.index:
        source_acc = arch_data.loc[source_arch, "Accuracy"]
        plt.axhline(y=source_acc, color='r', linestyle='--', 
                   label=f"{source_arch} Self-Test: {source_acc:.3f}")
        plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_compiler_impact(df, output_path, source_compiler):
    """Plot compiler impact on accuracy"""
    # Group by target compiler and architecture
    compiler_data = df.groupby(["Target Architecture", "Target Compiler"]).mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Sort by architecture first, then by compiler
    compiler_data = compiler_data.sort_values(["Target Architecture", "Target Compiler"])
    
    # Create grouped bar chart
    x = np.arange(len(compiler_data))
    
    # Plot accuracy bars
    plt.bar(x, compiler_data["Accuracy"], color='skyblue')
    
    # Add error bars if we have multiple data points per group
    if len(df) > len(compiler_data):
        # Calculate standard error for each group
        std_data = df.groupby(["Target Architecture", "Target Compiler"])["Accuracy"].std().reset_index()
        std_data = std_data.sort_values(["Target Architecture", "Target Compiler"])
        plt.errorbar(x, compiler_data["Accuracy"], yerr=std_data["Accuracy"], fmt='none', ecolor='black', capsize=5)
    
    # Customize x-axis labels
    labels = [f"{row['Target Architecture']}/{row['Target Compiler']}" 
              for _, row in compiler_data.iterrows()]
    
    plt.xlabel("Target Architecture/Compiler", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Impact of Compiler Choice on Accuracy\nSource Compiler: {source_compiler}", fontsize=16)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Highlight source compiler cases
    source_indices = [i for i, row in enumerate(labels) if source_compiler in row]
    for idx in source_indices:
        plt.bar(idx, compiler_data.iloc[idx]["Accuracy"], color='orange')
    
    # Add legend
    plt.bar(0, 0, color='skyblue', label='Other Compilers')
    plt.bar(0, 0, color='orange', label=f'Source Compiler ({source_compiler})')
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_optimization_trends(df, output_path, source_opt):
    """Plot optimization level trends"""
    # Group by target architecture and optimization level
    opt_data = df.groupby(["Target Architecture", "Target Optimization"]).mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    
    # Get unique architectures and optimization levels
    archs = sorted(opt_data["Target Architecture"].unique())
    opts = sorted(opt_data["Target Optimization"].unique())
    
    # Set up colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(archs)))
    
    # Plot a line for each architecture
    for i, arch in enumerate(archs):
        arch_data = opt_data[opt_data["Target Architecture"] == arch]
        arch_data = arch_data.sort_values("Target Optimization")
        
        plt.plot(
            arch_data["Target Optimization"],
            arch_data["Accuracy"],
            marker='o',
            linewidth=2,
            color=colors[i],
            label=arch
        )
    
    plt.xlabel("Target Optimization Level", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Impact of Optimization Levels on Accuracy\nSource Optimization: {source_opt}", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.legend(fontsize=12, title="Target Architecture")
    
    # Highlight source optimization level
    plt.axvline(x=source_opt, color='r', linestyle='--', 
               label=f"Source Optimization ({source_opt})")
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_summary_table(df, output_path, source_arch, source_compiler, source_opt):
    """Create a summary table of results"""
    # Overall average metrics
    overall_metrics = df.mean()
    
    # Per-architecture averages
    arch_metrics = df.groupby("Target Architecture").mean()
    
    # Per-compiler averages
    compiler_metrics = df.groupby("Target Compiler").mean()
    
    # Per-optimization averages
    opt_metrics = df.groupby("Target Optimization").mean()
    
    # Create an Excel writer object
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Convert overall metrics to DataFrame and write to Excel
        pd.DataFrame(overall_metrics).T.to_excel(writer, sheet_name='Overall')
        
        # Write architecture metrics
        arch_metrics.to_excel(writer, sheet_name='By Architecture')
        
        # Write compiler metrics
        compiler_metrics.to_excel(writer, sheet_name='By Compiler')
        
        # Write optimization metrics
        opt_metrics.to_excel(writer, sheet_name='By Optimization')
        
        # Write full results
        df.to_excel(writer, sheet_name='Full Results')
        
        # Get workbook and add a new worksheet for source configuration
        workbook = writer.book
        source_sheet = workbook.add_worksheet('Source Configuration')
        
        # Write source configuration
        source_sheet.write(0, 0, 'Source Architecture')
        source_sheet.write(0, 1, source_arch)
        source_sheet.write(1, 0, 'Source Compiler')
        source_sheet.write(1, 1, source_compiler)
        source_sheet.write(2, 0, 'Source Optimization')
        source_sheet.write(2, 1, source_opt)

def calculate_relative_performance(df, source_arch, source_compiler, source_opt):
    """Calculate relative performance compared to source configuration"""
    # Get accuracy for source configuration
    source_mask = (
        (df["Source Architecture"] == source_arch) &
        (df["Source Compiler"] == source_compiler) &
        (df["Source Optimization"] == source_opt) &
        (df["Target Architecture"] == source_arch) &
        (df["Target Compiler"] == source_compiler) &
        (df["Target Optimization"] == source_opt)
    )
    
    if not source_mask.any():
        # If no exact match, use source architecture results
        source_mask = (df["Target Architecture"] == source_arch)
    
    source_acc = df.loc[source_mask, "Accuracy"].mean()
    
    # Calculate relative performance
    df["Relative Accuracy"] = df["Accuracy"] / source_acc
    
    return df, source_acc

def plot_relative_performance(df, output_path, source_acc):
    """Plot relative performance compared to source configuration"""
    # Group by target architecture
    arch_data = df.groupby("Target Architecture")["Relative Accuracy"].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Sort bars by performance
    arch_data = arch_data.sort_values("Relative Accuracy", ascending=False)
    
    # Create bar chart
    bars = plt.bar(arch_data["Target Architecture"], arch_data["Relative Accuracy"], color='skyblue')
    
    # Add 100% line (source accuracy)
    plt.axhline(y=1.0, color='r', linestyle='--', label=f"Source (Accuracy: {source_acc:.3f})")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f"{height:.2%}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.xlabel("Target Architecture", fontsize=14)
    plt.ylabel("Relative Accuracy", fontsize=14)
    plt.title("Relative Performance Compared to Source Configuration", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(arch_data["Relative Accuracy"]) * 1.1)
    plt.legend()
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze model evaluation results')
    parser.add_argument('--source-arch', type=str, default='x86_64',
                        help='Source architecture used for training')
    parser.add_argument('--source-compiler', type=str, default='gcc',
                        help='Source compiler used for training')
    parser.add_argument('--source-opt', type=str, default='O2',
                        help='Source optimization level used for training')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Define model ID and results directory
    model_id = f"{args.source_arch}_{args.source_compiler}_{args.source_opt}"
    results_dir = os.path.join(config.RESULTS_DIR, model_id)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load results
    summary = load_results(model_id)
    
    if summary is None:
        logger.error(f"No results found for model {model_id}")
        return
    
    results = summary["results"]
    
    # Create DataFrame
    df = create_data_frame(results)
    
    if df.empty:
        logger.error("No valid results to analyze")
        return
    
    # Calculate relative performance
    df, source_acc = calculate_relative_performance(df, args.source_arch, args.source_compiler, args.source_opt)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Optimization matrix
    plot_optimization_matrix(
        df,
        os.path.join(results_dir, "optimization_matrix.png"),
        args.source_arch,
        args.source_compiler,
        args.source_opt
    )
    
    # 2. Architecture transfer performance
    plot_architecture_transfer(
        df,
        os.path.join(results_dir, "architecture_transfer.png"),
        args.source_arch
    )
    
    # 3. Compiler impact
    plot_compiler_impact(
        df,
        os.path.join(results_dir, "compiler_impact.png"),
        args.source_compiler
    )
    
    # 4. Optimization trends
    plot_optimization_trends(
        df,
        os.path.join(results_dir, "optimization_trends.png"),
        args.source_opt
    )
    
    # 5. Relative performance
    plot_relative_performance(
        df,
        os.path.join(results_dir, "relative_performance.png"),
        source_acc
    )
    
    # 6. Create summary table
    create_summary_table(
        df,
        os.path.join(results_dir, "results_summary.xlsx"),
        args.source_arch,
        args.source_compiler,
        args.source_opt
    )
    
    # Save as CSV as well
    df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)
    
    logger.info(f"Analysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()