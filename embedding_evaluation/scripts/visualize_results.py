#!/usr/bin/env python
"""
Script for visualizing experiment results.
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Add src to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize instruction embedding evaluation results"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Directory containing results"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualizations",
        help="Directory for saving visualizations"
    )
    
    parser.add_argument(
        "--format", 
        type=str, 
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for visualizations"
    )
    
    return parser.parse_args()

def load_latest_results(results_dir):
    """Load the latest results from the results directory."""
    # Find the latest score file
    score_files = glob.glob(os.path.join(results_dir, "scores_*.json"))
    if not score_files:
        raise FileNotFoundError(f"No score files found in {results_dir}")
    
    latest_score_file = max(score_files, key=os.path.getctime)
    
    # Load scores
    with open(latest_score_file, 'r') as f:
        scores = json.load(f)
    
    # Find corresponding result file
    timestamp = os.path.basename(latest_score_file).split('_')[1].split('.')[0]
    result_file = os.path.join(results_dir, f"results_{timestamp}.json")
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file {result_file} not found")
    
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return scores, results, timestamp

def create_overall_comparison(scores, output_path, format="png"):
    """Create overall comparison visualization."""
    # Extract overall metrics
    models = list(scores.keys())
    weighted_scores = [scores[model]['weighted_score'] for model in models]
    accuracies = [scores[model]['average_accuracy'] for model in models]
    f1_scores = [scores[model]['average_f1'] for model in models]
    
    # Sort by weighted score
    sorted_indices = np.argsort(weighted_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    weighted_scores = [weighted_scores[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up positions
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars
    ax.bar(x - width, weighted_scores, width, label='Weighted Score')
    ax.bar(x, accuracies, width, label='Average Accuracy')
    ax.bar(x + width, f1_scores, width, label='Average F1')
    
    # Set labels and title
    ax.set_xlabel('Embedding Model')
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(weighted_scores):
        ax.text(i - width, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(f1_scores):
        ax.text(i + width, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_path}/overall_comparison.{format}", dpi=300)
    plt.close()

def create_task_comparison(scores, task_name, output_path, format="png"):
    """Create task-specific comparison visualization."""
    # Collect metrics
    metrics = set()
    for model in scores:
        if task_name in scores[model]['task_scores']:
            metrics.update(scores[model]['task_scores'][task_name].keys())
    
    # Filter out non-numeric metrics
    numeric_metrics = []
    for metric in metrics:
        is_numeric = True
        for model in scores:
            if task_name in scores[model]['task_scores']:
                value = scores[model]['task_scores'][task_name].get(metric)
                if value is not None and not isinstance(value, (int, float)):
                    is_numeric = False
                    break
        
        if is_numeric:
            numeric_metrics.append(metric)
    
    # Extract metric values
    data = []
    for model in scores:
        if task_name in scores[model]['task_scores']:
            row = {'Model': model}
            for metric in numeric_metrics:
                row[metric] = scores[model]['task_scores'][task_name].get(metric, 0)
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    if 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Create a heatmap
    pivot_df = df.set_index('Model')
    
    # Normalize values for better visualization
    normalized_df = pivot_df.copy()
    for col in normalized_df.columns:
        normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / \
                            (normalized_df[col].max() - normalized_df[col].min())
    
    ax = sns.heatmap(normalized_df, annot=pivot_df, fmt='.4f', cmap='YlGnBu')
    
    plt.title(f'{task_name} Performance by Model')
    plt.tight_layout()
    
    # Save figure
    task_file = task_name.lower().replace(' ', '_')
    plt.savefig(f"{output_path}/{task_file}_comparison.{format}", dpi=300)
    plt.close()

def create_similarity_distribution(results, output_path, format="png"):
    """Create similarity distribution visualization."""
    # Collect similarity data
    synonym_data = defaultdict(list)
    block_data = defaultdict(list)
    
    for model_name, tasks in results.items():
        # Instruction synonym data
        if 'Instruction Synonym Detection' in tasks:
            task_results = tasks['Instruction Synonym Detection']['results']
            similarities = task_results['similarities']
            labels = task_results['true_labels']
            
            for sim, label in zip(similarities, labels):
                category = 'Synonym' if label else 'Non-Synonym'
                synonym_data[(model_name, category)].append(sim)
        
        # Semantic block data
        if 'Semantic Block Equivalence' in tasks:
            task_results = tasks['Semantic Block Equivalence']['results']
            similarities = task_results['similarities']
            labels = task_results['true_labels']
            
            for sim, label in zip(similarities, labels):
                category = 'Equivalent' if label else 'Non-Equivalent'
                block_data[(model_name, category)].append(sim)
    
    # Create visualizations if data exists
    if synonym_data:
        plt.figure(figsize=(12, 6))
        
        for model_name in set(k[0] for k in synonym_data.keys()):
            # Get synonym and non-synonym data for this model
            syn_sims = synonym_data.get((model_name, 'Synonym'), [])
            non_syn_sims = synonym_data.get((model_name, 'Non-Synonym'), [])
            
            if syn_sims and non_syn_sims:
                # Create subplot for this model
                plt.hist(
                    [syn_sims, non_syn_sims], 
                    bins=20, 
                    alpha=0.6, 
                    label=[f'{model_name} - Synonym', f'{model_name} - Non-Synonym']
                )
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Instruction Synonym Detection: Similarity Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/synonym_similarity_distribution.{format}", dpi=300)
        plt.close()
    
    if block_data:
        plt.figure(figsize=(12, 6))
        
        for model_name in set(k[0] for k in block_data.keys()):
            # Get equivalent and non-equivalent data for this model
            equiv_sims = block_data.get((model_name, 'Equivalent'), [])
            non_equiv_sims = block_data.get((model_name, 'Non-Equivalent'), [])
            
            if equiv_sims and non_equiv_sims:
                # Create subplot for this model
                plt.hist(
                    [equiv_sims, non_equiv_sims], 
                    bins=20, 
                    alpha=0.6, 
                    label=[f'{model_name} - Equivalent', f'{model_name} - Non-Equivalent']
                )
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Semantic Block Equivalence: Similarity Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/block_similarity_distribution.{format}", dpi=300)
        plt.close()

def create_impact_score_visualization(results, output_path, format="png"):
    """Create impact score visualization for dead code detection."""
    # Collect impact score data
    impact_data = defaultdict(list)
    
    for model_name, tasks in results.items():
        if 'Dead Code Detection' in tasks:
            task_results = tasks['Dead Code Detection']['results']
            impact_scores = task_results['impact_scores']
            labels = task_results['true_labels']
            
            for score, label in zip(impact_scores, labels):
                category = 'Dead Code' if label else 'Live Code'
                impact_data[(model_name, category)].append(score)
    
    # Create visualization if data exists
    if impact_data:
        plt.figure(figsize=(12, 6))
        
        for model_name in set(k[0] for k in impact_data.keys()):
            # Get dead code and live code data for this model
            dead_scores = impact_data.get((model_name, 'Dead Code'), [])
            live_scores = impact_data.get((model_name, 'Live Code'), [])
            
            if dead_scores and live_scores:
                plt.hist(
                    [dead_scores, live_scores], 
                    bins=20, 
                    alpha=0.6, 
                    label=[f'{model_name} - Dead Code', f'{model_name} - Live Code']
                )
        
        plt.xlabel('Impact Score')
        plt.ylabel('Frequency')
        plt.title('Dead Code Detection: Impact Score Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/dead_code_impact_distribution.{format}", dpi=300)
        plt.close()

def create_embedding_visualization(results, output_path, format="png"):
    """Create embedding space visualization."""
    # This would typically use dimensionality reduction to visualize the embedding space
    # For brevity, we'll just create a placeholder for this functionality
    
    # Note: For a real implementation, you would:
    # 1. Collect embeddings for a set of instructions
    # 2. Use PCA or t-SNE to reduce to 2D or 3D
    # 3. Create a scatter plot with semantically related instructions
    
    plt.figure(figsize=(12, 6))
    plt.text(
        0.5, 0.5, 
        "Embedding space visualization\nwould be implemented here\n(requires actual embeddings)", 
        ha='center', va='center', fontsize=14
    )
    plt.axis('off')
    plt.savefig(f"{output_path}/embedding_visualization_placeholder.{format}", dpi=300)
    plt.close()

def main():
    """Main function for visualizing results."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    scores, results, timestamp = load_latest_results(args.results_dir)
    
    print(f"Loaded results from {timestamp}")
    print(f"Creating visualizations in {args.output_dir}")
    
    # Create visualizations
    create_overall_comparison(scores, args.output_dir, args.format)
    
    # Task-specific visualizations
    task_names = set()
    for model in scores:
        task_names.update(scores[model]['task_scores'].keys())
    
    for task_name in task_names:
        create_task_comparison(scores, task_name, args.output_dir, args.format)
    
    # Additional visualizations
    create_similarity_distribution(results, args.output_dir, args.format)
    create_impact_score_visualization(results, args.output_dir, args.format)
    create_embedding_visualization(results, args.output_dir, args.format)
    
    print(f"Done! Visualizations saved in {args.output_dir}")

if __name__ == "__main__":
    main()