"""
Visualization utilities for instruction embeddings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from matplotlib.colors import LinearSegmentedColormap

def visualize_embedding_space(embeddings, labels=None, method='tsne', perplexity=30, 
                             output_path=None, title="Embedding Space Visualization"):
    """
    Visualize the embedding space using dimensionality reduction.
    
    Args:
        embeddings: Embedding vectors
        labels: Labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        perplexity: Perplexity parameter for t-SNE
        output_path: Path to save the visualization
        title: Plot title
    """
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    if labels is not None:
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Set up colormap
        cmap = cm.get_cmap('tab10', len(unique_labels))
        
        # Plot each label
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=[cmap(i)],
                label=str(label),
                alpha=0.7,
                edgecolors='none'
            )
        
        plt.legend(title="Labels")
    else:
        plt.scatter(
            reduced[:, 0], reduced[:, 1],
            alpha=0.7,
            edgecolors='none'
        )
    
    plt.title(title)
    plt.xlabel(f"Dimension 1")
    plt.ylabel(f"Dimension 2")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_similarity_matrix(similarity_matrix, labels_x=None, labels_y=None, 
                               output_path=None, title="Similarity Matrix"):
    """
    Visualize a similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix
        labels_x: Labels for x-axis
        labels_y: Labels for y-axis
        output_path: Path to save the visualization
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create a blue-white-red colormap for correlation matrix
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=100)
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=labels_x if labels_x is not None else True,
        yticklabels=labels_y if labels_y is not None else True
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_task_performance(model_scores, task_name, metrics=None, 
                              output_path=None, title=None):
    """
    Visualize task performance metrics across models.
    
    Args:
        model_scores: Dictionary mapping model names to score dictionaries
        task_name: Name of the task
        metrics: List of metrics to visualize (or None for all)
        output_path: Path to save the visualization
        title: Plot title
    """
    if not model_scores:
        return
    
    # Extract available metrics
    if metrics is None:
        # Find common metrics across all models
        metrics = set()
        for model, scores in model_scores.items():
            metrics.update(scores.keys())
        
        # Filter out non-numeric metrics
        metrics = [m for m in metrics if all(
            isinstance(scores.get(m, 0), (int, float)) 
            for scores in model_scores.values()
        )]
    
    # Sort metrics
    important_metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    metrics = sorted(metrics, key=lambda m: 
                    (0 if m in important_metrics else 1, important_metrics.index(m) if m in important_metrics else 0))
    
    # Set up plot
    num_metrics = len(metrics)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    models = list(model_scores.keys())
    num_models = len(models)
    bar_width = 0.8 / num_models
    
    # Create bars
    for i, model in enumerate(models):
        scores = model_scores[model]
        x = np.arange(num_metrics) + i * bar_width - 0.4 + bar_width / 2
        
        values = [scores.get(metric, 0) for metric in metrics]
        ax.bar(x, values, width=bar_width, label=model)
        
        # Add value labels
        for j, value in enumerate(values):
            ax.text(x[j], value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Set labels and title
    ax.set_xticks(np.arange(num_metrics))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylabel('Score')
    
    if title is None:
        title = f"{task_name} Performance by Model"
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_overall_comparison(scores, metrics=None, output_path=None):
    """
    Create overall comparison visualization across all models and tasks.
    
    Args:
        scores: Dictionary mapping model names to score dictionaries
        metrics: List of metrics to visualize (or None for default)
        output_path: Path to save the visualization
    """
    if metrics is None:
        metrics = ['weighted_score', 'average_accuracy', 'average_f1']
    
    # Extract data
    models = list(scores.keys())
    num_models = len(models)
    num_metrics = len(metrics)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    index = np.arange(num_models)
    bar_width = 0.8 / num_metrics
    
    # Create bars
    for i, metric in enumerate(metrics):
        values = [scores[model].get(metric, 0) for model in models]
        x = index + i * bar_width - 0.4 + bar_width / 2
        
        ax.bar(x, values, width=bar_width, label=metric)
        
        # Add value labels
        for j, value in enumerate(values):
            ax.text(x[j], value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Set labels and title
    ax.set_xticks(index)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_distribution(values_dict, output_path=None, title="Distribution Comparison", 
                          xlabel="Value", bins=30, density=True):
    """
    Visualize distributions of values.
    
    Args:
        values_dict: Dictionary mapping categories to lists of values
        output_path: Path to save the visualization
        title: Plot title
        xlabel: X-axis label
        bins: Number of histogram bins
        density: Whether to normalize the histogram
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram for each category
    for category, values in values_dict.items():
        plt.hist(values, bins=bins, alpha=0.5, label=category, density=density)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_confusion_matrix(true_labels, predictions, output_path=None, 
                              title="Confusion Matrix", cmap="Blues"):
    """
    Visualize a confusion matrix.
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        output_path: Path to save the visualization
        title: Plot title
        cmap: Colormap for the heatmap
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        square=True
    )
    
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def visualize_embedding_evolution(embeddings_dict, labels=None, method='tsne',
                                output_path=None, title_template="Embeddings After {}"):
    """
    Visualize evolution of embeddings across training.
    
    Args:
        embeddings_dict: Dictionary mapping stages to embedding arrays
        labels: Labels for coloring points
        method: Dimensionality reduction method
        output_path: Base path for saving visualizations
        title_template: Template for plot titles
    """
    # Sort stages
    stages = sorted(embeddings_dict.keys())
    
    # Create a common projection for fair comparison
    combined_embeddings = np.vstack([embeddings_dict[stage] for stage in stages])
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'")
    
    # Fit reducer on combined data
    combined_reduced = reducer.fit_transform(combined_embeddings)
    
    # Split back into separate stages
    start_idx = 0
    reduced_dict = {}
    
    for stage in stages:
        stage_size = len(embeddings_dict[stage])
        reduced_dict[stage] = combined_reduced[start_idx:start_idx + stage_size]
        start_idx += stage_size
    
    # Create visualizations for each stage
    for stage in stages:
        stage_reduced = reduced_dict[stage]
        stage_title = title_template.format(stage)
        
        if output_path:
            stage_output_path = output_path.replace('.png', f'_{stage}.png')
        else:
            stage_output_path = None
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        if labels is not None:
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Set up colormap
            cmap = cm.get_cmap('tab10', len(unique_labels))
            
            # Plot each label
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(
                    stage_reduced[mask, 0], stage_reduced[mask, 1],
                    c=[cmap(i)],
                    label=str(label),
                    alpha=0.7,
                    edgecolors='none'
                )
            
            plt.legend(title="Labels")
        else:
            plt.scatter(
                stage_reduced[:, 0], stage_reduced[:, 1],
                alpha=0.7,
                edgecolors='none'
            )
        
        plt.title(stage_title)
        plt.xlabel(f"Dimension 1")
        plt.ylabel(f"Dimension 2")
        plt.tight_layout()
        
        if stage_output_path:
            os.makedirs(os.path.dirname(stage_output_path), exist_ok=True)
            plt.savefig(stage_output_path, dpi=300)
        
        plt.close()