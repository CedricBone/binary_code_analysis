"""
Error analysis utilities for instruction embeddings.
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

def find_error_cases(predictions, true_labels, data, error_type='all'):
    """
    Find error cases in the evaluation results.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        data: Input data used for evaluation
        error_type: Type of errors to find ('fp', 'fn', or 'all')
            - 'fp': False positives (predicted True but actually False)
            - 'fn': False negatives (predicted False but actually True)
            - 'all': All errors
    
    Returns:
        list: List of (index, data_point, true_label, predicted_label) tuples for errors
    """
    error_cases = []
    
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if error_type == 'fp' and pred and not true:
            # False positive
            error_cases.append((i, data[i], true, pred))
        elif error_type == 'fn' and not pred and true:
            # False negative
            error_cases.append((i, data[i], true, pred))
        elif error_type == 'all' and pred != true:
            # Any error
            error_cases.append((i, data[i], true, pred))
    
    return error_cases

def analyze_errors(model_name, task_results, task_data):
    """
    Analyze patterns in model errors.
    
    Args:
        model_name: Name of the model
        task_results: Results from task evaluation
        task_data: Original task data
        
    Returns:
        dict: Error analysis report
    """
    predictions = task_results['predictions']
    true_labels = task_results['true_labels']
    
    # Find error indices
    error_indices = [i for i, (pred, true) in enumerate(zip(predictions, true_labels)) if pred != true]
    
    # Categorize errors
    error_categories = categorize_errors(error_indices, task_results, task_data)
    
    # Calculate statistics
    total_samples = len(predictions)
    total_errors = len(error_indices)
    error_rate = total_errors / total_samples if total_samples > 0 else 0
    
    # Count error types
    false_positives = sum(1 for i in error_indices if predictions[i] and not true_labels[i])
    false_negatives = sum(1 for i in error_indices if not predictions[i] and true_labels[i])
    
    fp_rate = false_positives / sum(1 for label in true_labels if not label) if sum(1 for label in true_labels if not label) > 0 else 0
    fn_rate = false_negatives / sum(true_labels) if sum(true_labels) > 0 else 0
    
    # Create report
    report = {
        'model_name': model_name,
        'total_samples': total_samples,
        'total_errors': total_errors,
        'error_rate': error_rate,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'error_categories': error_categories
    }
    
    return report

def categorize_errors(error_indices, task_results, task_data):
    """
    Categorize errors based on patterns.
    
    Args:
        error_indices: Indices of errors
        task_results: Results from task evaluation
        task_data: Original task data
        
    Returns:
        dict: Dictionary mapping error categories to counts
    """
    # This function needs to be customized based on the specific task
    # Here's a generic implementation that categorizes based on available scores
    
    error_categories = defaultdict(int)
    predictions = task_results['predictions']
    true_labels = task_results['true_labels']
    
    # Check if we have similarity or impact scores
    has_similarities = 'similarities' in task_results or 'enhanced_similarities' in task_results
    has_impacts = 'embedding_impacts' in task_results or 'combined_impacts' in task_results
    
    for idx in error_indices:
        # Determine error type
        if predictions[idx] and not true_labels[idx]:
            error_type = 'false_positive'
        else:
            error_type = 'false_negative'
        
        # Categorize based on scores
        if has_similarities:
            # Get similarity score
            if 'enhanced_similarities' in task_results:
                similarity = task_results['enhanced_similarities'][idx]
            else:
                similarity = task_results['similarities'][idx]
            
            # Categorize based on similarity
            if similarity > 0.9:
                error_categories[f'{error_type}_high_similarity'] += 1
            elif similarity > 0.7:
                error_categories[f'{error_type}_medium_similarity'] += 1
            else:
                error_categories[f'{error_type}_low_similarity'] += 1
        
        elif has_impacts:
            # Get impact score
            if 'combined_impacts' in task_results:
                impact = task_results['combined_impacts'][idx]
            else:
                impact = task_results['embedding_impacts'][idx]
            
            # Categorize based on impact
            if impact > 0.5:
                error_categories[f'{error_type}_high_impact'] += 1
            elif impact > 0.2:
                error_categories[f'{error_type}_medium_impact'] += 1
            else:
                error_categories[f'{error_type}_low_impact'] += 1
        
        else:
            # Basic categorization
            error_categories[error_type] += 1
    
    return dict(error_categories)

def visualize_error_distribution(error_categories, output_path=None, title="Error Distribution"):
    """
    Visualize error distribution across categories.
    
    Args:
        error_categories: Dictionary mapping error categories to counts
        output_path: Path to save the visualization
        title: Plot title
    """
    # Sort categories by count
    sorted_categories = sorted(error_categories.items(), key=lambda x: x[1], reverse=True)
    categories, counts = zip(*sorted_categories) if sorted_categories else ([], [])
    
    plt.figure(figsize=(12, 6))
    
    # Create horizontal bar chart
    y_pos = range(len(categories))
    plt.barh(y_pos, counts)
    
    # Set labels
    plt.yticks(y_pos, categories)
    plt.xlabel('Count')
    plt.title(title)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def analyze_confidence_vs_correctness(scores, true_labels, output_path=None):
    """
    Analyze and visualize relationship between confidence and correctness.
    
    Args:
        scores: Confidence scores for predictions
        true_labels: True labels
        output_path: Path to save the visualization
    """
    # Convert to numpy arrays
    scores = np.array(scores)
    true_labels = np.array(true_labels)
    
    # Create bins based on confidence
    num_bins = 10
    bin_edges = np.linspace(scores.min(), scores.max(), num_bins + 1)
    
    # Calculate accuracy in each bin
    bin_accuracies = []
    bin_counts = []
    
    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        # Find samples in this bin
        bin_mask = (scores >= lower) & (scores < upper)
        bin_samples = bin_mask.sum()
        
        if bin_samples > 0:
            # Calculate accuracy
            bin_correct = (scores[bin_mask] >= 0.5) == true_labels[bin_mask]
            bin_accuracy = bin_correct.mean()
        else:
            bin_accuracy = 0
        
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(bin_samples)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot bin accuracies
    plt.subplot(1, 2, 1)
    plt.bar(range(num_bins), bin_accuracies)
    plt.xlabel('Confidence Bin')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Confidence')
    plt.xticks(range(num_bins), [f'{lower:.2f}-{upper:.2f}' for lower, upper in zip(bin_edges[:-1], bin_edges[1:])], rotation=45)
    
    # Plot bin counts
    plt.subplot(1, 2, 2)
    plt.bar(range(num_bins), bin_counts)
    plt.xlabel('Confidence Bin')
    plt.ylabel('Count')
    plt.title('Sample Count per Confidence Bin')
    plt.xticks(range(num_bins), [f'{lower:.2f}-{upper:.2f}' for lower, upper in zip(bin_edges[:-1], bin_edges[1:])], rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def compare_error_patterns(model_reports, output_path=None):
    """
    Compare error patterns across models.
    
    Args:
        model_reports: Dictionary mapping model names to error reports
        output_path: Path to save the visualization
    """
    # Extract error rates
    models = list(model_reports.keys())
    error_rates = [report['error_rate'] for report in model_reports.values()]
    fp_rates = [report['fp_rate'] for report in model_reports.values()]
    fn_rates = [report['fn_rate'] for report in model_reports.values()]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, error_rates, width, label='Overall Error Rate')
    plt.bar(x, fp_rates, width, label='False Positive Rate')
    plt.bar(x + width, fn_rates, width, label='False Negative Rate')
    
    # Set labels
    plt.xlabel('Model')
    plt.ylabel('Rate')
    plt.title('Error Patterns Across Models')
    plt.xticks(x, models)
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()

def generate_error_report(model_name, task_name, task_results, task_data, output_dir=None):
    """
    Generate a comprehensive error analysis report.
    
    Args:
        model_name: Name of the model
        task_name: Name of the task
        task_results: Results from task evaluation
        task_data: Original task data
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Error analysis report
    """
    # Analyze errors
    report = analyze_errors(model_name, task_results, task_data)
    
    # Create visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize error distribution
        error_dist_path = os.path.join(output_dir, f"{model_name}_{task_name}_error_distribution.png")
        visualize_error_distribution(
            report['error_categories'],
            output_path=error_dist_path,
            title=f"{model_name} Error Distribution on {task_name}"
        )
        
        # Visualize confidence vs correctness
        if 'similarities' in task_results:
            scores = task_results['similarities']
        elif 'enhanced_similarities' in task_results:
            scores = task_results['enhanced_similarities']
        elif 'combined_impacts' in task_results:
            scores = task_results['combined_impacts']
        elif 'embedding_impacts' in task_results:
            scores = task_results['embedding_impacts']
        elif 'vulnerability_scores' in task_results:
            scores = task_results['vulnerability_scores']
        else:
            scores = None
        
        if scores is not None:
            confidence_path = os.path.join(output_dir, f"{model_name}_{task_name}_confidence_vs_correctness.png")
            analyze_confidence_vs_correctness(
                scores,
                task_results['true_labels'],
                output_path=confidence_path
            )
    
    return report