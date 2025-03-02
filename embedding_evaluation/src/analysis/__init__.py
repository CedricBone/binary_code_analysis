"""
Analysis utilities for instruction embeddings.
"""

from .visualization import (
    visualize_embedding_space,
    visualize_similarity_matrix,
    visualize_task_performance,
    visualize_overall_comparison,
    visualize_distribution,
    visualize_confusion_matrix,
    visualize_embedding_evolution
)

from .error_analysis import (
    find_error_cases,
    analyze_errors,
    categorize_errors,
    visualize_error_distribution,
    analyze_confidence_vs_correctness,
    compare_error_patterns,
    generate_error_report
)

__all__ = [
    # Visualization utilities
    'visualize_embedding_space',
    'visualize_similarity_matrix',
    'visualize_task_performance',
    'visualize_overall_comparison',
    'visualize_distribution',
    'visualize_confusion_matrix',
    'visualize_embedding_evolution',
    
    # Error analysis utilities
    'find_error_cases',
    'analyze_errors',
    'categorize_errors',
    'visualize_error_distribution',
    'analyze_confidence_vs_correctness',
    'compare_error_patterns',
    'generate_error_report'
]