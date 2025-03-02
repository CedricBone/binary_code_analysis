"""
Visualization utilities for binary code analysis
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Any, Optional

def visualize_cfg(graph: nx.DiGraph, output_path: Optional[str] = None):
    """
    Visualize a Control Flow Graph
    
    Args:
        graph: NetworkX DiGraph representing a CFG
        output_path: Path to save the visualization (if None, display instead)
    """
    plt.figure(figsize=(12, 8))
    
    # Get positions for nodes
    pos = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, 
                          node_color='lightblue', 
                          node_size=500, 
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, 
                          width=1.0, 
                          alpha=0.5, 
                          arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, 
                           font_size=10, 
                           font_family='sans-serif')
    
    plt.title("Control Flow Graph")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_similarity_matrix(matrix: np.ndarray, 
                               labels: List[str],
                               output_path: Optional[str] = None):
    """
    Visualize a similarity matrix as a heatmap
    
    Args:
        matrix: 2D numpy array of similarity scores
        labels: Labels for the rows and columns
        output_path: Path to save the visualization (if None, display instead)
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(matrix, 
                    xticklabels=labels, 
                    yticklabels=labels,
                    vmin=0, vmax=1, 
                    cmap='YlGnBu', 
                    annot=True, 
                    fmt='.2f')
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.title("Binary Function Similarity Matrix")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_analysis_results(results_path: str, 
                              output_dir: Optional[str] = None):
    """
    Visualize analysis results from a JSON file
    
    Args:
        results_path: Path to analysis results JSON file
        output_dir: Directory to save visualizations (if None, display instead)
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize similarity matrix if available
    if 'similarity' in results and 'matrix' in results['similarity']:
        matrix = np.array(results['similarity']['matrix'])
        labels = results['similarity']['function_names']
        
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, 'similarity_matrix.png')
            
        visualize_similarity_matrix(matrix, labels, output_path)
    
    # Visualize similar function pairs if available
    if 'similarity' in results and 'similar_pairs' in results['similarity']:
        pairs = results['similarity']['similar_pairs']
        
        if pairs:
            # Extract similarity scores and function pairs
            func_pairs = [(p['func1'], p['func2']) for p in pairs]
            similarity_scores = [p['similarity'] for p in pairs]
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(func_pairs))
            
            plt.barh(y_pos, similarity_scores, align='center', alpha=0.5)
            plt.yticks(y_pos, [f"{p[0]}\nvs\n{p[1]}" for p in func_pairs])
            plt.xlabel('Similarity Score')
            plt.title('Similar Function Pairs')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'similar_pairs.png'), 
                           dpi=300, 
                           bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    # Visualize descriptions if available
    if 'descriptions' in results:
        descriptions = results['descriptions']
        
        if descriptions:
            # Create bar plot of description lengths
            plt.figure(figsize=(10, 6))
            
            funcs = list(descriptions.keys())
            desc_lengths = [len(descriptions[f].split()) for f in funcs]
            
            plt.bar(funcs, desc_lengths, alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Description Length (words)')
            plt.title('Generated Description Lengths')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'description_lengths.png'), 
                           dpi=300, 
                           bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # Save descriptions to text file
            if output_dir:
                with open(os.path.join(output_dir, 'descriptions.txt'), 'w') as f:
                    for func_name, desc in descriptions.items():
                        f.write(f"Function: {func_name}\n")
                        f.write(f"Description: {desc}\n")
                        f.write("-" * 50 + "\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize binary code analysis results")
    parser.add_argument("--results", type=str, required=True, 
                       help="Path to analysis results JSON file")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    visualize_analysis_results(args.results, args.output_dir)
