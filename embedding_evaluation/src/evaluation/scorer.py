"""
Scoring utilities for evaluation tasks.
"""

import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

class EvaluationScorer:
    """Score aggregator for evaluation tasks."""
    
    def __init__(self):
        """Initialize the evaluation scorer."""
        self.results = {}
    
    def add_task_result(self, embedding_name, task_name, result):
        """
        Add a task result.
        
        Args:
            embedding_name: Name of the embedding model
            task_name: Name of the task
            result: Result from running the task
        """
        if embedding_name not in self.results:
            self.results[embedding_name] = {}
        
        self.results[embedding_name][task_name] = result
    
    def calculate_overall_scores(self):
        """
        Calculate overall scores for each embedding model.
        
        Returns:
            dict: Dictionary of overall scores
        """
        overall_scores = {}
        
        for embedding_name, tasks in self.results.items():
            # Initialize scores
            overall_scores[embedding_name] = {
                'average_accuracy': 0,
                'average_f1': 0,
                'weighted_score': 0,
                'task_scores': {}
            }
            
            # Initialize counters
            num_tasks = 0
            total_accuracy = 0
            total_f1 = 0
            
            # Process each task
            for task_name, result in tasks.items():
                scores = result['scores']
                
                # Store individual task scores
                overall_scores[embedding_name]['task_scores'][task_name] = scores
                
                # Accumulate metrics
                if 'accuracy' in scores:
                    total_accuracy += scores['accuracy']
                    num_tasks += 1
                
                if 'f1' in scores:
                    total_f1 += scores['f1']
                
                # Task-specific weighted scoring
                if task_name == 'Instruction Synonym Detection':
                    weight = 0.35
                    task_score = scores['accuracy'] * 0.5 + scores['f1'] * 0.5
                elif task_name == 'Semantic Block Equivalence':
                    weight = 0.4
                    task_score = scores['accuracy'] * 0.3 + scores['f1'] * 0.3 + scores['separation'] * 0.4
                elif task_name == 'Dead Code Detection':
                    weight = 0.25
                    task_score = scores['accuracy'] * 0.3 + scores['f1'] * 0.3 + scores['roc_auc'] * 0.4
                else:
                    weight = 0.0
                    task_score = 0.0
                
                overall_scores[embedding_name]['weighted_score'] += task_score * weight
            
            # Calculate averages
            if num_tasks > 0:
                overall_scores[embedding_name]['average_accuracy'] = total_accuracy / num_tasks
                overall_scores[embedding_name]['average_f1'] = total_f1 / num_tasks
        
        return overall_scores
    
    def get_best_model(self):
        """
        Get the best performing model.
        
        Returns:
            tuple: (model_name, score)
        """
        overall_scores = self.calculate_overall_scores()
        
        if not overall_scores:
            return None, 0
        
        best_model = max(overall_scores.items(), 
                          key=lambda x: x[1]['weighted_score'])
        
        return best_model[0], best_model[1]['weighted_score']
    
    def format_results(self):
        """
        Format results for display.
        
        Returns:
            str: Formatted results
        """
        overall_scores = self.calculate_overall_scores()
        
        if not overall_scores:
            return "No results available."
        
        # Sort models by weighted score
        sorted_models = sorted(
            overall_scores.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        # Build formatted output
        lines = ["# Instruction Embedding Evaluation Results", ""]
        
        # Overall ranking
        lines.append("## Overall Ranking")
        lines.append("| Rank | Model | Weighted Score | Avg Accuracy | Avg F1 |")
        lines.append("|------|-------|---------------|--------------|--------|")
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            lines.append(f"| {i} | {model_name} | {scores['weighted_score']:.4f} | {scores['average_accuracy']:.4f} | {scores['average_f1']:.4f} |")
        
        lines.append("")
        
        # Task-specific results
        for task_name in self.results[sorted_models[0][0]].keys():
            lines.append(f"## {task_name} Results")
            
            # Create table header based on available metrics
            metrics = set()
            for model, tasks in self.results.items():
                if task_name in tasks:
                    metrics.update(tasks[task_name]['scores'].keys())
            
            # Sort metrics for consistent output
            sorted_metrics = sorted(metrics)
            
            # Table header
            header = "| Model | " + " | ".join(sorted_metrics) + " |"
            lines.append(header)
            
            # Table separator
            separator = "|------|" + "|".join(["------" for _ in sorted_metrics]) + "|"
            lines.append(separator)
            
            # Model results
            for model_name, _ in sorted_models:
                if task_name in self.results[model_name]:
                    scores = self.results[model_name][task_name]['scores']
                    row = f"| {model_name} | "
                    row += " | ".join([f"{scores.get(metric, 'N/A'):.4f}" 
                                       if isinstance(scores.get(metric), (int, float)) 
                                       else str(scores.get(metric, 'N/A')) 
                                       for metric in sorted_metrics])
                    row += " |"
                    lines.append(row)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_results(self, output_dir):
        """
        Save results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            else:
                return obj
        
        # Convert and save raw results
        serializable_results = convert_numpy_types(self.results)
        with open(os.path.join(output_dir, f"results_{timestamp}.json"), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save overall scores
        overall_scores = convert_numpy_types(self.calculate_overall_scores())
        with open(os.path.join(output_dir, f"scores_{timestamp}.json"), 'w') as f:
            json.dump(overall_scores, f, indent=2)
        
        # Save formatted results
        formatted_results = self.format_results()
        with open(os.path.join(output_dir, f"report_{timestamp}.md"), 'w') as f:
            f.write(formatted_results)