"""
Task for evaluating dead code detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .base import BaseTask

class DeadCodeTask(BaseTask):
    """Task for evaluating dead code detection."""
    
    def __init__(self, impact_threshold=0.1):
        """
        Initialize the dead code detection task.
        
        Args:
            impact_threshold: Threshold for semantic impact (below which code is considered "dead")
        """
        super().__init__(
            name="Dead Code Detection",
            description="Detect instructions with no semantic impact on program behavior"
        )
        self.impact_threshold = impact_threshold
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on dead code detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys:
                       - 'code_blocks': List of instruction blocks
                       - 'target_indices': List of indices of instructions to evaluate
                       - 'labels': List of boolean values indicating if the instruction at 
                                  the corresponding index is dead code
            
        Returns:
            dict: Evaluation results
        """
        code_blocks = test_data['code_blocks']
        target_indices = test_data['target_indices']
        true_labels = test_data['labels']
        
        impact_scores = []
        for block, idx in zip(code_blocks, target_indices):
            # Calculate semantic impact by comparing block with and without the instruction
            block_with = list(block)  # Create a copy
            removed_instr = block_with.pop(idx)
            
            # Skip if block is now empty
            if not block_with:
                # Edge case: if removing the instruction makes the block empty,
                # it was likely not dead code
                impact_scores.append(1.0)
                continue
            
            # Get embeddings
            vec_original = embedding_model.transform([block])[0]
            vec_without = embedding_model.transform([block_with])[0]
            
            # Calculate cosine similarity
            norm_orig = np.linalg.norm(vec_original)
            norm_without = np.linalg.norm(vec_without)
            
            if norm_orig == 0 or norm_without == 0:
                similarity = 0.0
            else:
                similarity = np.dot(vec_original, vec_without) / (norm_orig * norm_without)
            
            # Impact is inverse of similarity: high similarity means low impact
            impact = 1.0 - similarity
            impact_scores.append(impact)
        
        # Predict dead code based on impact threshold
        predictions = [impact < self.impact_threshold for impact in impact_scores]
        
        return {
            'impact_scores': impact_scores,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def score(self, results):
        """
        Calculate scores for dead code detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        impact_scores = results['impact_scores']
        
        # Calculate standard metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle potential division by zero
        if sum(predictions) == 0:
            precision = 0
        else:
            precision = precision_score(true_labels, predictions)
            
        if sum(true_labels) == 0:
            recall = 0
        else:
            recall = recall_score(true_labels, predictions)
            
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = f1_score(true_labels, predictions)
        
        # Calculate ROC AUC (using impact scores as the decision function)
        # Invert impact scores since lower impact means higher probability of dead code
        decision_scores = [1.0 - score for score in impact_scores]
        
        try:
            roc_auc = roc_auc_score(true_labels, decision_scores)
        except ValueError:
            # This can happen if all examples are of one class
            roc_auc = 0.5
        
        # Calculate additional dead code-specific metrics
        avg_impact_dead = np.mean([
            score for score, label in zip(impact_scores, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_impact_live = np.mean([
            score for score, label in zip(impact_scores, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        impact_separation = avg_impact_live - avg_impact_dead
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_impact_dead': avg_impact_dead,
            'avg_impact_live': avg_impact_live,
            'impact_separation': impact_separation
        }