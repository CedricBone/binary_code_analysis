"""
Task for evaluating instruction synonym detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base import BaseTask

class InstructionSynonymTask(BaseTask):
    """Task for evaluating instruction synonym detection."""
    
    def __init__(self, threshold=0.8):
        """
        Initialize the instruction synonym task.
        
        Args:
            threshold: Similarity threshold for synonym detection
        """
        super().__init__(
            name="Instruction Synonym Detection",
            description="Detect different instructions with the same semantic effect"
        )
        self.threshold = threshold
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on instruction synonym detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys 'instruction_pairs' and 'labels'
                       where 'instruction_pairs' is a list of (instr1, instr2) tuples
                       and 'labels' is a list of boolean values indicating if the pair
                       represents synonymous instructions
            
        Returns:
            dict: Evaluation results
        """
        instruction_pairs = test_data['instruction_pairs']
        true_labels = test_data['labels']
        
        # Calculate similarities
        similarities = []
        for instr1, instr2 in instruction_pairs:
            similarity = embedding_model.instruction_similarity(instr1, instr2)
            similarities.append(similarity)
        
        # Predict synonyms based on threshold
        predictions = [similarity >= self.threshold for similarity in similarities]
        
        return {
            'similarities': similarities,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def score(self, results):
        """
        Calculate scores for instruction synonym detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle potential division by zero for edge cases
        if sum(predictions) == 0:  # No positive predictions
            precision = 0
        else:
            precision = precision_score(true_labels, predictions)
            
        if sum(true_labels) == 0:  # No positive ground truth
            recall = 0
        else:
            recall = recall_score(true_labels, predictions)
            
        if precision + recall == 0:  # Both precision and recall are 0
            f1 = 0
        else:
            f1 = f1_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }