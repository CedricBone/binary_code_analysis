"""
Task for evaluating semantic block equivalence detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base import BaseTask

class SemanticBlockTask(BaseTask):
    """Task for evaluating semantic block equivalence detection."""
    
    def __init__(self, threshold=0.75):
        """
        Initialize the semantic block equivalence task.
        
        Args:
            threshold: Similarity threshold for block equivalence detection
        """
        super().__init__(
            name="Semantic Block Equivalence",
            description="Detect different instruction sequences with the same semantic outcome"
        )
        self.threshold = threshold
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on semantic block equivalence detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys 'block_pairs' and 'labels'
                       where 'block_pairs' is a list of (block1, block2) tuples
                       and 'labels' is a list of boolean values indicating if the pair
                       of blocks has the same semantic outcome
            
        Returns:
            dict: Evaluation results
        """
        block_pairs = test_data['block_pairs']
        true_labels = test_data['labels']
        
        # Calculate block embedding similarities
        similarities = []
        
        for block1, block2 in block_pairs:
            # Get embeddings for each block
            vec1 = embedding_model.transform([block1])[0]
            vec2 = embedding_model.transform([block2])[0]
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                
            similarities.append(similarity)
        
        # Predict block equivalence based on threshold
        predictions = [similarity >= self.threshold for similarity in similarities]
        
        return {
            'similarities': similarities,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def score(self, results):
        """
        Calculate scores for semantic block equivalence detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        similarities = results['similarities']
        
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
        
        # Calculate additional block-specific metrics
        avg_similarity_equivalent = np.mean([
            sim for sim, label in zip(similarities, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_similarity_different = np.mean([
            sim for sim, label in zip(similarities, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        separation = avg_similarity_equivalent - avg_similarity_different
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_similarity_equivalent': avg_similarity_equivalent,
            'avg_similarity_different': avg_similarity_different,
            'separation': separation
        }