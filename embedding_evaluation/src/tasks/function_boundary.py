"""
Task for evaluating function boundary detection in binary code.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from .base import BaseTask

class FunctionBoundaryTask(BaseTask):
    """Task for evaluating function boundary detection in binary code."""
    
    def __init__(self, threshold=0.7):
        """
        Initialize the function boundary detection task.
        
        Args:
            threshold: Threshold for boundary detection
        """
        super().__init__(
            name="Function Boundary Detection",
            description="Detect function boundaries in binaries using instruction embeddings"
        )
        self.threshold = threshold
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on function boundary detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with:
                - 'instruction_sequences': List of instruction sequences
                - 'boundaries': List of lists of boundary indices
            
        Returns:
            dict: Evaluation results
        """
        instruction_sequences = test_data['instruction_sequences']
        true_boundaries = test_data['boundaries']
        
        all_predicted_boundaries = []
        boundary_scores = []
        
        for i, sequence in enumerate(instruction_sequences):
            # Get embeddings for each instruction
            instruction_embeddings = []
            for j in range(len(sequence)):
                # Get embedding for this instruction
                instr = sequence[j]
                vec = embedding_model.transform([instr])[0]
                instruction_embeddings.append(vec)
            
            instruction_embeddings = np.array(instruction_embeddings)
            
            # Detect boundaries as points of high semantic difference
            predicted, scores = self._detect_boundaries(instruction_embeddings)
            
            all_predicted_boundaries.append(predicted)
            boundary_scores.append(scores)
        
        return {
            'predicted_boundaries': all_predicted_boundaries,
            'boundary_scores': boundary_scores,
            'true_boundaries': true_boundaries
        }
    
    def _detect_boundaries(self, instruction_embeddings):
        """
        Detect function boundaries as points of high semantic difference.
        
        Args:
            instruction_embeddings: Embeddings for instruction sequence
            
        Returns:
            tuple: (predicted_boundaries, boundary_scores)
        """
        if len(instruction_embeddings) < 3:
            return [], []
        
        # Calculate semantic difference between adjacent instructions
        semantic_diffs = []
        for i in range(1, len(instruction_embeddings)):
            prev_emb = instruction_embeddings[i-1]
            curr_emb = instruction_embeddings[i]
            
            # Normalize embeddings
            prev_norm = np.linalg.norm(prev_emb)
            curr_norm = np.linalg.norm(curr_emb)
            
            if prev_norm == 0 or curr_norm == 0:
                diff = 0.0
            else:
                # Semantic difference = 1 - cosine similarity
                similarity = np.dot(prev_emb, curr_emb) / (prev_norm * curr_norm)
                diff = 1.0 - similarity
            
            semantic_diffs.append(diff)
        
        # Calculate rolling window average to smooth out noise
        window_size = 3
        smoothed_diffs = semantic_diffs.copy()
        
        for i in range(len(semantic_diffs)):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(semantic_diffs), i + window_size // 2 + 1)
            window = semantic_diffs[window_start:window_end]
            smoothed_diffs[i] = sum(window) / len(window)
        
        # Find local maxima above threshold
        boundary_indices = []
        for i in range(1, len(smoothed_diffs) - 1):
            if (smoothed_diffs[i] > smoothed_diffs[i-1] and 
                smoothed_diffs[i] > smoothed_diffs[i+1] and 
                smoothed_diffs[i] > self.threshold):
                boundary_indices.append(i + 1)  # +1 because diffs are between positions
        
        return boundary_indices, smoothed_diffs
    
    def score(self, results):
        """
        Calculate scores for function boundary detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predicted_boundaries = results['predicted_boundaries']
        true_boundaries = results['true_boundaries']
        
        # Calculate metrics with tolerance
        precision, recall, f1 = self._calculate_boundary_metrics(
            predicted_boundaries, true_boundaries, tolerance=3
        )
        
        # Calculate metrics with stricter tolerance
        precision_strict, recall_strict, f1_strict = self._calculate_boundary_metrics(
            predicted_boundaries, true_boundaries, tolerance=1
        )
        
        # Calculate average boundary score at true boundaries
        avg_score_at_boundary = self._calculate_avg_score_at_boundary(
            results['boundary_scores'], true_boundaries
        )
        
        # Calculate average boundary score at non-boundaries
        avg_score_at_non_boundary = self._calculate_avg_score_at_non_boundary(
            results['boundary_scores'], true_boundaries
        )
        
        # Calculate score separation
        score_separation = avg_score_at_boundary - avg_score_at_non_boundary
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_strict': precision_strict,
            'recall_strict': recall_strict,
            'f1_strict': f1_strict,
            'avg_score_at_boundary': avg_score_at_boundary,
            'avg_score_at_non_boundary': avg_score_at_non_boundary,
            'score_separation': score_separation,
            'threshold': self.threshold
        }
    
    def _calculate_boundary_metrics(self, predicted_boundaries, true_boundaries, tolerance=3):
        """
        Calculate boundary detection metrics with tolerance.
        
        Args:
            predicted_boundaries: List of lists of predicted boundary indices
            true_boundaries: List of lists of true boundary indices
            tolerance: Maximum distance to consider a match
            
        Returns:
            tuple: (precision, recall, f1)
        """
        # Count true positives, false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(len(predicted_boundaries)):
            # Get predictions and ground truth for this sequence
            pred = predicted_boundaries[i]
            true = true_boundaries[i]
            
            # Count matches
            matched_pred = set()
            matched_true = set()
            
            for p_idx in pred:
                # Find closest true boundary
                closest_true = None
                min_dist = float('inf')
                
                for t_idx in true:
                    dist = abs(p_idx - t_idx)
                    if dist < min_dist:
                        min_dist = dist
                        closest_true = t_idx
                
                # Check if within tolerance
                if closest_true is not None and min_dist <= tolerance:
                    matched_pred.add(p_idx)
                    matched_true.add(closest_true)
            
            # Update counts
            tp += len(matched_true)
            fp += len(pred) - len(matched_pred)
            fn += len(true) - len(matched_true)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _calculate_avg_score_at_boundary(self, boundary_scores, true_boundaries):
        """
        Calculate average boundary score at true boundaries.
        
        Args:
            boundary_scores: List of boundary scores for each sequence
            true_boundaries: List of lists of true boundary indices
            
        Returns:
            float: Average boundary score at true boundaries
        """
        scores = []
        
        for i in range(len(boundary_scores)):
            scores_i = boundary_scores[i]
            boundaries_i = true_boundaries[i]
            
            for boundary in boundaries_i:
                # Boundary index is 1-based in the sequence
                if boundary <= 0 or boundary >= len(scores_i) + 1:
                    continue
                
                # Convert to 0-based index for scores
                score_idx = boundary - 1
                
                if score_idx < len(scores_i):
                    scores.append(scores_i[score_idx])
        
        return np.mean(scores) if scores else 0
    
    def _calculate_avg_score_at_non_boundary(self, boundary_scores, true_boundaries):
        """
        Calculate average boundary score at non-boundary positions.
        
        Args:
            boundary_scores: List of boundary scores for each sequence
            true_boundaries: List of lists of true boundary indices
            
        Returns:
            float: Average boundary score at non-boundary positions
        """
        scores = []
        
        for i in range(len(boundary_scores)):
            scores_i = boundary_scores[i]
            boundaries_i = set(true_boundaries[i])
            
            for j in range(len(scores_i)):
                # Convert to 1-based index for boundaries
                pos = j + 1
                
                if pos not in boundaries_i:
                    scores.append(scores_i[j])
        
        return np.mean(scores) if scores else 0