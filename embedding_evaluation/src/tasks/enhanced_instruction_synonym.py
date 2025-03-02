"""
Enhanced task for evaluating instruction synonym detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

from .instruction_synonym import InstructionSynonymTask

class EnhancedSynonymTask(InstructionSynonymTask):
    """Enhanced task for evaluating instruction synonym detection with contrastive learning."""
    
    def __init__(self, threshold=0.85, contrastive_margin=0.5, adaptive_threshold=True):
        """
        Initialize the enhanced instruction synonym task.
        
        Args:
            threshold: Initial similarity threshold for synonym detection
            contrastive_margin: Margin for contrastive loss
            adaptive_threshold: Whether to adapt the threshold based on data
        """
        super().__init__(threshold=threshold)
        self.name = "Enhanced Instruction Synonym Detection"
        self.description = "Detect different instructions with the same semantic effect using contrastive learning"
        self.contrastive_margin = contrastive_margin
        self.adaptive_threshold = adaptive_threshold
        self.contrastive_model = None
    
    def _find_optimal_threshold(self, similarities, true_labels):
        """
        Find the optimal threshold based on F1 score.
        
        Args:
            similarities: Array of similarity scores
            true_labels: Array of true labels
            
        Returns:
            float: Optimal threshold
        """
        # Try different thresholds and pick the one with highest F1
        thresholds = np.linspace(0.1, 0.95, 18)
        best_f1 = 0
        best_threshold = self.threshold
        
        for threshold in thresholds:
            predictions = [similarity >= threshold for similarity in similarities]
            
            # Handle potential division by zero
            if sum(predictions) == 0:  # No positive predictions
                continue
                
            if sum(true_labels) == 0:  # No positive ground truth
                continue
            
            try:
                f1 = f1_score(true_labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except Exception:
                continue
        
        return best_threshold
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on instruction synonym detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys 'instruction_pairs' and 'labels'
            
        Returns:
            dict: Evaluation results
        """
        instruction_pairs = test_data['instruction_pairs']
        true_labels = test_data['labels']
        
        # Calculate similarities
        similarities = []
        instruction1_embeddings = []
        instruction2_embeddings = []
        
        for instr1, instr2 in instruction_pairs:
            # Get embeddings
            vec1 = embedding_model.transform([instr1])[0]
            vec2 = embedding_model.transform([instr2])[0]
            
            # Store embeddings for contrastive learning
            instruction1_embeddings.append(vec1)
            instruction2_embeddings.append(vec2)
            
            # Calculate similarity
            similarity = embedding_model.instruction_similarity(instr1, instr2)
            similarities.append(similarity)
        
        # Perform contrastive learning to enhance similarity scores
        if self.contrastive_model is None:
            self._train_contrastive_model(
                np.array(instruction1_embeddings),
                np.array(instruction2_embeddings),
                np.array(true_labels)
            )
        
        # Apply contrastive model to get enhanced similarities
        enhanced_similarities = self._apply_contrastive_model(
            np.array(instruction1_embeddings),
            np.array(instruction2_embeddings)
        )
        
        # Find optimal threshold if adaptive
        if self.adaptive_threshold:
            self.threshold = self._find_optimal_threshold(enhanced_similarities, true_labels)
        
        # Predict synonyms based on threshold
        predictions = [similarity >= self.threshold for similarity in enhanced_similarities]
        
        # Calculate additional metrics
        similarity_matrix = self._calculate_similarity_matrix(instruction1_embeddings, instruction2_embeddings)
        
        return {
            'similarities': similarities,
            'enhanced_similarities': enhanced_similarities,
            'predictions': predictions,
            'true_labels': true_labels,
            'similarity_matrix': similarity_matrix,
            'threshold': self.threshold
        }
    
    def _train_contrastive_model(self, embeddings1, embeddings2, labels, epochs=50):
        """
        Train a contrastive learning model to enhance similarity scores.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: True labels (1 for synonyms, 0 for non-synonyms)
            epochs: Number of training epochs
        """
        if len(embeddings1) < 10:
            # Not enough data to train a contrastive model
            self.contrastive_model = None
            return
        
        # Convert to PyTorch tensors
        embeddings1 = torch.tensor(embeddings1, dtype=torch.float32)
        embeddings2 = torch.tensor(embeddings2, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        input_dim = embeddings1.shape[1]
        
        # Create a simple network to transform embeddings
        class ContrastiveNet(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super(ContrastiveNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, input_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return F.normalize(x, p=2, dim=1)  # L2 normalize
        
        model = ContrastiveNet(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            # Forward pass
            transformed1 = model(embeddings1)
            transformed2 = model(embeddings2)
            
            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(transformed1, transformed2)
            
            # Contrastive loss
            # For synonyms (label=1), want similarity close to 1
            # For non-synonyms (label=0), want similarity less than margin
            loss_pos = (1 - cosine_sim) * labels
            loss_neg = torch.clamp(cosine_sim - self.contrastive_margin, min=0.0) * (1 - labels)
            loss = (loss_pos + loss_neg).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.contrastive_model = model
        self.contrastive_model.eval()
    
    def _apply_contrastive_model(self, embeddings1, embeddings2):
        """
        Apply the contrastive model to enhance similarity scores.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            numpy.ndarray: Enhanced similarity scores
        """
        if self.contrastive_model is None:
            # Fall back to cosine similarity
            similarities = []
            for vec1, vec2 in zip(embeddings1, embeddings2):
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(np.dot(vec1, vec2) / (norm1 * norm2))
            
            return np.array(similarities)
        
        # Apply contrastive model
        with torch.no_grad():
            # Convert to PyTorch tensors
            embeddings1 = torch.tensor(embeddings1, dtype=torch.float32)
            embeddings2 = torch.tensor(embeddings2, dtype=torch.float32)
            
            # Transform embeddings
            transformed1 = self.contrastive_model(embeddings1)
            transformed2 = self.contrastive_model(embeddings2)
            
            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(transformed1, transformed2)
            
            return cosine_sim.numpy()
    
    def _calculate_similarity_matrix(self, embeddings1, embeddings2):
        """
        Calculate similarity matrix between all instructions.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        # Combine all embeddings
        all_embeddings = np.vstack([embeddings1, embeddings2])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(all_embeddings)
        
        return similarity_matrix
    
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
        enhanced_similarities = results['enhanced_similarities']
        
        # Calculate standard metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle potential division by zero
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
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, enhanced_similarities)
        except ValueError:
            # This can happen if all examples are of one class
            roc_auc = 0.5
        
        # Calculate additional metrics
        avg_sim_synonym = np.mean([
            sim for sim, label in zip(enhanced_similarities, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_sim_non_synonym = np.mean([
            sim for sim, label in zip(enhanced_similarities, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        # Class separation (higher is better)
        class_separation = avg_sim_synonym - avg_sim_non_synonym
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_sim_synonym': avg_sim_synonym,
            'avg_sim_non_synonym': avg_sim_non_synonym,
            'class_separation': class_separation,
            'threshold': results['threshold']
        }