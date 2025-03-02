"""
Base class for evaluation tasks.
"""

from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Base class for all evaluation tasks."""
    
    def __init__(self, name, description=""):
        """
        Initialize the evaluation task.
        
        Args:
            name: Name of the task
            description: Description of the task
        """
        self.name = name
        self.description = description
        self.metrics = {}
    
    @abstractmethod
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on the task.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Test data for the task
            
        Returns:
            dict: Evaluation results
        """
        pass
    
    @abstractmethod
    def score(self, results):
        """
        Calculate scores based on evaluation results.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        pass
    
    def run(self, embedding_model, test_data):
        """
        Run the full evaluation.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Test data for the task
            
        Returns:
            dict: Combined results and scores
        """
        results = self.evaluate(embedding_model, test_data)
        scores = self.score(results)
        
        return {
            'task': self.name,
            'description': self.description,
            'results': results,
            'scores': scores
        }