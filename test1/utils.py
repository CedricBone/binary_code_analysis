"""
Utility functions for training and evaluation
"""
import os
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging to file and console
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    best_metric: float,
    args: Any,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metric: Current metric value
        best_metric: Best metric value so far
        args: Training arguments
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
        "best_metric": best_metric,
        "args": vars(args) if hasattr(args, "__dict__") else args
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # If this is the best checkpoint, save a copy
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
        torch.save(checkpoint, best_path)

def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None
) -> Dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into
        map_location: Device to map tensors to
        
    Returns:
        Checkpoint dictionary
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state dict
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state dict
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint

def compute_metrics(similarities: List[float], labels: List[int]) -> Dict[str, float]:
    """
    Compute evaluation metrics for binary similarity
    
    Args:
        similarities: List of similarity scores
        labels: List of binary labels (1 for similar, 0 for dissimilar)
        
    Returns:
        Dictionary of metrics
    """
    # Convert lists to numpy arrays
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Compute ROC AUC
    auc_score = roc_auc_score(labels, similarities)
    
    # Compute precision-recall curve and AUC
    precision, recall, thresholds = precision_recall_curve(labels, similarities)
    pr_auc = auc(recall, precision)
    
    # Convert similarities to binary predictions using 0.5 as threshold
    predictions = (similarities > 0.5).astype(int)
    
    # Compute F1 score
    f1 = f1_score(labels, predictions)
    
    # Compute precision and recall
    precision_val = precision_score(labels, predictions)
    recall_val = recall_score(labels, predictions)
    
    return {
        "auc": auc_score,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision_val,
        "recall": recall_val
    }

def compute_language_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
    metric_type: str = "bleu"
) -> Dict[str, float]:
    """
    Compute evaluation metrics for language generation
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
        metric_type: Type of metric to compute
        
    Returns:
        Dictionary of metrics
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from rouge import Rouge
    except ImportError:
        raise ImportError("Please install nltk and rouge: pip install nltk rouge")
    
    metrics = {}
    
    if metric_type in ["bleu", "all"]:
        # Compute BLEU score
        smoothing = SmoothingFunction().method1
        
        # Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4
        for n in range(1, 5):
            bleu_n = corpus_bleu(
                list_of_references=[[r] for r in references],
                hypotheses=predictions,
                weights=tuple([1.0 / n] * n),
                smoothing_function=smoothing
            )
            metrics[f"bleu-{n}"] = bleu_n
    
    if metric_type in ["rouge", "all"]:
        # Compute ROUGE score
        rouge = Rouge()
        
        # Convert token lists to strings
        pred_texts = [" ".join(p) for p in predictions]
        ref_texts = [" ".join(r) for r in references]
        
        # Handle empty predictions
        for i in range(len(pred_texts)):
            if not pred_texts[i]:
                pred_texts[i] = "empty"
        
        try:
            rouge_scores = rouge.get_scores(pred_texts, ref_texts, avg=True)
            
            # Add ROUGE scores to metrics
            for rouge_type, scores in rouge_scores.items():
                for score_type, value in scores.items():
                    metrics[f"{rouge_type}-{score_type}"] = value
        except Exception as e:
            print(f"Error computing ROUGE score: {e}")
            # Add default values if ROUGE fails
            for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
                for score_type in ["f", "p", "r"]:
                    metrics[f"{rouge_type}-{score_type}"] = 0.0
    
    return metrics

def idx_to_tokens(idx_list: List[int], idx_to_token: Dict[int, str]) -> List[str]:
    """
    Convert token indices to tokens
    
    Args:
        idx_list: List of token indices
        idx_to_token: Dictionary mapping indices to tokens
        
    Returns:
        List of tokens
    """
    return [idx_to_token.get(idx, "<UNK>") for idx in idx_list]

def build_idx_to_token(token_to_idx: Dict[str, int]) -> Dict[int, str]:
    """
    Build a mapping from token indices to tokens
    
    Args:
        token_to_idx: Dictionary mapping tokens to indices
        
    Returns:
        Dictionary mapping indices to tokens
    """
    return {idx: token for token, idx in token_to_idx.items()}

def filter_special_tokens(tokens: List[str], 
                          special_tokens: List[str] = ["<PAD>", "<UNK>", "<START>", "<END>"]) -> List[str]:
    """
    Filter out special tokens from a list of tokens
    
    Args:
        tokens: List of tokens
        special_tokens: List of special tokens to remove
        
    Returns:
        Filtered list of tokens
    """
    return [token for token in tokens if token not in special_tokens]
