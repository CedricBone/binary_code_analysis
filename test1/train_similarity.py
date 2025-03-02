"""
Training script for binary similarity model
"""
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from preprocessing import BinaryPreprocessor
from dataset import BinarySimilarityDataset, create_dataloader, similarity_collate_fn
from models import (
    SAFEEncoder, 
    GeminiEncoder, 
    HybridEncoder,
    SiameseSimilarityModel
)
from utils import (
    save_checkpoint, 
    load_checkpoint, 
    compute_metrics,
    setup_logging,
    AverageMeter
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train binary similarity model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="safe", choices=["safe", "gemini", "hybrid"],
                        help="Model type: safe, gemini, or hybrid")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden state dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attention_heads", type=int, default=8, help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for contrastive loss")
    parser.add_argument("--early_stopping", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    
    # Runtime arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device (cuda or cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    start_time = time.time()
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Train]")):
        # Move tensors to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], list) and all(isinstance(item, torch.Tensor) for item in batch[key]):
                batch[key] = [item.to(device) for item in batch[key]]
        
        # Handle graph data separately since we kept them as lists
        if model.encoder.__class__.__name__ in ["GeminiEncoder", "HybridEncoder"]:
            # Process each graph in the batch separately
            batch_size = len(batch['func1_node_features'])
            all_similarities = []
            all_embeddings = []
            
            for i in range(batch_size):
                # Extract single item from batch
                func1_data = {
                    'func_opcode_ids': batch['func1_opcode_ids'][i].unsqueeze(0),
                    'func_operand_ids': batch['func1_operand_ids'][i].unsqueeze(0),
                    'func_node_features': batch['func1_node_features'][i].unsqueeze(0),
                    'func_adj_matrix': batch['func1_adj_matrix'][i].unsqueeze(0)
                }
                
                func2_data = {
                    'func_opcode_ids': batch['func2_opcode_ids'][i].unsqueeze(0),
                    'func_operand_ids': batch['func2_operand_ids'][i].unsqueeze(0),
                    'func_node_features': batch['func2_node_features'][i].unsqueeze(0),
                    'func_adj_matrix': batch['func2_adj_matrix'][i].unsqueeze(0)
                }
                
                # Forward pass for this single item
                similarity, embeddings = model(func1_data, func2_data)
                all_similarities.append(similarity)
                all_embeddings.append(embeddings)
            
            # Combine results
            similarity = torch.cat(all_similarities)
            # embeddings now a list of tuples, but we don't use it for loss calculation
        else:
            # For sequence-only models, process the whole batch together
            func1_data = {
                'func_opcode_ids': batch['func1_opcode_ids'],
                'func_operand_ids': batch['func1_operand_ids']
            }
            
            func2_data = {
                'func_opcode_ids': batch['func2_opcode_ids'],
                'func_operand_ids': batch['func2_operand_ids']
            }
            
            # Forward pass
            similarity, embeddings = model(func1_data, func2_data)
        
        # Compute loss
        loss = model.compute_loss(similarity, batch['label'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = (similarity > 0.5).float()
        accuracy = (predictions == batch['label']).float().mean().item()
        
        # Update meters
        loss_meter.update(loss.item(), batch['label'].size(0))
        accuracy_meter.update(accuracy, batch['label'].size(0))
        
    elapsed_time = time.time() - start_time
    
    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_meter.avg,
        "elapsed_time": elapsed_time
    }

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Evaluate model on validation data"""
    model.eval()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    all_similarities = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Eval]"):
            # Move tensors to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], list) and all(isinstance(item, torch.Tensor) for item in batch[key]):
                    batch[key] = [item.to(device) for item in batch[key]]
            
            # Handle graph data separately since we kept them as lists
            if model.encoder.__class__.__name__ in ["GeminiEncoder", "HybridEncoder"]:
                # Process each graph in the batch separately
                batch_size = len(batch['func1_node_features'])
                batch_similarities = []
                
                for i in range(batch_size):
                    # Extract single item from batch
                    func1_data = {
                        'func_opcode_ids': batch['func1_opcode_ids'][i].unsqueeze(0),
                        'func_operand_ids': batch['func1_operand_ids'][i].unsqueeze(0),
                        'func_node_features': batch['func1_node_features'][i].unsqueeze(0),
                        'func_adj_matrix': batch['func1_adj_matrix'][i].unsqueeze(0)
                    }
                    
                    func2_data = {
                        'func_opcode_ids': batch['func2_opcode_ids'][i].unsqueeze(0),
                        'func_operand_ids': batch['func2_operand_ids'][i].unsqueeze(0),
                        'func_node_features': batch['func2_node_features'][i].unsqueeze(0),
                        'func_adj_matrix': batch['func2_adj_matrix'][i].unsqueeze(0)
                    }
                    
                    # Forward pass for this single item
                    similarity, _ = model(func1_data, func2_data)
                    batch_similarities.append(similarity)
                
                # Combine results
                similarity = torch.cat(batch_similarities)
            else:
                # For sequence-only models, process the whole batch together
                func1_data = {
                    'func_opcode_ids': batch['func1_opcode_ids'],
                    'func_operand_ids': batch['func1_operand_ids']
                }
                
                func2_data = {
                    'func_opcode_ids': batch['func2_opcode_ids'],
                    'func_operand_ids': batch['func2_operand_ids']
                }
                
                # Forward pass
                similarity, _ = model(func1_data, func2_data)
            
            # Compute loss
            loss = model.compute_loss(similarity, batch['label'])
            
            # Compute accuracy
            predictions = (similarity > 0.5).float()
            accuracy = (predictions == batch['label']).float().mean().item()
            
            # Update meters
            loss_meter.update(loss.item(), batch['label'].size(0))
            accuracy_meter.update(accuracy, batch['label'].size(0))
            
            # Store for metrics
            all_similarities.extend(similarity.cpu().tolist())
            all_labels.extend(batch['label'].cpu().tolist())
    
    elapsed_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(all_similarities, all_labels)
    metrics.update({
        "loss": loss_meter.avg,
        "accuracy": accuracy_meter.avg,
        "elapsed_time": elapsed_time
    })
    
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, "training.log"))
    logger.info(f"Arguments: {args}")
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = BinaryPreprocessor(
        max_seq_length=args.max_seq_length,
        use_cfg=(args.model_type in ["gemini", "hybrid"]),
        normalize_registers=True,
        normalize_addresses=True
    )
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}...")
    train_dataset = BinarySimilarityDataset(
        args.train_data,
        preprocessor,
        is_training=True
    )
    
    logger.info(f"Loading validation data from {args.val_data}...")
    val_dataset = BinarySimilarityDataset(
        args.val_data,
        preprocessor,
        is_training=False
    )
    
    # Create dataloaders with custom collate function
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        is_training=True,
        collate_fn=similarity_collate_fn,
        num_workers=args.num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        is_training=False,
        collate_fn=similarity_collate_fn,
        num_workers=args.num_workers
    )
    
    # Get vocabulary sizes
    opcode_vocab_size = len(preprocessor.opcode_vocab)
    operand_vocab_size = len(preprocessor.operand_vocab)
    logger.info(f"Opcode vocabulary size: {opcode_vocab_size}")
    logger.info(f"Operand vocabulary size: {operand_vocab_size}")
    
    # Save vocabularies
    preprocessor.save_vocabularies(args.output_dir)
    
    # Initialize model
    logger.info(f"Initializing {args.model_type.upper()} model...")
    
    if args.model_type == "safe":
        encoder = SAFEEncoder(
            opcode_vocab_size=opcode_vocab_size,
            operand_vocab_size=operand_vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attention_heads=args.attention_heads,
            dropout=args.dropout
        )
    elif args.model_type == "gemini":
        encoder = GeminiEncoder(
            node_feature_dim=5,  # Number of features per node
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model_type == "hybrid":
        encoder = HybridEncoder(
            opcode_vocab_size=opcode_vocab_size,
            operand_vocab_size=operand_vocab_size,
            embedding_dim=args.embedding_dim,
            node_feature_dim=5,  # Number of features per node
            hidden_dim=args.hidden_dim,
            num_lstm_layers=args.num_layers,
            num_gnn_layers=args.num_layers,
            attention_heads=args.attention_heads,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = SiameseSimilarityModel(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        margin=args.margin
    )
    
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float("inf")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", float("inf"))
    
    # Training loop
    logger.info("Starting training...")
    
    no_improvement = 0
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        logger.info(f"Train metrics: Loss={train_metrics['loss']:.4f}, "
                   f"Accuracy={train_metrics['accuracy']:.4f}, "
                   f"Time={train_metrics['elapsed_time']:.2f}s")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )
        
        logger.info(f"Validation metrics: Loss={val_metrics['loss']:.4f}, "
                   f"Accuracy={val_metrics['accuracy']:.4f}, "
                   f"AUC={val_metrics.get('auc', 0):.4f}, "
                   f"F1={val_metrics.get('f1', 0):.4f}, "
                   f"Precision={val_metrics.get('precision', 0):.4f}, "
                   f"Recall={val_metrics.get('recall', 0):.4f}, "
                   f"Time={val_metrics['elapsed_time']:.2f}s")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric=val_metrics['loss'],
            best_metric=best_metric,
            args=args,
            save_path=os.path.join(args.output_dir, f"checkpoint-{epoch}.pt"),
            is_best=val_metrics['loss'] < best_metric
        )
        
        # Update best metric
        if val_metrics['loss'] < best_metric:
            best_metric = val_metrics['loss']
            no_improvement = 0
        else:
            no_improvement += 1
            
        # Early stopping
        if no_improvement >= args.early_stopping:
            logger.info(f"No improvement for {args.early_stopping} epochs, stopping...")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_metric:.4f}")

if __name__ == "__main__":
    main()