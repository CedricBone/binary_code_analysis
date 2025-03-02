"""
Training script for binary code description generation
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
from typing import Dict, List, Tuple, Optional, Any

from preprocessing import BinaryPreprocessor
from dataset import BinaryDescriptionDataset, create_dataloader, description_collate_fn
from models import DescriptionGenerator
from utils import (
    save_checkpoint, 
    load_checkpoint, 
    compute_language_metrics,
    setup_logging,
    AverageMeter,
    build_idx_to_token,
    idx_to_tokens,
    filter_special_tokens
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train binary code description generator")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--max_desc_length", type=int, default=30, help="Maximum description length")
    
    # Model arguments
    parser.add_argument("--encoder_type", type=str, default="transformer", choices=["transformer", "safe"],
                        help="Encoder type: transformer or safe")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden state dimension")
    parser.add_argument("--encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
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
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    start_time = time.time()
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Train]")):
        # Move tensors to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], list) and all(isinstance(item, torch.Tensor) for item in batch[key]):
                batch[key] = [item.to(device) for item in batch[key]]
        
        # Handle graph data separately since we kept them as lists
        if model.encoder_type in ["gemini", "hybrid"]:
            # Process each sample in the batch separately
            batch_size = len(batch['func_node_features'])
            batch_loss = 0
            
            for i in range(batch_size):
                # Extract single item from batch
                opcode_ids = batch['func_opcode_ids'][i].unsqueeze(0)
                operand_ids = batch['func_operand_ids'][i].unsqueeze(0)
                desc_ids = batch['desc_token_ids'][i].unsqueeze(0)
                
                # Forward pass for this single item (with teacher forcing)
                logits = model(
                    opcode_ids=opcode_ids,
                    operand_ids=operand_ids,
                    desc_ids=desc_ids,
                    teacher_forcing=True
                )
                
                # Target for loss calculation (shift right by 1)
                target = desc_ids[:, 1:].contiguous()
                
                # Reshape logits and targets for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target.view(-1)
                
                # Compute loss (ignore padding)
                loss = criterion(logits_flat, target_flat)
                batch_loss += loss.item()
                
                # Backward pass for this single item
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
            
            # Update meter with average loss
            loss_meter.update(batch_loss / batch_size, batch_size)
            
        else:
            # For sequence-only models, process the whole batch together
            # Forward pass with teacher forcing
            logits = model(
                opcode_ids=batch['func_opcode_ids'],
                operand_ids=batch['func_operand_ids'],
                desc_ids=batch['desc_token_ids'],
                teacher_forcing=True
            )
            
            # Target for loss calculation (shift right by 1)
            target = batch['desc_token_ids'][:, 1:].contiguous()
            
            # Reshape logits and targets for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target.view(-1)
            
            # Compute loss (ignore padding)
            loss = criterion(logits_flat, target_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            # Update meter
            loss_meter.update(loss.item(), batch['desc_token_ids'].size(0))
        
    elapsed_time = time.time() - start_time
    
    return {
        "loss": loss_meter.avg,
        "elapsed_time": elapsed_time
    }

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_token: Dict[int, str],
    epoch: int
) -> Dict[str, float]:
    """Evaluate model on validation data"""
    model.eval()
    loss_meter = AverageMeter()
    
    all_predictions = []
    all_references = []
    
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
            if model.encoder_type in ["gemini", "hybrid"]:
                # Process each sample in the batch separately
                batch_size = len(batch['func_node_features'])
                batch_loss = 0
                
                for i in range(batch_size):
                    # Extract single item from batch
                    opcode_ids = batch['func_opcode_ids'][i].unsqueeze(0)
                    operand_ids = batch['func_operand_ids'][i].unsqueeze(0)
                    desc_ids = batch['desc_token_ids'][i].unsqueeze(0)
                    
                    # Forward pass for this single item (with teacher forcing)
                    logits = model(
                        opcode_ids=opcode_ids,
                        operand_ids=operand_ids,
                        desc_ids=desc_ids,
                        teacher_forcing=True
                    )
                    
                    # Target for loss calculation (shift right by 1)
                    target = desc_ids[:, 1:].contiguous()
                    
                    # Reshape logits and targets for loss calculation
                    logits_flat = logits.view(-1, logits.size(-1))
                    target_flat = target.view(-1)
                    
                    # Compute loss (ignore padding)
                    loss = criterion(logits_flat, target_flat)
                    batch_loss += loss.item()
                    
                    # Generate descriptions (no teacher forcing)
                    generated_ids = model.generate(
                        opcode_ids=opcode_ids,
                        operand_ids=operand_ids,
                        beam_size=1  # Greedy decoding
                    )
                    
                    # Convert IDs to tokens
                    gen_tokens = idx_to_tokens(generated_ids[0], idx_to_token)
                    gen_tokens = filter_special_tokens(gen_tokens)
                    all_predictions.append(gen_tokens)
                    
                    # Extract reference description
                    tokens = batch['raw_description'][i].lower().split()
                    all_references.append(tokens)
                
                # Update meter with average loss
                loss_meter.update(batch_loss / batch_size, batch_size)
                
            else:
                # For sequence-only models, process the whole batch together
                # Forward pass for loss calculation (teacher forcing)
                logits = model(
                    opcode_ids=batch['func_opcode_ids'],
                    operand_ids=batch['func_operand_ids'],
                    desc_ids=batch['desc_token_ids'],
                    teacher_forcing=True
                )
                
                # Target for loss calculation (shift right by 1)
                target = batch['desc_token_ids'][:, 1:].contiguous()
                
                # Reshape logits and targets for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target.view(-1)
                
                # Compute loss (ignore padding)
                loss = criterion(logits_flat, target_flat)
                loss_meter.update(loss.item(), batch['desc_token_ids'].size(0))
                
                # Generate descriptions (no teacher forcing)
                generated_ids = model.generate(
                    opcode_ids=batch['func_opcode_ids'],
                    operand_ids=batch['func_operand_ids'],
                    beam_size=1  # Greedy decoding
                )
                
                # Convert IDs to tokens
                for gen_ids in generated_ids:
                    gen_tokens = idx_to_tokens(gen_ids, idx_to_token)
                    gen_tokens = filter_special_tokens(gen_tokens)
                    all_predictions.append(gen_tokens)
                
                # Extract reference descriptions
                for desc in batch['raw_description']:
                    tokens = desc.lower().split()
                    all_references.append(tokens)
    
    elapsed_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_language_metrics(all_predictions, all_references, metric_type="all")
    metrics.update({
        "loss": loss_meter.avg,
        "elapsed_time": elapsed_time
    })
    
    # Log sample generations
    log_samples = min(5, len(all_predictions))
    for i in range(log_samples):
        pred = " ".join(all_predictions[i])
        ref = " ".join(all_references[i])
        metrics[f"sample_{i+1}"] = {
            "prediction": pred,
            "reference": ref
        }
    
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
        max_description_length=args.max_desc_length,
        use_cfg=(args.encoder_type in ["gemini", "hybrid"]),
        normalize_registers=True,
        normalize_addresses=True
    )
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}...")
    train_dataset = BinaryDescriptionDataset(
        args.train_data,
        preprocessor,
        is_training=True
    )
    
    logger.info(f"Loading validation data from {args.val_data}...")
    val_dataset = BinaryDescriptionDataset(
        args.val_data,
        preprocessor,
        is_training=False
    )
    
    # Create dataloaders with custom collate function
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        is_training=True,
        collate_fn=description_collate_fn,
        num_workers=args.num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        is_training=False,
        collate_fn=description_collate_fn,
        num_workers=args.num_workers
    )
    
    # Get vocabulary sizes
    opcode_vocab_size = len(preprocessor.opcode_vocab)
    operand_vocab_size = len(preprocessor.operand_vocab)
    desc_vocab_size = len(preprocessor.description_vocab)
    logger.info(f"Opcode vocabulary size: {opcode_vocab_size}")
    logger.info(f"Operand vocabulary size: {operand_vocab_size}")
    logger.info(f"Description vocabulary size: {desc_vocab_size}")
    
    # Save vocabularies
    preprocessor.save_vocabularies(args.output_dir)
    
    # Build index to token mapping for description vocabulary
    idx_to_token_map = build_idx_to_token(preprocessor.description_vocab)
    
    # Initialize model
    logger.info(f"Initializing description generator with {args.encoder_type} encoder...")
    
    model = DescriptionGenerator(
        opcode_vocab_size=opcode_vocab_size,
        operand_vocab_size=operand_vocab_size,
        desc_vocab_size=desc_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        max_desc_length=args.max_desc_length,
        encoder_type=args.encoder_type
    )
    
    model.to(device)
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
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
            criterion=criterion,
            device=device,
            clip_grad=args.clip_grad,
            epoch=epoch
        )
        
        logger.info(f"Train metrics: Loss={train_metrics['loss']:.4f}, "
                   f"Time={train_metrics['elapsed_time']:.2f}s")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            idx_to_token=idx_to_token_map,
            epoch=epoch
        )
        
        logger.info(f"Validation metrics: Loss={val_metrics['loss']:.4f}, "
                   f"BLEU-1={val_metrics.get('bleu-1', 0):.4f}, "
                   f"BLEU-4={val_metrics.get('bleu-4', 0):.4f}, "
                   f"ROUGE-L F1={val_metrics.get('rouge-l-f', 0):.4f}, "
                   f"Time={val_metrics['elapsed_time']:.2f}s")
        
        # Log sample generations
        for i in range(1, 6):
            sample_key = f"sample_{i}"
            if sample_key in val_metrics:
                pred = val_metrics[sample_key]["prediction"]
                ref = val_metrics[sample_key]["reference"]
                logger.info(f"Sample {i}:")
                logger.info(f"  Prediction: {pred}")
                logger.info(f"  Reference: {ref}")
        
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