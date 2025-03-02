#!/usr/bin/env python
"""
Main script for running experiments with the instruction embedding evaluation framework.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Add src to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.embeddings import get_embedding_model
from src.tasks import get_task
from src.data import SyntheticDataGenerator, BinaryPreprocessor
from src.evaluation import EvaluationScorer

def setup_logging(log_dir):
    """Set up logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"experiment_{timestamp}.log")),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("experiment")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run instruction embedding evaluation experiments"
    )
    
    parser.add_argument(
        "--embeddings", 
        type=str, 
        default="word2vec,palmtree",
        help="Comma-separated list of embedding models to evaluate"
    )
    
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="synonym,block,dead_code",
        help="Comma-separated list of tasks to run"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Directory for data storage"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Directory for results storage"
    )
    
    parser.add_argument(
        "--use_real_data", 
        action="store_true",
        help="Use real binary data instead of synthetic data"
    )
    
    parser.add_argument(
        "--binary_dir", 
        type=str, 
        default="data/raw/binaries",
        help="Directory containing binary files (if use_real_data is set)"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=500,
        help="Number of samples per task (for synthetic data)"
    )
    
    parser.add_argument(
        "--embedding_dim", 
        type=int, 
        default=100,
        help="Dimension of embedding vectors"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def load_or_generate_data(args, logger):
    """Load or generate evaluation data."""
    if args.use_real_data:
        logger.info(f"Using real binary data from {args.binary_dir}")
        return load_real_data(args, logger)
    else:
        logger.info(f"Generating synthetic data with {args.num_samples} samples per task")
        return generate_synthetic_data(args, logger)

def load_real_data(args, logger):
    """Load real binary data for evaluation."""
    preprocessor = BinaryPreprocessor()
    
    # Load or process binary files
    processed_file = os.path.join(args.data_dir, "processed", "binary_data.json")
    
    if os.path.exists(processed_file):
        logger.info(f"Loading processed data from {processed_file}")
        data = preprocessor.load_processed_data(processed_file)
    else:
        logger.info(f"Processing binary files from {args.binary_dir}")
        data = preprocessor.process_directory(
            args.binary_dir, 
            output_path=processed_file
        )
    
    logger.info(f"Loaded {len(data)} binary functions")
    
    # Create instruction sequences for training
    instruction_sequences = []
    for func in data:
        instruction_sequences.append(func['tokens'])
    
    # Extract instructions for synonym detection
    instructions = []
    for seq in instruction_sequences:
        instructions.extend(seq)
    
    logger.info(f"Extracted {len(instructions)} unique instructions")
    
    # Create evaluation data
    # Note: For real data, we don't have ground truth, so we'll create synthetic test data
    generator = SyntheticDataGenerator(seed=args.seed)
    
    task_data = {}
    for task_name in args.tasks.split(","):
        if task_name == "synonym":
            task_data[task_name] = generator.generate_instruction_synonym_data(args.num_samples)
        elif task_name == "block":
            task_data[task_name] = generator.generate_semantic_block_data(args.num_samples)
        elif task_name == "dead_code":
            task_data[task_name] = generator.generate_dead_code_data(args.num_samples)
    
    return {
        'instruction_sequences': instruction_sequences,
        'task_data': task_data
    }

def generate_synthetic_data(args, logger):
    """Generate synthetic data for evaluation."""
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate instruction sequences for training
    num_sequences = 1000
    sequence_length = 20
    
    logger.info(f"Generating {num_sequences} synthetic instruction sequences")
    instruction_sequences = [
        generator.generate_instruction_sequence(sequence_length)
        for _ in range(num_sequences)
    ]
    
    # Generate task data
    logger.info(f"Generating task data with {args.num_samples} samples per task")
    task_data = generator.generate_all_task_data(args.num_samples)
    
    return {
        'instruction_sequences': instruction_sequences,
        'task_data': task_data
    }

def train_embedding_models(embedding_models, data, logger):
    """Train embedding models on instruction sequences."""
    trained_models = {}
    
    for name, model in embedding_models.items():
        logger.info(f"Training {name} embedding model...")
        model.fit(data['instruction_sequences'])
        trained_models[name] = model
        logger.info(f"Finished training {name}")
    
    return trained_models

def run_tasks(tasks, embedding_models, data, logger):
    """Run evaluation tasks on embedding models."""
    scorer = EvaluationScorer()
    
    for model_name, model in embedding_models.items():
        logger.info(f"Evaluating {model_name}...")
        
        for task_name, task in tasks.items():
            logger.info(f"Running {task.name} task...")
            
            # Get task-specific data
            task_data = data['task_data'][task_name]
            
            # Run evaluation
            result = task.run(model, task_data)
            
            # Add result to scorer
            scorer.add_task_result(model_name, task.name, result)
            
            logger.info(f"Completed {task.name} task")
        
        logger.info(f"Finished evaluating {model_name}")
    
    return scorer

def main():
    """Main function for running experiments."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(os.path.join(args.results_dir, "logs"))
    logger.info("Starting experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create directories
    os.makedirs(os.path.join(args.data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "processed"), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load or generate data
    logger.info("Loading/generating data...")
    data = load_or_generate_data(args, logger)
    
    # Initialize embedding models
    logger.info("Initializing embedding models...")
    embedding_models = {}
    for model_name in args.embeddings.split(","):
        embedding_models[model_name] = get_embedding_model(
            model_name, 
            embedding_dim=args.embedding_dim
        )
    
    # Initialize tasks
    logger.info("Initializing evaluation tasks...")
    tasks = {}
    for task_name in args.tasks.split(","):
        tasks[task_name] = get_task(task_name)
    
    # Train embedding models
    logger.info("Training embedding models...")
    trained_models = train_embedding_models(embedding_models, data, logger)
    
    # Run evaluation tasks
    logger.info("Running evaluation tasks...")
    scorer = run_tasks(tasks, trained_models, data, logger)
    
    # Save results
    logger.info("Saving results...")
    scorer.save_results(args.results_dir)
    
    # Print summary
    best_model, best_score = scorer.get_best_model()
    logger.info(f"Best performing model: {best_model} with score {best_score:.4f}")
    
    print("\n" + scorer.format_results())
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()