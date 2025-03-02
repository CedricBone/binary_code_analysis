#!/usr/bin/env python
"""
Enhanced script for running experiments with the instruction embedding evaluation framework.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import random
import torch
from datetime import datetime
from tqdm import tqdm

# Add src to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.embeddings import get_embedding_model
from src.tasks import get_task
from src.data import SyntheticDataGenerator, RealBinaryLoader, CrossArchitectureProcessor
from src.evaluation import EvaluationScorer
from src.analysis import (
    visualize_embedding_space,
    visualize_task_performance,
    visualize_overall_comparison,
    visualize_distribution,
    visualize_confusion_matrix,
    generate_error_report
)
from src.config import ExperimentConfig

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
    
    return logging.getLogger("experiment"), timestamp

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run instruction embedding evaluation experiments"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--data_type", 
        type=str, 
        choices=["synthetic", "real", "cross_arch"],
        help="Type of data to use"
    )
    
    parser.add_argument(
        "--embeddings", 
        type=str, 
        help="Comma-separated list of embedding models to evaluate"
    )
    
    parser.add_argument(
        "--tasks", 
        type=str, 
        help="Comma-separated list of tasks to run"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        help="Directory for results storage"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--embedding_dim", 
        type=int, 
        help="Dimension of embedding vectors"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (smaller dataset, fewer iterations)"
    )
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """
    Update configuration from command line arguments.
    
    Args:
        config: ExperimentConfig instance
        args: Command line arguments
    """
    # Update only if argument is provided
    if args.data_type:
        config.set(args.data_type, "data", "type")
    
    if args.embeddings:
        config.set(args.embeddings.split(","), "embeddings", "models")
    
    if args.tasks:
        # Parse tasks and assign to appropriate categories
        tasks = args.tasks.split(",")
        basic_tasks = [t for t in tasks if t in ["synonym", "block", "dead_code"]]
        enhanced_tasks = [t for t in tasks if t.startswith("enhanced_")]
        new_tasks = [t for t in tasks if t in ["function_boundary", "vulnerability"]]
        
        if basic_tasks:
            config.set(basic_tasks, "tasks", "basic")
        if enhanced_tasks:
            config.set(enhanced_tasks, "tasks", "enhanced")
        if new_tasks:
            config.set(new_tasks, "tasks", "new")
    
    if args.results_dir:
        config.set(args.results_dir, "output", "results_dir")
    
    if args.seed:
        config.set(args.seed, "experiment", "seed")
    
    if args.embedding_dim:
        config.set(args.embedding_dim, "embeddings", "embedding_dim")
    
    if args.debug:
        # Set debug mode configurations
        config.set(50, "data", "synthetic", "num_samples")
        config.set(10, "embeddings", "params", "bert", "epochs")
        config.set(10, "embeddings", "params", "palmtree", "epochs")
        config.set(10, "embeddings", "params", "graph", "epochs")

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_or_generate_data(config, logger):
    """
    Load or generate evaluation data based on configuration.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        dict: Dictionary with data for evaluation
    """
    data_type = config.get("data", "type", default="synthetic")
    
    if data_type == "synthetic":
        return generate_synthetic_data(config, logger)
    elif data_type == "real":
        return load_real_binary_data(config, logger)
    elif data_type == "cross_arch":
        return load_cross_architecture_data(config, logger)
    else:
        logger.error(f"Unknown data type: {data_type}")
        raise ValueError(f"Unknown data type: {data_type}")

def generate_synthetic_data(config, logger):
    """
    Generate synthetic data for evaluation.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        dict: Dictionary with synthetic data
    """
    num_samples = config.get("data", "synthetic", "num_samples", default=500)
    sequence_length = config.get("data", "synthetic", "sequence_length", default=20)
    seed = config.get("experiment", "seed", default=42)
    
    logger.info(f"Generating synthetic data with {num_samples} samples per task")
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=seed)
    
    # Generate instruction sequences for training
    num_sequences = 1000
    logger.info(f"Generating {num_sequences} synthetic instruction sequences")
    instruction_sequences = [
        generator.generate_instruction_sequence(sequence_length)
        for _ in range(num_sequences)
    ]
    
    # Generate task data
    logger.info(f"Generating task data with {num_samples} samples per task")
    task_data = generator.generate_all_task_data(num_samples)
    
    # Generate additional task data for new tasks
    logger.info("Generating data for additional tasks")
    
    # Function boundary detection data
    function_boundaries = []
    boundary_labels = []
    
    for i in range(min(100, num_sequences)):
        # Insert random function boundaries
        sequence = instruction_sequences[i]
        boundaries = sorted(random.sample(range(1, len(sequence)), 3))
        
        function_boundaries.append(sequence)
        boundary_labels.append(boundaries)
    
    task_data['function_boundary'] = {
        'instruction_sequences': function_boundaries,
        'boundaries': boundary_labels
    }
    
    # Vulnerability detection data
    vulnerability_task = get_task('vulnerability')
    task_data['vulnerability'] = vulnerability_task.generate_synthetic_data(
        num_samples=num_samples,
        num_patterns=5,
        seed=seed
    )
    
    return {
        'instruction_sequences': instruction_sequences,
        'task_data': task_data
    }

def load_real_binary_data(config, logger):
    """
    Load real binary data for evaluation.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        dict: Dictionary with real binary data
    """
    repo_urls = config.get("data", "real", "repo_urls", default=[])
    use_cache = config.get("data", "real", "use_cache", default=True)
    results_dir = config.get("output", "results_dir", default="results")
    
    logger.info(f"Loading real binary data")
    
    # Initialize loader
    output_dir = os.path.join(results_dir, "real_data")
    loader = RealBinaryLoader(output_dir=output_dir, use_cache=use_cache)
    
    # Process binaries and generate ground truth
    data = loader.process_and_generate_ground_truth(repo_urls=repo_urls)
    
    # Prepare data for evaluation
    instruction_sequences = []
    task_data = {}
    
    # Load functions from processed files
    functions_dir = os.path.join(output_dir, "functions")
    if os.path.exists(functions_dir):
        for file_name in os.listdir(functions_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(functions_dir, file_name), 'r') as f:
                    func_data = json.load(f)
                
                # Extract instruction sequences
                for func_name, instructions in func_data.get("functions", {}).items():
                    instruction_sequences.append(instructions)
    
    # Get task-specific data
    for task_name, task_data_item in data.items():
        if task_name == 'instruction_synonyms':
            task_data['synonym'] = task_data_item
        elif task_name == 'block_equivalence':
            task_data['block'] = task_data_item
        elif task_name == 'dead_code':
            task_data['dead_code'] = task_data_item
    
    # Use enhanced versions for enhanced tasks
    task_data['enhanced_synonym'] = task_data.get('synonym')
    task_data['enhanced_block'] = task_data.get('block')
    task_data['enhanced_dead_code'] = task_data.get('dead_code')
    
    return {
        'instruction_sequences': instruction_sequences,
        'task_data': task_data
    }

def load_cross_architecture_data(config, logger):
    """
    Load cross-architecture data for evaluation.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        dict: Dictionary with cross-architecture data
    """
    architectures = config.get("data", "cross_arch", "architectures", default=["x86_64", "arm64"])
    compiler = config.get("data", "cross_arch", "compiler", default="gcc")
    opt_level = config.get("data", "cross_arch", "opt_level", default="-O0")
    results_dir = config.get("output", "results_dir", default="results")
    
    logger.info(f"Loading cross-architecture data for {', '.join(architectures)}")
    
    # Initialize processor
    output_dir = os.path.join(results_dir, "cross_arch_data")
    processor = CrossArchitectureProcessor(output_dir=output_dir)
    
    # Check available architectures
    available_archs = processor.get_available_architectures()
    archs_to_use = [arch for arch in architectures if arch in available_archs]
    
    if not archs_to_use:
        logger.warning(f"No requested architectures available. Using {available_archs[0]}")
        archs_to_use = [available_archs[0]]
    
    logger.info(f"Using architectures: {', '.join(archs_to_use)}")
    
    # Process data
    # For simplicity, we'll use synthetic source files
    temp_dir = os.path.join(output_dir, "temp_sources")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a simple source file
    source_path = os.path.join(temp_dir, "test.c")
    with open(source_path, 'w') as f:
        f.write("""
        #include <stdio.h>
        
        int add(int a, int b) {
            return a + b;
        }
        
        int subtract(int a, int b) {
            return a - b;
        }
        
        int main() {
            int x = 10;
            int y = 5;
            printf("Sum: %d\\n", add(x, y));
            printf("Difference: %d\\n", subtract(x, y));
            return 0;
        }
        """)
    
    # Process source directory
    mappings = processor.process_source_directory(
        source_dir=temp_dir,
        architectures=archs_to_use,
        compiler=compiler,
        opt_level=opt_level
    )
    
    # Prepare data for evaluation
    instruction_sequences = []
    task_data = {}
    
    # Load functions from processed files
    functions_dir = os.path.join(output_dir, "functions")
    if os.path.exists(functions_dir):
        for file_name in os.listdir(functions_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(functions_dir, file_name), 'r') as f:
                    func_data = json.load(f)
                
                # Extract instruction sequences
                for func_name, instructions in func_data.get("functions", {}).items():
                    instruction_sequences.append(instructions)
    
    # Get task-specific data
    if 'instruction_mappings' in mappings:
        task_data['synonym'] = mappings['instruction_mappings']
        task_data['enhanced_synonym'] = mappings['instruction_mappings']
    
    if 'block_mappings' in mappings:
        task_data['block'] = mappings['block_mappings']
        task_data['enhanced_block'] = mappings['block_mappings']
    
    return {
        'instruction_sequences': instruction_sequences,
        'task_data': task_data
    }

def train_embedding_models(config, data, logger):
    """
    Train embedding models on instruction sequences.
    
    Args:
        config: Experiment configuration
        data: Dictionary with instruction_sequences
        logger: Logger instance
        
    Returns:
        dict: Dictionary of trained models
    """
    embedding_models = {}
    model_names = config.get_embedding_models()
    
    for name in model_names:
        logger.info(f"Training {name} embedding model...")
        
        # Get model-specific parameters
        params = config.get_embedding_params(name)
        
        # Initialize and train model
        model = get_embedding_model(name, **params)
        model.fit(data['instruction_sequences'])
        
        embedding_models[name] = model
        logger.info(f"Finished training {name}")
    
    return embedding_models

def run_tasks(config, tasks, embedding_models, data, logger):
    """
    Run evaluation tasks on embedding models.
    
    Args:
        config: Experiment configuration
        tasks: Dictionary of task instances
        embedding_models: Dictionary of trained models
        data: Dictionary with task data
        logger: Logger instance
        
    Returns:
        dict: Dictionary mapping models to task results
    """
    all_results = {}
    
    for model_name, model in embedding_models.items():
        logger.info(f"Evaluating {model_name}...")
        model_results = {}
        
        for task_name, task in tasks.items():
            logger.info(f"Running {task.name} task...")
            
            # Get task-specific data
            if task_name in data['task_data']:
                task_data = data['task_data'][task_name]
                
                # Run evaluation
                result = task.run(model, task_data)
                
                # Store result
                model_results[task.name] = result
                
                logger.info(f"Completed {task.name} task")
            else:
                logger.warning(f"No data available for {task_name} task. Skipping.")
        
        all_results[model_name] = model_results
        logger.info(f"Finished evaluating {model_name}")
    
    return all_results

def analyze_results(config, all_results, embedding_models, data, logger):
    """
    Analyze and visualize results.
    
    Args:
        config: Experiment configuration
        all_results: Dictionary mapping models to task results
        embedding_models: Dictionary of trained models
        data: Dictionary with evaluation data
        logger: Logger instance
        
    Returns:
        tuple: (model_errors, embedding_visualizations)
    """
    results_dir, vis_dir = config.get_output_dirs()
    vis_enabled = config.get("visualization", "enabled", default=True)
    vis_formats = config.get("visualization", "formats", default=["png"])
    tasks_to_visualize = config.get("visualization", "tasks_to_visualize", default="all")
    projection_method = config.get("visualization", "embedding_projection", default="tsne")
    
    # Skip visualization if disabled
    if not vis_enabled:
        return {}, {}
    
    # Create output directories
    os.makedirs(vis_dir, exist_ok=True)
    
    # Determine which tasks to visualize
    if tasks_to_visualize == "all":
        task_names = set()
        for model_results in all_results.values():
            task_names.update(model_results.keys())
    else:
        task_names = set()
        for category in tasks_to_visualize.split(","):
            task_names.update(config.get_tasks(category))
    
    # Analyze each task
    model_errors = {}
    embedding_visualizations = {}
    
    for task_name in task_names:
        # Collect scores for each model
        model_scores = {}
        for model_name, model_results in all_results.items():
            if task_name in model_results:
                model_scores[model_name] = model_results[task_name]['scores']
        
        # Skip if no results
        if not model_scores:
            continue
        
        # Visualize task performance
        task_vis_path = os.path.join(vis_dir, f"task_{task_name.lower().replace(' ', '_')}.png")
        visualize_task_performance(
            model_scores=model_scores,
            task_name=task_name,
            output_path=task_vis_path
        )
        
        logger.info(f"Generated visualization for {task_name}")
        
        # Generate error reports
        task_error_dir = os.path.join(vis_dir, "errors", task_name.lower().replace(' ', '_'))
        os.makedirs(task_error_dir, exist_ok=True)
        
        task_data = None
        for td_name, td in data['task_data'].items():
            if td_name.lower().replace('_', ' ') == task_name.lower().replace('_', ' '):
                task_data = td
                break
        
        if task_data:
            for model_name, model_results in all_results.items():
                if task_name in model_results:
                    task_result = model_results[task_name]
                    
                    error_report = generate_error_report(
                        model_name=model_name,
                        task_name=task_name,
                        task_results=task_result['results'],
                        task_data=task_data,
                        output_dir=task_error_dir
                    )
                    
                    if model_name not in model_errors:
                        model_errors[model_name] = {}
                    
                    model_errors[model_name][task_name] = error_report
    
    # Visualize embedding spaces
    for model_name, model in embedding_models.items():
        # Sample a subset of instructions for visualization
        if len(data['instruction_sequences']) > 1000:
            sample_indices = random.sample(range(len(data['instruction_sequences'])), 1000)
            sample_sequences = [data['instruction_sequences'][i] for i in sample_indices]
        else:
            sample_sequences = data['instruction_sequences']
        
        # Flatten sequences to get individual instructions
        flat_instructions = []
        for sequence in sample_sequences:
            flat_instructions.extend(sequence)
        
        # Generate embeddings
        embeddings = model.transform(flat_instructions)
        
        # Create visualization
        emb_vis_path = os.path.join(vis_dir, f"embedding_space_{model_name}.png")
        visualize_embedding_space(
            embeddings=embeddings,
            method=projection_method,
            output_path=emb_vis_path,
            title=f"{model_name} Embedding Space"
        )
        
        embedding_visualizations[model_name] = emb_vis_path
        logger.info(f"Generated embedding visualization for {model_name}")
    
    return model_errors, embedding_visualizations

def save_models_and_embeddings(config, embedding_models, data, logger):
    """
    Save trained models and embeddings.
    
    Args:
        config: Experiment configuration
        embedding_models: Dictionary of trained models
        data: Dictionary with evaluation data
        logger: Logger instance
    """
    results_dir, _ = config.get_output_dirs()
    save_models = config.get("output", "save_models", default=True)
    save_embeddings = config.get("output", "save_embeddings", default=True)
    
    if not (save_models or save_embeddings):
        return
    
    # Create output directories
    models_dir = os.path.join(results_dir, "models")
    embeddings_dir = os.path.join(results_dir, "embeddings")
    
    if save_models:
        os.makedirs(models_dir, exist_ok=True)
    
    if save_embeddings:
        os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save each model and its embeddings
    for model_name, model in embedding_models.items():
        if save_models:
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
            try:
                model.save(model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name} model: {e}")
        
        if save_embeddings:
            # Generate embeddings for a sample of instructions
            sample_instructions = []
            
            # Select a few instructions from each sequence
            for sequence in data['instruction_sequences'][:100]:
                sample_size = min(10, len(sequence))
                sample = random.sample(sequence, sample_size)
                sample_instructions.extend(sample)
            
            # Generate embeddings
            try:
                embeddings = model.transform(sample_instructions)
                
                # Save as numpy array
                emb_path = os.path.join(embeddings_dir, f"{model_name}_embeddings.npy")
                np.save(emb_path, embeddings)
                
                # Save instructions for reference
                instr_path = os.path.join(embeddings_dir, f"{model_name}_instructions.txt")
                with open(instr_path, 'w') as f:
                    for instr in sample_instructions:
                        f.write(f"{instr}\n")
                
                logger.info(f"Saved {model_name} embeddings to {emb_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name} embeddings: {e}")

def main():
    """Main function for running experiments."""
    # Parse arguments
    args = parse_args()
    
    # Initialize configuration
    config = ExperimentConfig(args.config)
    update_config_from_args(config, args)
    
    # Get output directories
    results_dir, _ = config.get_output_dirs()
    
    # Setup logging
    logger, timestamp = setup_logging(os.path.join(results_dir, "logs"))
    
    logger.info("Starting experiment")
    logger.info(f"Configuration: {json.dumps(config.config, indent=2)}")
    
    # Save configuration
    config_path = os.path.join(results_dir, f"config_{timestamp}.json")
    config.save(config_path)
    
    # Set random seed for reproducibility
    seed = config.get("experiment", "seed", default=42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    try:
        # Load or generate data
        logger.info("Loading/generating data...")
        data = load_or_generate_data(config, logger)
        
        # Initialize tasks
        logger.info("Initializing evaluation tasks...")
        tasks = {}
        
        for task_category in ["basic", "enhanced", "new"]:
            for task_name in config.get_tasks(task_category):
                tasks[task_name] = get_task(task_name)
                logger.info(f"Initialized {tasks[task_name].name} task")
        
        # Train embedding models
        logger.info("Training embedding models...")
        embedding_models = train_embedding_models(config, data, logger)
        
        # Run evaluation tasks
        logger.info("Running evaluation tasks...")
        task_results = run_tasks(config, tasks, embedding_models, data, logger)
        
        # Initialize scorer
        scorer = EvaluationScorer()
        
        # Add results to scorer
        for model_name, model_results in task_results.items():
            for task_name, result in model_results.items():
                scorer.add_task_result(model_name, task_name, result)
        
        # Save results
        logger.info("Saving results...")
        scorer.save_results(results_dir)
        
        # Save models and embeddings
        save_models_and_embeddings(config, embedding_models, data, logger)
        
        # Analyze and visualize results
        logger.info("Analyzing results...")
        model_errors, embedding_visualizations = analyze_results(
            config, task_results, embedding_models, data, logger
        )
        
        # Print summary
        best_model, best_score = scorer.get_best_model()
        logger.info(f"Best performing model: {best_model} with score {best_score:.4f}")
        
        print("\n" + scorer.format_results())
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        raise

if __name__ == "__main__":
    main()