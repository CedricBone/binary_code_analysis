#!/usr/bin/env python3
"""
Test trained model on different architectures and optimization levels.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from multiprocessing import Pool
from tqdm import tqdm
import config
from utils import (
    logger, create_directories, set_random_seed, get_available_architectures,
    generate_synthetic_functions
)
from train_model import (
    load_functions, create_function_pairs, FunctionPairGenerator, 
    InstructionTokenizer
)

def load_model_and_tokenizer(model_dir):
    """Load a trained model and tokenizer"""
    # Load model
    model_path = os.path.join(model_dir, "model.h5")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found: {tokenizer_path}")
        return model, None
    
    try:
        tokenizer = InstructionTokenizer.load(tokenizer_path)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return model, None
    
    return model, tokenizer

def evaluate_model(model, tokenizer, function_pairs, batch_size=32, max_length=150):
    """Evaluate a model on a set of function pairs"""
    # Check if we have enough pairs to evaluate
    if len(function_pairs) < batch_size:
        logger.warning(f"Not enough function pairs ({len(function_pairs)}) for evaluation batch size {batch_size}")
        
        # Duplicate pairs if needed to reach batch size
        while len(function_pairs) < batch_size:
            function_pairs.extend(function_pairs[:batch_size - len(function_pairs)])
    
    # Create data generator
    test_generator = FunctionPairGenerator(
        function_pairs,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        training=False
    )
    
    # Make predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = np.array([pair["label"] for pair in function_pairs])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate AUC (handle case of single class)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        auc = 0.5  # Default for random performance
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Return metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": conf_matrix.tolist()
    }
    
    return metrics

def test_configuration(args):
    """Test a model on a specific configuration"""
    model, tokenizer, source_arch, source_compiler, source_opt, target_arch, target_compiler, target_opt, out_dir, use_synthetic_data = args
    
    # Create a unique ID for this test
    test_id = f"{source_arch}_{source_compiler}_{source_opt}_to_{target_arch}_{target_compiler}_{target_opt}"
    logger.info(f"Testing configuration: {test_id}")
    
    # Load functions for target architecture, compiler, and optimization level
    target_functions = load_functions(target_arch, target_compiler, target_opt)
    
    # Generate synthetic data if needed
    if len(target_functions) == 0 or use_synthetic_data:
        logger.warning(f"No functions found for {target_arch} with {target_compiler} -{target_opt} or synthetic data requested")
        logger.info("Generating synthetic function data for testing...")
        target_functions = generate_synthetic_functions(num_functions=1000, architecture=target_arch)
        
        # Save synthetic functions to file for future reference
        synthetic_dir = os.path.join(config.FUNCTION_DIR, "synthetic", target_arch, target_compiler, target_opt)
        os.makedirs(synthetic_dir, exist_ok=True)
        synthetic_path = os.path.join(synthetic_dir, "test_functions.json")
        with open(synthetic_path, 'w') as f:
            json.dump(target_functions, f, indent=2)
        logger.info(f"Saved {len(target_functions)} synthetic functions to {synthetic_path}")
    
    # Create function pairs for testing
    function_pairs = create_function_pairs(
        target_functions,
        similar_ratio=0.5,
        max_pairs=config.NUM_TEST_PAIRS
    )
    
    # Evaluate model on the function pairs
    try:
        metrics = evaluate_model(
            model,
            tokenizer,
            function_pairs,
            batch_size=config.BATCH_SIZE,
            max_length=config.SEQUENCE_LENGTH
        )
        
        # Save results
        result = {
            "source_arch": source_arch,
            "source_compiler": source_compiler,
            "source_opt": source_opt,
            "target_arch": target_arch,
            "target_compiler": target_compiler,
            "target_opt": target_opt,
            "metrics": metrics,
            "error": None,
            "using_synthetic_data": len(target_functions) == 0 or use_synthetic_data,
            "num_functions": len(target_functions),
            "num_pairs": len(function_pairs)
        }
        
        # Save to file
        result_path = os.path.join(out_dir, f"{test_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Test results saved to {result_path}")
        
        # Plot confusion matrix
        if metrics["confusion_matrix"]:
            cm = np.array(metrics["confusion_matrix"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Dissimilar", "Similar"],
                yticklabels=["Dissimilar", "Similar"]
            )
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(f'Confusion Matrix for {test_id}')
            
            # Save plot
            cm_path = os.path.join(out_dir, f"{test_id}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating model on {test_id}: {e}")
        return {
            "source_arch": source_arch,
            "source_compiler": source_compiler,
            "source_opt": source_opt,
            "target_arch": target_arch,
            "target_compiler": target_compiler,
            "target_opt": target_opt,
            "metrics": None,
            "error": str(e),
            "using_synthetic_data": len(target_functions) == 0 or use_synthetic_data,
            "num_functions": len(target_functions),
            "num_pairs": len(function_pairs) if 'function_pairs' in locals() else 0
        }

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot a confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_metrics(results, output_dir):
    """Plot metrics across architectures, compilers, and optimization levels"""
    # Extract metrics
    data = []
    for result in results:
        if result["metrics"] is not None:
            data.append({
                "source_arch": result["source_arch"],
                "source_compiler": result["source_compiler"],
                "source_opt": result["source_opt"],
                "target_arch": result["target_arch"],
                "target_compiler": result["target_compiler"],
                "target_opt": result["target_opt"],
                "accuracy": result["metrics"]["accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "auc": result["metrics"]["auc"],
                "using_synthetic_data": result.get("using_synthetic_data", False)
            })
    
    if not data:
        logger.error("No valid results to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    # Create architecture transfer plot
    arch_data = df.groupby("target_arch").mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    
    for metric in metrics:
        plt.bar(
            np.arange(len(arch_data)) + (metrics.index(metric) - len(metrics)/2) * 0.15,
            arch_data[metric],
            width=0.15,
            label=metric.capitalize()
        )
    
    plt.xlabel("Target Architecture")
    plt.ylabel("Score")
    plt.title("Performance Across Target Architectures")
    plt.xticks(np.arange(len(arch_data)), arch_data["target_arch"])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "architecture_transfer.png"), dpi=300)
    plt.close()
    
    # Create compiler impact plot for each architecture
    compiler_data = df.groupby(["target_arch", "target_compiler"]).mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    
    for i, arch in enumerate(compiler_data["target_arch"].unique()):
        arch_df = compiler_data[compiler_data["target_arch"] == arch]
        
        for metric in metrics:
            plt.bar(
                np.arange(len(arch_df)) + i * len(arch_df["target_compiler"].unique()) + (metrics.index(metric) - len(metrics)/2) * 0.15,
                arch_df[metric],
                width=0.15,
                label=f"{arch}_{metric}" if i == 0 else ""
            )
    
    plt.xlabel("Target Compiler")
    plt.ylabel("Score")
    plt.title("Compiler Impact Across Architectures")
    plt.xticks(np.arange(len(compiler_data["target_compiler"].unique())), compiler_data["target_compiler"].unique())
    plt.legend(metrics)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compiler_impact.png"), dpi=300)
    plt.close()
    
    # Create optimization level impact plot
    opt_data = df.groupby(["target_arch", "target_opt"]).mean().reset_index()
    
    # Create a table for optimization level impact
    opt_table = pd.pivot_table(
        opt_data,
        values="accuracy",
        index=["target_arch"],
        columns=["target_opt"]
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        opt_table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy"}
    )
    
    plt.title("Optimization Level Impact on Accuracy")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_impact.png"), dpi=300)
    plt.close()
    
    # Create detailed heatmap for all configurations
    accuracy_table = pd.pivot_table(
        df,
        values="accuracy",
        index=["target_arch", "target_compiler"],
        columns=["target_opt"]
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        accuracy_table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy"}
    )
    
    plt.title("Detailed Accuracy Heatmap for All Configurations")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_accuracy_heatmap.png"), dpi=300)
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test a binary similarity model')
    parser.add_argument('--source-arch', type=str, default='x86_64',
                        help='Source architecture used for training')
    parser.add_argument('--source-compiler', type=str, default='gcc',
                        help='Source compiler used for training')
    parser.add_argument('--source-opt', type=str, default='O2',
                        help='Source optimization level used for training')
    parser.add_argument('--target-arch', type=str, default=None,
                        help='Specific target architecture to test on')
    parser.add_argument('--target-compiler', type=str, default=None,
                        help='Specific target compiler to test on')
    parser.add_argument('--target-opt', type=str, default=None,
                        help='Specific target optimization level to test on')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel tests')
    parser.add_argument('--use-synthetic-data', action='store_true',
                        help='Use synthetic data for testing')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed()
    
    # Create directories
    create_directories()
    
    # Define model and results directories
    model_id = f"{args.source_arch}_{args.source_compiler}_{args.source_opt}"
    model_dir = os.path.join(config.MODEL_DIR, model_id)
    results_dir = os.path.join(config.RESULTS_DIR, model_id)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    if model is None:
        logger.error(f"Failed to load model from {model_dir}")
        return
    
    if tokenizer is None:
        logger.error(f"Failed to load tokenizer from {model_dir}")
        return
    
    # Prepare test configurations
    test_configs = []
    
    # Determine available architectures
    available_archs = get_available_architectures()
    if args.target_arch:
        if args.target_arch not in available_archs and not args.use_synthetic_data:
            logger.warning(f"Target architecture {args.target_arch} is not available. Using synthetic data.")
            args.use_synthetic_data = True
        available_archs = [args.target_arch]
    
    # Filter compilers
    compilers = [args.target_compiler] if args.target_compiler else config.COMPILERS.keys()
    
    # Filter optimization levels
    opt_levels = [args.target_opt] if args.target_opt else config.OPTIMIZATION_LEVELS
    
    # Generate all test configurations
    for target_arch in available_archs:
        for target_compiler in compilers:
            for target_opt in opt_levels:
                test_configs.append((
                    model,
                    tokenizer,
                    args.source_arch,
                    args.source_compiler,
                    args.source_opt,
                    target_arch,
                    target_compiler,
                    target_opt,
                    results_dir,
                    args.use_synthetic_data
                ))
    
    logger.info(f"Testing model on {len(test_configs)} configurations")
    
    # Run tests in parallel
    with Pool(processes=args.parallel) as pool:
        results = list(tqdm(pool.imap(test_configuration, test_configs), total=len(test_configs)))
    
    # Plot metrics
    plot_metrics(results, results_dir)
    
    # Save summary
    summary = {
        "source_arch": args.source_arch,
        "source_compiler": args.source_compiler,
        "source_opt": args.source_opt,
        "num_configs_tested": len(test_configs),
        "num_configs_succeeded": len([r for r in results if r["metrics"] is not None]),
        "using_synthetic_data": args.use_synthetic_data,
        "results": results
    }
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Testing complete. Results saved to {results_dir}")
    
    # Print summary
    success_rate = summary["num_configs_succeeded"] / summary["num_configs_tested"] * 100
    logger.info(f"Success rate: {success_rate:.2f}% ({summary['num_configs_succeeded']}/{summary['num_configs_tested']})")
    
    # Calculate average metrics
    if summary["num_configs_succeeded"] > 0:
        avg_metrics = {
            "accuracy": np.mean([r["metrics"]["accuracy"] for r in results if r["metrics"] is not None]),
            "precision": np.mean([r["metrics"]["precision"] for r in results if r["metrics"] is not None]),
            "recall": np.mean([r["metrics"]["recall"] for r in results if r["metrics"] is not None]),
            "f1": np.mean([r["metrics"]["f1"] for r in results if r["metrics"] is not None]),
            "auc": np.mean([r["metrics"]["auc"] for r in results if r["metrics"] is not None])
        }
        
        logger.info(f"Average metrics:")
        for metric, value in avg_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()