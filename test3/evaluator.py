#!/usr/bin/env python3
"""
Evaluate the robustness of binary similarity detection models against obfuscations.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from typing import Dict, List, Any, Tuple
import pickle
from model_manager import ModelManager, GraphBasedSimilarity, Asm2VecModel, GeminiModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustnessEvaluator:
    """Evaluates binary similarity models against various obfuscation techniques."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.model_manager = ModelManager()
        self.setup_models()
    
    def setup_models(self):
        """Set up and load all models."""
        # Add all models
        self.model_manager.add_model("graph", GraphBasedSimilarity())
        self.model_manager.add_model("asm2vec", Asm2VecModel())
        self.model_manager.add_model("gemini", GeminiModel())
        
        # Load trained models
        for model_name in self.model_manager.models:
            model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                logger.info(f"Loading model: {model_name}")
                self.model_manager.load_model(model_name, model_path)
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    def evaluate(self, features_dir: str, output_dir: str, baseline_prefix: str = "original"):
        """
        Evaluate all models against different obfuscation techniques.
        
        Args:
            features_dir: Directory containing feature files
            output_dir: Directory to save evaluation results
            baseline_prefix: Prefix of baseline (non-obfuscated) binaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Group feature files by program and obfuscation technique
        programs = self._group_feature_files(features_dir, baseline_prefix)
        
        # Initialize results storage
        results = {
            "program": [],
            "obfuscation": [],
            "model": [],
            "similarity": []
        }
        
        # Evaluate each model on each program and obfuscation technique
        for program, techniques in programs.items():
            logger.info(f"Evaluating program: {program}")
            
            # Get baseline features
            baseline_features = self._load_features(techniques[baseline_prefix])
            
            for technique, feature_file in techniques.items():
                if technique == baseline_prefix:
                    continue  # Skip baseline
                    
                logger.info(f"  Evaluating obfuscation: {technique}")
                
                # Load obfuscated features
                obfuscated_features = self._load_features(feature_file)
                
                # Evaluate each model
                for model_name in self.model_manager.models:
                    logger.info(f"    Using model: {model_name}")
                    
                    # Calculate similarity
                    similarity = self.model_manager.predict_similarity(
                        model_name, baseline_features, obfuscated_features
                    )
                    
                    # Store result
                    results["program"].append(program)
                    results["obfuscation"].append(technique)
                    results["model"].append(model_name)
                    results["similarity"].append(similarity)
                    
                    logger.info(f"    Similarity: {similarity:.4f}")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        output_csv = os.path.join(output_dir, "robustness_results.csv")
        df_results.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
        
        # Generate visualizations
        self._generate_visualizations(df_results, output_dir)
    
    def _group_feature_files(self, features_dir: str, baseline_prefix: str) -> Dict[str, Dict[str, str]]:
        """
        Group feature files by program and obfuscation technique.
        
        Args:
            features_dir: Directory containing feature files
            baseline_prefix: Prefix of baseline (non-obfuscated) binaries
            
        Returns:
            Dictionary mapping program names to dictionaries of obfuscation techniques and feature files
        """
        programs = {}
        
        # Get all feature files
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.json')]
        
        for file in feature_files:
            # Expected format: program_technique.json
            parts = os.path.splitext(file)[0].split('_')
            
            if len(parts) < 2:
                logger.warning(f"Skipping file with unexpected format: {file}")
                continue
                
            program = parts[0]
            technique = '_'.join(parts[1:])
            
            if program not in programs:
                programs[program] = {}
                
            programs[program][technique] = os.path.join(features_dir, file)
            
            # Ensure each program has a baseline
            if technique.startswith(baseline_prefix) and baseline_prefix not in programs[program]:
                programs[program][baseline_prefix] = os.path.join(features_dir, file)
        
        # Filter out programs that don't have a baseline
        programs = {p: t for p, t in programs.items() if baseline_prefix in t}
        
        if not programs:
            logger.warning(f"No programs with baseline ({baseline_prefix}) found")
            
        return programs
    
    def _load_features(self, feature_file: str) -> Dict[str, Any]:
        """
        Load features from a JSON file.
        
        Args:
            feature_file: Path to the feature file
            
        Returns:
            Dictionary of features
        """
        with open(feature_file, 'r') as f:
            return json.load(f)
    
    def _generate_visualizations(self, df: pd.DataFrame, output_dir: str):
        """
        Generate visualizations from evaluation results.
        
        Args:
            df: DataFrame containing evaluation results
            output_dir: Directory to save visualizations
        """
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Heatmap of model performance across obfuscation techniques
        plt.figure(figsize=(12, 8))
        heatmap_data = df.pivot_table(
            values="similarity", 
            index="obfuscation", 
            columns="model", 
            aggfunc="mean"
        )
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Average Similarity Scores by Model and Obfuscation Technique")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_obfuscation_model.png"), dpi=300)
        plt.close()
        
        # 2. Bar plot comparing models across techniques
        plt.figure(figsize=(14, 8))
        sns.barplot(x="obfuscation", y="similarity", hue="model", data=df)
        plt.title("Model Performance Across Obfuscation Techniques")
        plt.xlabel("Obfuscation Technique")
        plt.ylabel("Similarity Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "barplot_model_comparison.png"), dpi=300)
        plt.close()
        
        # 3. Box plot showing distribution of similarity scores by model
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="model", y="similarity", data=df)
        plt.title("Distribution of Similarity Scores by Model")
        plt.xlabel("Model")
        plt.ylabel("Similarity Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "boxplot_model_scores.png"), dpi=300)
        plt.close()
        
        # 4. Line chart showing model robustness decay across techniques
        # First, order obfuscation techniques by difficulty
        technique_difficulty = {
            "flatten": 1,
            "encodeliterals": 2,
            "virtualize": 3,
            "ollvm_sub": 4,
            "ollvm_bcf": 5,
            "ollvm_fla": 6
        }
        
        # Add difficulty ranking to DataFrame
        df["difficulty"] = df["obfuscation"].map(lambda x: technique_difficulty.get(x, 99))
        
        # Sort by difficulty
        df_sorted = df.sort_values("difficulty")
        
        plt.figure(figsize=(12, 6))
        for model in df["model"].unique():
            model_data = df_sorted[df_sorted["model"] == model]
            plt.plot(model_data["obfuscation"], model_data["similarity"], marker='o', label=model)
            
        plt.title("Model Robustness Across Increasing Obfuscation Complexity")
        plt.xlabel("Obfuscation Technique (Increasing Complexity)")
        plt.ylabel("Average Similarity Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "line_robustness_decay.png"), dpi=300)
        plt.close()
        
        # 5. Generate summary statistics table
        summary = df.groupby(["model", "obfuscation"])["similarity"].agg(["mean", "std", "min", "max"])
        summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
        
        # 6. Create a radar chart for model comparison
        model_avg = df.groupby("model")["similarity"].mean().reset_index()
        model_min = df.groupby("model")["similarity"].min().reset_index()
        
        # Radar chart requires a bit more setup
        labels = model_avg["model"].tolist()
        avg_scores = model_avg["similarity"].tolist()
        min_scores = model_min["similarity"].tolist()
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        avg_scores += avg_scores[:1]
        min_scores += min_scores[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, avg_scores, 'o-', linewidth=2, label='Average Similarity')
        ax.plot(angles, min_scores, 'o-', linewidth=2, label='Minimum Similarity')
        ax.fill(angles, avg_scores, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        plt.title("Model Comparison: Average and Minimum Similarity")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radar_model_comparison.png"), dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate robustness of binary similarity models")
    parser.add_argument("--features-dir", required=True, help="Directory containing feature files")
    parser.add_argument("--models-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--baseline", default="original", help="Prefix of baseline (non-obfuscated) binaries")
    
    args = parser.parse_args()
    
    evaluator = RobustnessEvaluator(args.models_dir)
    evaluator.evaluate(args.features_dir, args.output_dir, args.baseline)

if __name__ == "__main__":
    main()