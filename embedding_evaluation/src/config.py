"""
Configuration system for reproducible experiments.
"""

import os
import json
import copy
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "experiment": {
            "name": "instruction_embedding_evaluation",
            "description": "Evaluation of instruction embedding techniques for binary analysis",
            "seed": 42,
            "reproducible": True
        },
        "data": {
            "type": "synthetic",  # Options: synthetic, real, cross_arch
            "synthetic": {
                "num_samples": 500,
                "sequence_length": 20
            },
            "real": {
                "repo_urls": [
                    "https://github.com/antirez/redis",
                    "https://github.com/tmux/tmux"
                ],
                "use_cache": True
            },
            "cross_arch": {
                "architectures": ["x86_64", "arm64"],
                "compiler": "gcc",
                "opt_level": "-O0"
            }
        },
        "embeddings": {
            "models": ["word2vec", "palmtree", "bert", "graph", "tfidf"],
            "embedding_dim": 100,
            "params": {
                "word2vec": {
                    "window": 5,
                    "min_count": 1,
                    "workers": 4
                },
                "palmtree": {
                    "context_size": 5,
                    "epochs": 50
                },
                "bert": {
                    "max_seq_length": 128,
                    "batch_size": 32,
                    "epochs": 3
                },
                "graph": {
                    "hidden_dim": 128,
                    "initial_dim": 64,
                    "batch_size": 32,
                    "epochs": 50
                },
                "tfidf": {
                    "max_features": 100
                }
            }
        },
        "tasks": {
            "basic": ["synonym", "block", "dead_code"],
            "enhanced": ["enhanced_synonym", "enhanced_block", "enhanced_dead_code"],
            "new": ["function_boundary", "vulnerability"]
        },
        "visualization": {
            "enabled": True,
            "formats": ["png"],
            "embedding_projection": "tsne",  # Options: tsne, pca, umap
            "tasks_to_visualize": "all"      # Options: all, basic, enhanced, new
        },
        "output": {
            "results_dir": "results",
            "visualizations_dir": "visualizations",
            "save_models": True,
            "save_embeddings": True
        }
    }
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to JSON configuration file (or None for default)
        """
        # Start with default configuration
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration recursively
            self._update_dict_recursive(self.config, loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    def _update_dict_recursive(self, target, source):
        """
        Update dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict_recursive(target[key], value)
            else:
                target[key] = value
    
    def save(self, output_path=None):
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save configuration (or None for default)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config["output"]["results_dir"]
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"config_{timestamp}.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
    
    def get(self, *keys, default=None):
        """
        Get configuration value using nested keys.
        
        Args:
            *keys: Sequence of keys to traverse
            default: Default value if key not found
            
        Returns:
            Value at the specified key path
        """
        result = self.config
        try:
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError):
            return default
    
    def set(self, value, *keys):
        """
        Set configuration value using nested keys.
        
        Args:
            value: Value to set
            *keys: Sequence of keys to traverse
        """
        if not keys:
            return
        
        target = self.config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def get_embedding_models(self):
        """
        Get list of embedding models to evaluate.
        
        Returns:
            list: List of model names
        """
        return self.get("embeddings", "models", default=[])
    
    def get_embedding_params(self, model_name):
        """
        Get parameters for a specific embedding model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            dict: Parameters for the model
        """
        params = self.get("embeddings", "params", model_name, default={}).copy()
        params["embedding_dim"] = self.get("embeddings", "embedding_dim", default=100)
        return params
    
    def get_tasks(self, category=None):
        """
        Get list of tasks to evaluate.
        
        Args:
            category: Task category (basic, enhanced, new, or None for all)
            
        Returns:
            list: List of task names
        """
        if category:
            return self.get("tasks", category, default=[])
        else:
            # Combine all task categories
            tasks = []
            for cat in ["basic", "enhanced", "new"]:
                tasks.extend(self.get("tasks", cat, default=[]))
            return tasks
    
    def get_data_config(self):
        """
        Get data configuration.
        
        Returns:
            dict: Data configuration
        """
        return self.get("data", default={})
    
    def get_output_dirs(self):
        """
        Get output directories.
        
        Returns:
            tuple: (results_dir, visualizations_dir)
        """
        results_dir = self.get("output", "results_dir", default="results")
        vis_dir = self.get("output", "visualizations_dir", default="visualizations")
        return results_dir, vis_dir
    
    def generate_experiment_id(self):
        """
        Generate a unique experiment ID.
        
        Returns:
            str: Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.get("experiment", "name", default="experiment")
        return f"{name}_{timestamp}"