"""
Dataset classes for binary similarity and description generation
"""
import os
import json
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from preprocessing import BinaryPreprocessor

class BinarySimilarityDataset(Dataset):
    """Dataset for binary code similarity detection"""
    
    def __init__(self, 
                 data_path: str,
                 preprocessor: BinaryPreprocessor,
                 is_training: bool = True,
                 max_pairs: Optional[int] = None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to dataset directory or file
            preprocessor: Preprocessor instance for tokenization
            is_training: Whether this dataset is for training
            max_pairs: Maximum number of pairs to load (for debugging)
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.max_pairs = max_pairs
        
        self.function_pairs = []
        self.load_data()
        
    def load_data(self):
        """Load function pairs from data path"""
        
        if os.path.isfile(self.data_path) and self.data_path.endswith('.json'):
            # Load from single JSON file
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
                # Assuming JSON structure: [{"func1": "...", "func2": "...", "label": 1}, ...]
                for item in data:
                    if 'func1' in item and 'func2' in item and 'label' in item:
                        # Filter out empty lines
                        func1_lines = [line for line in item['func1'].split('\n') if line.strip()]
                        func2_lines = [line for line in item['func2'].split('\n') if line.strip()]
                        
                        if func1_lines and func2_lines:  # Only add if both have content
                            self.function_pairs.append({
                                'func1': func1_lines,
                                'func2': func2_lines,
                                'label': item['label']
                            })
                        
                        if self.max_pairs and len(self.function_pairs) >= self.max_pairs:
                            break
        
        elif os.path.isdir(self.data_path):
            # Load from directory structure
            # Assuming directory structure:
            # - similar_pairs/
            #   - pair1/
            #     - func1.asm
            #     - func2.asm
            #   - pair2/...
            # - dissimilar_pairs/...
            
            similar_dir = os.path.join(self.data_path, 'similar_pairs')
            dissimilar_dir = os.path.join(self.data_path, 'dissimilar_pairs')
            
            # Load similar pairs
            if os.path.exists(similar_dir):
                for pair_dir in os.listdir(similar_dir):
                    pair_path = os.path.join(similar_dir, pair_dir)
                    if os.path.isdir(pair_path):
                        func1_path = os.path.join(pair_path, 'func1.asm')
                        func2_path = os.path.join(pair_path, 'func2.asm')
                        
                        if os.path.exists(func1_path) and os.path.exists(func2_path):
                            with open(func1_path, 'r') as f1, open(func2_path, 'r') as f2:
                                # Filter out empty lines
                                func1 = [line for line in f1.read().strip().split('\n') if line.strip()]
                                func2 = [line for line in f2.read().strip().split('\n') if line.strip()]
                                
                                if func1 and func2:  # Only add if both have content
                                    self.function_pairs.append({
                                        'func1': func1,
                                        'func2': func2,
                                        'label': 1  # Similar
                                    })
                                
                                if self.max_pairs and len(self.function_pairs) >= self.max_pairs:
                                    break
            
            # Load dissimilar pairs
            if os.path.exists(dissimilar_dir) and (not self.max_pairs or len(self.function_pairs) < self.max_pairs):
                for pair_dir in os.listdir(dissimilar_dir):
                    pair_path = os.path.join(dissimilar_dir, pair_dir)
                    if os.path.isdir(pair_path):
                        func1_path = os.path.join(pair_path, 'func1.asm')
                        func2_path = os.path.join(pair_path, 'func2.asm')
                        
                        if os.path.exists(func1_path) and os.path.exists(func2_path):
                            with open(func1_path, 'r') as f1, open(func2_path, 'r') as f2:
                                # Filter out empty lines
                                func1 = [line for line in f1.read().strip().split('\n') if line.strip()]
                                func2 = [line for line in f2.read().strip().split('\n') if line.strip()]
                                
                                if func1 and func2:  # Only add if both have content
                                    self.function_pairs.append({
                                        'func1': func1,
                                        'func2': func2,
                                        'label': 0  # Dissimilar
                                    })
                                
                                if self.max_pairs and len(self.function_pairs) >= self.max_pairs:
                                    break
        
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
            
        print(f"Loaded {len(self.function_pairs)} function pairs")

    def __len__(self):
        return len(self.function_pairs)
    
    def __getitem__(self, idx):
        """Get a function pair and its label"""
        pair = self.function_pairs[idx]
        
        # Tokenize both functions
        func1_tokens = self.preprocessor.tokenize_function(pair['func1'])
        func2_tokens = self.preprocessor.tokenize_function(pair['func2'])
        
        # Convert to tensors
        func1_opcode_ids = torch.tensor(func1_tokens['opcode_ids'], dtype=torch.long)
        
        # Use a fixed max_operands size for all samples
        max_operands = 10  # Set a reasonable fixed size
        func1_operand_ids = torch.zeros((len(func1_tokens['operand_ids_list']), max_operands), dtype=torch.long)
        
        # Fix for operand IDs assignment
        for i, ops in enumerate(func1_tokens['operand_ids_list']):
            if ops:
                # Create a padded operands list
                padded_ops = ops[:max_operands] + [0] * (max_operands - len(ops[:max_operands]))
                # Assign values individually
                for j, op_id in enumerate(padded_ops):
                    func1_operand_ids[i, j] = op_id
        
        # Same for func2
        func2_opcode_ids = torch.tensor(func2_tokens['opcode_ids'], dtype=torch.long)
        
        func2_operand_ids = torch.zeros((len(func2_tokens['operand_ids_list']), max_operands), dtype=torch.long)
        
        # Fix for operand IDs assignment
        for i, ops in enumerate(func2_tokens['operand_ids_list']):
            if ops:
                # Create a padded operands list
                padded_ops = ops[:max_operands] + [0] * (max_operands - len(ops[:max_operands]))
                # Assign values individually
                for j, op_id in enumerate(padded_ops):
                    func2_operand_ids[i, j] = op_id
        
        # Label
        label = torch.tensor(pair['label'], dtype=torch.float)
        
        # If using CFG, process graph data
        if self.preprocessor.use_cfg:
            # This is simplified - in practice, you would use a proper graph representation
            # such as those from PyTorch Geometric
            
            # For func1
            func1_cfg = func1_tokens['cfg']
            func1_nodes = list(func1_cfg.nodes)
            func1_node_features = np.zeros((1, 5), dtype=np.float32) if not func1_nodes else np.zeros((len(func1_nodes), 5), dtype=np.float32)
            
            for i, node in enumerate(func1_nodes):
                if 'features' in func1_cfg.nodes[node]:
                    func1_node_features[i] = func1_cfg.nodes[node]['features']
            
            func1_adj_matrix = np.zeros((1, 1)) if not func1_nodes else nx.to_numpy_array(func1_cfg, nodelist=func1_nodes)
            
            # For func2
            func2_cfg = func2_tokens['cfg']
            func2_nodes = list(func2_cfg.nodes)
            func2_node_features = np.zeros((1, 5), dtype=np.float32) if not func2_nodes else np.zeros((len(func2_nodes), 5), dtype=np.float32)
            
            for i, node in enumerate(func2_nodes):
                if 'features' in func2_cfg.nodes[node]:
                    func2_node_features[i] = func2_cfg.nodes[node]['features']
            
            func2_adj_matrix = np.zeros((1, 1)) if not func2_nodes else nx.to_numpy_array(func2_cfg, nodelist=func2_nodes)
            
            # Convert to tensors
            func1_node_features = torch.tensor(func1_node_features, dtype=torch.float)
            func1_adj_matrix = torch.tensor(func1_adj_matrix, dtype=torch.float)
            
            func2_node_features = torch.tensor(func2_node_features, dtype=torch.float)
            func2_adj_matrix = torch.tensor(func2_adj_matrix, dtype=torch.float)
            
            return {
                'func1_opcode_ids': func1_opcode_ids,
                'func1_operand_ids': func1_operand_ids,
                'func1_node_features': func1_node_features,
                'func1_adj_matrix': func1_adj_matrix,
                'func2_opcode_ids': func2_opcode_ids,
                'func2_operand_ids': func2_operand_ids,
                'func2_node_features': func2_node_features,
                'func2_adj_matrix': func2_adj_matrix,
                'label': label
            }
        else:
            # Sequence-only mode
            return {
                'func1_opcode_ids': func1_opcode_ids,
                'func1_operand_ids': func1_operand_ids,
                'func2_opcode_ids': func2_opcode_ids,
                'func2_operand_ids': func2_operand_ids,
                'label': label
            }


class BinaryDescriptionDataset(Dataset):
    """Dataset for generating descriptions from binary code"""
    
    def __init__(self, 
                 data_path: str,
                 preprocessor: BinaryPreprocessor,
                 is_training: bool = True,
                 max_samples: Optional[int] = None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to dataset directory or file
            preprocessor: Preprocessor instance for tokenization
            is_training: Whether this dataset is for training
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.max_samples = max_samples
        
        self.samples = []
        self.load_data()
        
    def load_data(self):
        """Load function samples with descriptions"""
        
        if os.path.isfile(self.data_path) and self.data_path.endswith('.json'):
            # Load from single JSON file
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
                # Assuming JSON structure: [{"code": "...", "description": "..."}, ...]
                for item in data:
                    if 'code' in item and 'description' in item:
                        # Filter out empty lines
                        code_lines = [line for line in item['code'].split('\n') if line.strip()]
                        
                        if code_lines:  # Only add if code has content
                            self.samples.append({
                                'code': code_lines,
                                'description': item['description']
                            })
                        
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                            
        elif os.path.isdir(self.data_path):
            # Load from directory structure
            # Assuming directory structure:
            # - samples/
            #   - sample1/
            #     - func.asm
            #     - desc.txt
            #   - sample2/...
            
            samples_dir = os.path.join(self.data_path, 'samples')
            if os.path.exists(samples_dir):
                for sample_dir in os.listdir(samples_dir):
                    sample_path = os.path.join(samples_dir, sample_dir)
                    if os.path.isdir(sample_path):
                        code_path = os.path.join(sample_path, 'func.asm')
                        desc_path = os.path.join(sample_path, 'desc.txt')
                        
                        if os.path.exists(code_path) and os.path.exists(desc_path):
                            with open(code_path, 'r') as fc, open(desc_path, 'r') as fd:
                                # Filter out empty lines
                                code = [line for line in fc.read().strip().split('\n') if line.strip()]
                                description = fd.read().strip()
                                
                                if code and description:  # Only add if both have content
                                    self.samples.append({
                                        'code': code,
                                        'description': description
                                    })
                                
                                if self.max_samples and len(self.samples) >= self.max_samples:
                                    break
        
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
            
        print(f"Loaded {len(self.samples)} function samples with descriptions")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a function and its description"""
        sample = self.samples[idx]
        
        # Tokenize function
        func_tokens = self.preprocessor.tokenize_function(sample['code'])
        
        # Tokenize description
        desc_tokens = self.preprocessor.tokenize_description(sample['description'])
        
        # Convert to tensors
        func_opcode_ids = torch.tensor(func_tokens['opcode_ids'], dtype=torch.long)
        
        # Use a fixed max_operands size for all samples
        max_operands = 10  # Set a reasonable fixed size
        func_operand_ids = torch.zeros((len(func_tokens['operand_ids_list']), max_operands), dtype=torch.long)
        
        # Fix for operand IDs assignment
        for i, ops in enumerate(func_tokens['operand_ids_list']):
            if ops:
                # Create a padded operands list
                padded_ops = ops[:max_operands] + [0] * (max_operands - len(ops[:max_operands]))
                # Assign values individually
                for j, op_id in enumerate(padded_ops):
                    func_operand_ids[i, j] = op_id
        
        # Description tokens
        desc_token_ids = torch.tensor(desc_tokens, dtype=torch.long)
        
        # If using CFG, process graph data
        if self.preprocessor.use_cfg:
            # This is simplified - in practice, you would use a proper graph representation
            
            func_cfg = func_tokens['cfg']
            func_nodes = list(func_cfg.nodes)
            func_node_features = np.zeros((1, 5), dtype=np.float32) if not func_nodes else np.zeros((len(func_nodes), 5), dtype=np.float32)
            
            for i, node in enumerate(func_nodes):
                if 'features' in func_cfg.nodes[node]:
                    func_node_features[i] = func_cfg.nodes[node]['features']
            
            func_adj_matrix = np.zeros((1, 1)) if not func_nodes else nx.to_numpy_array(func_cfg, nodelist=func_nodes)
            
            # Convert to tensors
            func_node_features = torch.tensor(func_node_features, dtype=torch.float)
            func_adj_matrix = torch.tensor(func_adj_matrix, dtype=torch.float)
            
            return {
                'func_opcode_ids': func_opcode_ids,
                'func_operand_ids': func_operand_ids,
                'func_node_features': func_node_features,
                'func_adj_matrix': func_adj_matrix,
                'desc_token_ids': desc_token_ids,
                'raw_description': sample['description']  # Keep for evaluation
            }
        else:
            # Sequence-only mode
            return {
                'func_opcode_ids': func_opcode_ids,
                'func_operand_ids': func_operand_ids,
                'desc_token_ids': desc_token_ids,
                'raw_description': sample['description']  # Keep for evaluation
            }


def similarity_collate_fn(batch):
    """
    Custom collate function for similarity dataset
    Handles batching variable-sized graph data
    """
    # Get all keys from the first item in the batch
    keys = batch[0].keys()
    
    collated_batch = {}
    
    for key in keys:
        if key in ['func1_node_features', 'func1_adj_matrix', 'func2_node_features', 'func2_adj_matrix']:
            # For graph data, we can't batch directly, so we keep them as a list
            collated_batch[key] = [item[key] for item in batch]
        else:
            # For other tensors, use standard batching
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            except:
                # Fallback for any tensor that can't be stacked
                collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch


def description_collate_fn(batch):
    """
    Custom collate function for description dataset
    Handles batching variable-sized graph data and raw text
    """
    # Get all keys from the first item in the batch
    keys = batch[0].keys()
    
    collated_batch = {}
    
    for key in keys:
        if key in ['func_node_features', 'func_adj_matrix', 'raw_description']:
            # For graph data and text, we can't batch directly, so we keep them as a list
            collated_batch[key] = [item[key] for item in batch]
        else:
            # For other tensors, use standard batching
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            except:
                # Fallback for any tensor that can't be stacked
                collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch


def create_dataloader(dataset: Dataset, batch_size: int, 
                      is_training: bool = True, 
                      collate_fn=None,
                      num_workers: int = 4):
    """
    Create a DataLoader for the dataset
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        is_training: Whether to shuffle data
        collate_fn: Optional custom collate function
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    # Determine appropriate collate function if not provided
    if collate_fn is None:
        if isinstance(dataset, BinarySimilarityDataset):
            collate_fn = similarity_collate_fn
        elif isinstance(dataset, BinaryDescriptionDataset):
            collate_fn = description_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_training
    )