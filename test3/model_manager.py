#!/usr/bin/env python3
"""
Implementation and management of binary similarity detection models.
"""
import os
import json
import numpy as np
import pickle
from typing import Dict, List, Any, Tuple, Optional
import logging
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers, Model
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinarySimilarityModel(ABC):
    """Abstract base class for binary similarity models."""
    
    @abstractmethod
    def train(self, features_dir: str, model_save_path: str):
        """
        Train the model on extracted features.
        
        Args:
            features_dir: Directory containing JSON feature files
            model_save_path: Path to save the trained model
        """
        pass
    
    @abstractmethod
    def predict_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Predict similarity between two binaries.
        
        Args:
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Similarity score (0-1)
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        pass
    
    @abstractmethod
    def save(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        pass

class GraphBasedSimilarity(BinarySimilarityModel):
    """Implements a graph-based binary similarity approach similar to BinDiff."""
    
    def __init__(self):
        """Initialize the graph-based similarity model."""
        self.function_features = {}
        
    def train(self, features_dir: str, model_save_path: str):
        """
        Train the model on extracted features.
        
        Args:
            features_dir: Directory containing JSON feature files
            model_save_path: Path to save the trained model
        """
        # Graph-based models typically don't need training
        # They use direct comparison of features
        logger.info("Graph-based model doesn't require training phase")
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        self.save(model_save_path)
    
    def predict_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Predict similarity between two binaries based on structural properties.
        
        Args:
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Similarity score (0-1)
        """
        # Build function graphs for both binaries
        g1 = self._build_callgraph(features1)
        g2 = self._build_callgraph(features2)
        
        # Match functions based on initial attributes
        matched_funcs = self._match_functions(features1, features2)
        
        # Calculate similarity based on matched functions
        similarity = len(matched_funcs) / max(len(features1["functions"]), len(features2["functions"]))
        
        # Enhance similarity score with graph similarity
        if g1.number_of_nodes() > 0 and g2.number_of_nodes() > 0:
            # Calculate graph edit distance (normalized)
            try:
                graph_sim = self._graph_similarity(g1, g2, matched_funcs)
                # Combine structural and content similarity
                similarity = 0.5 * similarity + 0.5 * graph_sim
            except Exception as e:
                logger.warning(f"Error calculating graph similarity: {e}")
        
        return similarity
    
    def _build_callgraph(self, features: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a callgraph from binary features.
        
        Args:
            features: Binary features
            
        Returns:
            NetworkX directed graph of the callgraph
        """
        G = nx.DiGraph()
        
        # Add nodes (functions)
        for func in features["functions"]:
            # Calculate node attributes (simplified)
            num_inst = len(func["instructions"])
            num_blocks = len(func["basic_blocks"])
            calls = len([i for i in func["instructions"] if "call" in i.get("disasm", "")])
            
            # Add node with attributes
            G.add_node(func["offset"], 
                      name=func["name"],
                      size=func["size"],
                      num_inst=num_inst,
                      num_blocks=num_blocks,
                      calls=calls)
        
        # Add edges (function calls)
        for func in features["functions"]:
            for call in func["callgraph"]:
                if call["from"] in G and call["to"] in G:
                    G.add_edge(call["from"], call["to"])
        
        return G
    
    def _match_functions(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[int, int]:
        """
        Match functions between two binaries.
        
        Args:
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Dictionary mapping function offsets from binary1 to binary2
        """
        matches = {}
        
        # Extract functions from both binaries
        funcs1 = features1["functions"]
        funcs2 = features2["functions"]
        
        # First pass: match by name (if available)
        name_matches = {}
        for i, f1 in enumerate(funcs1):
            if "name" in f1 and f1["name"] and not f1["name"].startswith("fcn."):
                for j, f2 in enumerate(funcs2):
                    if "name" in f2 and f2["name"] == f1["name"]:
                        name_matches[f1["offset"]] = f2["offset"]
                        break
        
        # Second pass: match by structure similarity
        for i, f1 in enumerate(funcs1):
            if f1["offset"] in name_matches:
                matches[f1["offset"]] = name_matches[f1["offset"]]
                continue
                
            best_match = None
            best_score = 0.6  # Threshold for matching
            
            for j, f2 in enumerate(funcs2):
                if f2["offset"] in matches.values():
                    continue  # Already matched
                
                # Calculate function similarity based on basic features
                sim_score = self._calculate_function_similarity(f1, f2)
                
                if sim_score > best_score:
                    best_score = sim_score
                    best_match = f2["offset"]
            
            if best_match is not None:
                matches[f1["offset"]] = best_match
        
        return matches
    
    def _calculate_function_similarity(self, func1: Dict[str, Any], func2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two functions based on features.
        
        Args:
            func1: Features of the first function
            func2: Features of the second function
            
        Returns:
            Similarity score (0-1)
        """
        # Feature 1: Size similarity
        if func1["size"] == 0 or func2["size"] == 0:
            size_sim = 0
        else:
            size_ratio = min(func1["size"], func2["size"]) / max(func1["size"], func2["size"])
            size_sim = size_ratio
        
        # Feature 2: Number of basic blocks
        if not func1["basic_blocks"] or not func2["basic_blocks"]:
            bb_sim = 0
        else:
            bb_ratio = min(len(func1["basic_blocks"]), len(func2["basic_blocks"])) / \
                      max(len(func1["basic_blocks"]), len(func2["basic_blocks"]))
            bb_sim = bb_ratio
        
        # Feature 3: Number of instructions
        if not func1["instructions"] or not func2["instructions"]:
            inst_sim = 0
        else:
            inst_ratio = min(len(func1["instructions"]), len(func2["instructions"])) / \
                        max(len(func1["instructions"]), len(func2["instructions"]))
            inst_sim = inst_ratio
        
        # Feature 4: Instruction n-gram similarity (simplified)
        if not func1["instructions"] or not func2["instructions"]:
            ngram_sim = 0
        else:
            ngram_sim = self._calculate_instruction_similarity(func1["instructions"], func2["instructions"])
        
        # Weighted combination
        final_sim = 0.2 * size_sim + 0.2 * bb_sim + 0.2 * inst_sim + 0.4 * ngram_sim
        
        return final_sim
    
    def _calculate_instruction_similarity(self, insts1: List[Dict[str, Any]], insts2: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between two instruction sequences.
        
        Args:
            insts1: Instructions of the first function
            insts2: Instructions of the second function
            
        Returns:
            Similarity score (0-1)
        """
        # Extract operation types only (for simplicity)
        ops1 = [inst.get("disasm", "").split()[0] if " " in inst.get("disasm", "") else inst.get("disasm", "") 
                for inst in insts1]
        ops2 = [inst.get("disasm", "").split()[0] if " " in inst.get("disasm", "") else inst.get("disasm", "") 
                for inst in insts2]
        
        # Count common operations
        op_set1 = set(ops1)
        op_set2 = set(ops2)
        common_ops = op_set1.intersection(op_set2)
        
        op_sim = len(common_ops) / max(len(op_set1), len(op_set2)) if op_set1 or op_set2 else 0
        
        return op_sim
    
    def _graph_similarity(self, g1: nx.DiGraph, g2: nx.DiGraph, matches: Dict[int, int]) -> float:
        """
        Calculate structural similarity between two callgraphs.
        
        Args:
            g1: Callgraph of the first binary
            g2: Callgraph of the second binary
            matches: Dictionary of matched functions
            
        Returns:
            Similarity score (0-1)
        """
        # Calculate a simple graph similarity measure based on matched nodes
        if not matches:
            return 0
        
        # Count how many edges in g1 are preserved in g2
        preserved_edges = 0
        total_edges = g1.number_of_edges()
        
        for src, dst in g1.edges():
            if src in matches and dst in matches:
                src_match = matches[src]
                dst_match = matches[dst]
                if g2.has_edge(src_match, dst_match):
                    preserved_edges += 1
        
        if total_edges == 0:
            return 0
            
        return preserved_edges / total_edges
    
    def load(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        # Graph-based models typically don't have parameters to load
        logger.info(f"Loaded graph-based model from {model_path}")
    
    def save(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        # Graph-based models typically don't have parameters to save
        with open(model_path, 'w') as f:
            f.write("Graph-based similarity model - no parameters to save")
        logger.info(f"Saved graph-based model to {model_path}")


class Asm2VecModel(BinarySimilarityModel):
    """Implements a simplified version of the Asm2Vec approach."""
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize the Asm2Vec model.
        
        Args:
            embedding_dim: Dimension of instruction embeddings
        """
        self.embedding_dim = embedding_dim
        self.instruction_embeddings = {}
        self.function_embeddings = {}
        self.vocab = set()
    
    def train(self, features_dir: str, model_save_path: str):
        """
        Train the model on extracted features.
        
        Args:
            features_dir: Directory containing JSON feature files
            model_save_path: Path to save the trained model
        """
        logger.info("Training Asm2Vec model...")
        
        # Collect instructions from all feature files to build vocabulary
        json_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.json')]
        
        # Build vocabulary
        logger.info("Building vocabulary...")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                features = json.load(f)
                
            for func in features["functions"]:
                for inst in func["instructions"]:
                    if "disasm" in inst:
                        # Tokenize instruction
                        tokens = self._tokenize_instruction(inst["disasm"])
                        self.vocab.update(tokens)
        
        # Initialize embeddings randomly
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        np.random.seed(42)
        for token in self.vocab:
            self.instruction_embeddings[token] = np.random.rand(self.embedding_dim)
        
        # Learn embeddings using simplified skip-gram approach
        logger.info("Learning embeddings...")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                features = json.load(f)
                
            # Process each function
            for func in features["functions"]:
                if not func["instructions"]:
                    continue
                
                # Generate function embedding
                func_embedding = self._generate_function_embedding(func)
                if func_embedding is not None:
                    self.function_embeddings[f"{features['binary_path']}:{func['offset']}"] = func_embedding
        
        # Save model
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        self.save(model_save_path)
        logger.info("Asm2Vec training completed")
    
    def _tokenize_instruction(self, disasm: str) -> List[str]:
        """
        Tokenize an assembly instruction.
        
        Args:
            disasm: Disassembled instruction string
            
        Returns:
            List of tokens
        """
        # Simple tokenization by splitting on whitespace and removing common separators
        clean_disasm = disasm.replace(',', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')
        return [token.lower() for token in clean_disasm.split() if token]
    
    def _generate_function_embedding(self, func: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Generate embedding for a function.
        
        Args:
            func: Function features
            
        Returns:
            Function embedding vector
        """
        if not func["instructions"]:
            return None
            
        # Extract tokenized instructions
        tokenized_instructions = []
        for inst in func["instructions"]:
            if "disasm" in inst:
                tokens = self._tokenize_instruction(inst["disasm"])
                if tokens:
                    tokenized_instructions.append(tokens)
        
        if not tokenized_instructions:
            return None
            
        # Calculate function embedding by averaging token embeddings
        func_embedding = np.zeros(self.embedding_dim)
        token_count = 0
        
        for tokens in tokenized_instructions:
            for token in tokens:
                if token in self.instruction_embeddings:
                    func_embedding += self.instruction_embeddings[token]
                    token_count += 1
        
        if token_count > 0:
            func_embedding /= token_count
            
        return func_embedding
    
    def predict_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Predict similarity between two binaries based on function embeddings.
        
        Args:
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Similarity score (0-1)
        """
        # First, generate embeddings for all functions
        func_embeddings1 = {}
        func_embeddings2 = {}
        
        for func in features1["functions"]:
            emb = self._generate_function_embedding(func)
            if emb is not None:
                func_embeddings1[func["offset"]] = emb
                
        for func in features2["functions"]:
            emb = self._generate_function_embedding(func)
            if emb is not None:
                func_embeddings2[func["offset"]] = emb
        
        if not func_embeddings1 or not func_embeddings2:
            return 0.0
        
        # Calculate similarity matrix
        matches = []
        similarity_sum = 0.0
        
        # Find best matching functions
        for offset1, emb1 in func_embeddings1.items():
            best_sim = 0.0
            best_match = None
            
            for offset2, emb2 in func_embeddings2.items():
                if offset2 in [m[1] for m in matches]:
                    continue  # Already matched
                
                sim = cosine_similarity([emb1], [emb2])[0][0]
                if sim > best_sim and sim > 0.7:  # Threshold
                    best_sim = sim
                    best_match = offset2
            
            if best_match is not None:
                matches.append((offset1, best_match))
                similarity_sum += best_sim
        
        # Calculate final score based on matched functions
        if not matches:
            return 0.0
            
        # Normalize by total number of functions
        final_sim = len(matches) / max(len(func_embeddings1), len(func_embeddings2))
        
        # Factor in average similarity of matches
        avg_match_sim = similarity_sum / len(matches)
        
        return 0.7 * final_sim + 0.3 * avg_match_sim
    
    def load(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.embedding_dim = model_data["embedding_dim"]
        self.instruction_embeddings = model_data["instruction_embeddings"]
        self.function_embeddings = model_data["function_embeddings"]
        self.vocab = model_data["vocab"]
        
        logger.info(f"Loaded Asm2Vec model from {model_path}")
    
    def save(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        model_data = {
            "embedding_dim": self.embedding_dim,
            "instruction_embeddings": self.instruction_embeddings,
            "function_embeddings": self.function_embeddings,
            "vocab": self.vocab
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Saved Asm2Vec model to {model_path}")


class GeminiModel(BinarySimilarityModel):
    """Implements a simplified version of the Gemini neural network approach."""
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize the Gemini model.
        
        Args:
            embedding_dim: Dimension of function embeddings
        """
        self.embedding_dim = embedding_dim
        self.model = None
        self.function_embeddings = {}
    
    def _build_model(self):
        """Build and compile the neural network model."""
        # Define input features
        instruction_count = layers.Input(shape=(1,), name="instruction_count")
        block_count = layers.Input(shape=(1,), name="block_count")
        string_count = layers.Input(shape=(1,), name="string_count")
        const_count = layers.Input(shape=(1,), name="const_count")
        call_count = layers.Input(shape=(1,), name="call_count")
        
        # Create a simple feed-forward network
        merged = layers.concatenate([
            instruction_count, block_count, string_count, const_count, call_count
        ])
        
        dense1 = layers.Dense(32, activation="relu")(merged)
        dense2 = layers.Dense(64, activation="relu")(dense1)
        output = layers.Dense(self.embedding_dim, activation="tanh")(dense2)
        
        model = Model(
            inputs=[instruction_count, block_count, string_count, const_count, call_count],
            outputs=output
        )
        
        model.compile(optimizer="adam", loss="mse")
        return model
    
    def train(self, features_dir: str, model_save_path: str):
        """
        Train the model on extracted features.
        
        Args:
            features_dir: Directory containing JSON feature files
            model_save_path: Path to save the trained model
        """
        logger.info("Training Gemini model...")
        
        # Build model
        self.model = self._build_model()
        
        # Collect function features from all feature files
        json_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.json')]
        
        # Extract function features
        X = {
            "instruction_count": [],
            "block_count": [],
            "string_count": [],
            "const_count": [],
            "call_count": []
        }
        function_ids = []
        
        logger.info("Extracting function features...")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                features = json.load(f)
                
            for func in features["functions"]:
                # Extract features for each function
                instruction_count = len(func["instructions"])
                block_count = len(func["basic_blocks"])
                
                # Count strings (simplified)
                string_count = sum(1 for inst in func["instructions"] if ".str" in inst.get("disasm", ""))
                
                # Count constants (simplified)
                const_count = sum(1 for inst in func["instructions"] 
                                 if any(c.isdigit() for c in inst.get("disasm", "")))
                
                # Count calls
                call_count = sum(1 for inst in func["instructions"] 
                                if "call" in inst.get("disasm", ""))
                
                # Skip very small functions
                if instruction_count < 5:
                    continue
                
                X["instruction_count"].append(instruction_count)
                X["block_count"].append(block_count)
                X["string_count"].append(string_count)
                X["const_count"].append(const_count)
                X["call_count"].append(call_count)
                
                function_ids.append(f"{features['binary_path']}:{func['offset']}")
        
        if not function_ids:
            logger.warning("No functions found for training. Check feature extraction.")
            return
        
        # Convert to numpy arrays
        for key in X:
            X[key] = np.array(X[key], dtype=np.float32)
            # Normalize
            if np.max(X[key]) > 0:
                X[key] = X[key] / np.max(X[key])
        
        # Train model to learn embeddings (autoencoder-like approach)
        # Since we don't have labels, we'll train to reconstruct the input
        logger.info("Training the model...")
        self.model.fit(
            x=X,
            y=np.zeros((len(function_ids), self.embedding_dim)),  # Dummy target
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # Generate embeddings for all functions
        logger.info("Generating function embeddings...")
        embeddings = self.model.predict(X)
        
        for i, func_id in enumerate(function_ids):
            self.function_embeddings[func_id] = embeddings[i]
        
        # Save model
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        self.save(model_save_path)
        logger.info("Gemini training completed")
    
    def predict_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Predict similarity between two binaries based on function embeddings.
        
        Args:
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Similarity score (0-1)
        """
        if self.model is None:
            logger.error("Model not trained or loaded. Cannot predict similarity.")
            return 0.0
        
        # Generate embeddings for all functions in both binaries
        embeddings1 = self._generate_embeddings(features1)
        embeddings2 = self._generate_embeddings(features2)
        
        if not embeddings1 or not embeddings2:
            return 0.0
        
        # Calculate similarity matrix between all function pairs
        similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))
        
        for i, (func_id1, emb1) in enumerate(embeddings1.items()):
            for j, (func_id2, emb2) in enumerate(embeddings2.items()):
                # Calculate cosine similarity
                similarity_matrix[i, j] = cosine_similarity([emb1], [emb2])[0][0]
        
        # Find best matches with a simple greedy algorithm
        matched_pairs = []
        similarity_sum = 0.0
        
        while similarity_matrix.size > 0:
            # Find highest similarity
            max_idx = np.argmax(similarity_matrix)
            i, j = np.unravel_index(max_idx, similarity_matrix.shape)
            
            if similarity_matrix[i, j] < 0.7:  # Threshold
                break
                
            # Get function IDs
            func_id1 = list(embeddings1.keys())[i]
            func_id2 = list(embeddings2.keys())[j]
            
            matched_pairs.append((func_id1, func_id2))
            similarity_sum += similarity_matrix[i, j]
            
            # Remove matched rows and columns
            similarity_matrix = np.delete(similarity_matrix, i, axis=0)
            similarity_matrix = np.delete(similarity_matrix, j, axis=1)
            
            # Update embeddings dictionaries
            embeddings1 = {k: v for idx, (k, v) in enumerate(embeddings1.items()) if idx != i}
            embeddings2 = {k: v for idx, (k, v) in enumerate(embeddings2.items()) if idx != j}
        
        # Calculate final similarity score
        if not matched_pairs:
            return 0.0
            
        # Based on matched functions and their similarity
        match_ratio = len(matched_pairs) / max(len(features1["functions"]), len(features2["functions"]))
        avg_similarity = similarity_sum / len(matched_pairs)
        
        # Combine the two metrics
        final_similarity = 0.6 * match_ratio + 0.4 * avg_similarity
        
        return final_similarity
    
    def _generate_embeddings(self, features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all functions in a binary.
        
        Args:
            features: Binary features
            
        Returns:
            Dictionary mapping function IDs to embeddings
        """
        embeddings = {}
        
        # Prepare input features
        X = {
            "instruction_count": [],
            "block_count": [],
            "string_count": [],
            "const_count": [],
            "call_count": []
        }
        func_ids = []
        
        for func in features["functions"]:
            # Extract features for each function
            instruction_count = len(func["instructions"])
            block_count = len(func["basic_blocks"])
            
            # Skip very small functions
            if instruction_count < 5:
                continue
            
            # Count strings (simplified)
            string_count = sum(1 for inst in func["instructions"] if ".str" in inst.get("disasm", ""))
            
            # Count constants (simplified)
            const_count = sum(1 for inst in func["instructions"] 
                             if any(c.isdigit() for c in inst.get("disasm", "")))
            
            # Count calls
            call_count = sum(1 for inst in func["instructions"] 
                            if "call" in inst.get("disasm", ""))
            
            X["instruction_count"].append(instruction_count)
            X["block_count"].append(block_count)
            X["string_count"].append(string_count)
            X["const_count"].append(const_count)
            X["call_count"].append(call_count)
            
            func_ids.append(f"{features['binary_path']}:{func['offset']}")
        
        if not func_ids:
            return embeddings
        
        # Convert to numpy arrays and normalize
        for key in X:
            X[key] = np.array(X[key], dtype=np.float32)
            # Normalize
            if np.max(X[key]) > 0:
                X[key] = X[key] / np.max(X[key])
        
        # Generate embeddings
        predicted_embeddings = self.model.predict(X)
        
        for i, func_id in enumerate(func_ids):
            embeddings[func_id] = predicted_embeddings[i]
        
        return embeddings
    
    def load(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        model_dir = os.path.dirname(model_path)
        
        # Load model architecture and weights
        self.model = tf.keras.models.load_model(f"{model_dir}/gemini_model")
        
        # Load function embeddings
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.embedding_dim = model_data["embedding_dim"]
        self.function_embeddings = model_data["function_embeddings"]
        
        logger.info(f"Loaded Gemini model from {model_path}")
    
    def save(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        model_dir = os.path.dirname(model_path)
        
        # Save Keras model
        if self.model is not None:
            self.model.save(f"{model_dir}/gemini_model")
        
        # Save embeddings and other data
        model_data = {
            "embedding_dim": self.embedding_dim,
            "function_embeddings": self.function_embeddings
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Saved Gemini model to {model_path}")


class ModelManager:
    """Manages different binary similarity models."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        
    def add_model(self, name: str, model: BinarySimilarityModel):
        """
        Add a model to the manager.
        
        Args:
            name: Name of the model
            model: Binary similarity model
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def train_model(self, name: str, features_dir: str, model_save_path: str):
        """
        Train a specific model.
        
        Args:
            name: Name of the model
            features_dir: Directory containing feature files
            model_save_path: Path to save the trained model
        """
        if name not in self.models:
            logger.error(f"Model {name} not found")
            return
            
        self.models[name].train(features_dir, model_save_path)
    
    def load_model(self, name: str, model_path: str):
        """
        Load a trained model.
        
        Args:
            name: Name of the model
            model_path: Path to the saved model
        """
        if name not in self.models:
            logger.error(f"Model {name} not found")
            return
            
        self.models[name].load(model_path)
    
    def predict_similarity(self, name: str, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Predict similarity using a specific model.
        
        Args:
            name: Name of the model
            features1: Features of the first binary
            features2: Features of the second binary
            
        Returns:
            Similarity score (0-1)
        """
        if name not in self.models:
            logger.error(f"Model {name} not found")
            return 0.0
            
        return self.models[name].predict_similarity(features1, features2)


def main():
    # Example usage
    parser = argparse.ArgumentParser(description="Binary similarity model manager")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--features-dir", help="Directory containing feature files")
    parser.add_argument("--model-dir", help="Directory to save/load model files")
    
    args = parser.parse_args()
    
    # Create model manager and add models
    manager = ModelManager()
    manager.add_model("graph", GraphBasedSimilarity())
    manager.add_model("asm2vec", Asm2VecModel())
    manager.add_model("gemini", GeminiModel())
    
    if args.train and args.features_dir and args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Train all models
        for model_name in manager.models:
            model_path = os.path.join(args.model_dir, f"{model_name}_model.pkl")
            manager.train_model(model_name, args.features_dir, model_path)

if __name__ == "__main__":
    main()