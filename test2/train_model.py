#!/usr/bin/env python3
"""
Train a binary similarity model on source architecture.
"""

import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import config
from utils import logger, create_directories, set_random_seed, save_json, load_json

class InstructionTokenizer:
    """Tokenizer for instructions"""
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = {}
        self.num_words = 2  # Start with padding and unknown tokens
        self.fitted = False
    
    def fit(self, instructions):
        """Fit the tokenizer on a list of instructions"""
        # Count word frequencies
        for instruction in instructions:
            for token in instruction.split():
                if token not in self.word_counts:
                    self.word_counts[token] = 0
                self.word_counts[token] += 1
        
        # Create vocabulary (sort by frequency)
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Assign indices to words
        for word, count in sorted_words:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.num_words += 1
        
        self.fitted = True
        logger.info(f"Tokenizer fitted with {self.num_words} unique tokens")
    
    def tokenize(self, instruction):
        """Convert an instruction to a list of token indices"""
        if not self.fitted:
            raise ValueError("Tokenizer must be fitted before tokenizing")
        
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in instruction.split()]
    
    def save(self, file_path):
        """Save the tokenizer to a file"""
        tokenizer_data = {
            "word2idx": self.word2idx,
            "idx2word": {int(k): v for k, v in self.idx2word.items()},
            "word_counts": self.word_counts,
            "num_words": self.num_words
        }
        with open(file_path, 'w') as f:
            json.dump(tokenizer_data, f)
    
    @classmethod
    def load(cls, file_path):
        """Load a tokenizer from a file"""
        with open(file_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls()
        tokenizer.word2idx = tokenizer_data["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in tokenizer_data["idx2word"].items()}
        tokenizer.word_counts = tokenizer_data["word_counts"]
        tokenizer.num_words = tokenizer_data["num_words"]
        tokenizer.fitted = True
        
        return tokenizer

class FunctionPairGenerator(Sequence):
    """Data generator for function pairs"""
    def __init__(self, function_pairs, tokenizer, batch_size=32, max_length=150, training=True):
        self.function_pairs = function_pairs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.training = training
        self.indices = np.arange(len(self.function_pairs))
        if self.training:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.function_pairs) // self.batch_size
    
    def __getitem__(self, idx):
        """Get a batch of function pairs"""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        func1_batch = np.zeros((self.batch_size, self.max_length), dtype=np.int32)
        func2_batch = np.zeros((self.batch_size, self.max_length), dtype=np.int32)
        labels_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
        
        # Fill batch arrays
        for i, idx in enumerate(batch_indices):
            pair = self.function_pairs[idx]
            func1 = pair["func1"]
            func2 = pair["func2"]
            label = pair["label"]
            
            # Tokenize and pad instructions
            func1_tokens = []
            for instr in func1["instructions"][:self.max_length]:
                func1_tokens.extend(self.tokenizer.tokenize(instr))
            func1_tokens = func1_tokens[:self.max_length]
            
            func2_tokens = []
            for instr in func2["instructions"][:self.max_length]:
                func2_tokens.extend(self.tokenizer.tokenize(instr))
            func2_tokens = func2_tokens[:self.max_length]
            
            # Pad sequences
            func1_batch[i, :len(func1_tokens)] = func1_tokens
            func2_batch[i, :len(func2_tokens)] = func2_tokens
            labels_batch[i] = label
        
        return [func1_batch, func2_batch], labels_batch
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.training:
            np.random.shuffle(self.indices)

def load_functions(architecture, compiler, opt_level, project=None):
    """Load functions for a specific configuration"""
    functions = []
    
    # Determine function file paths
    if project:
        project_data = next((p for p in config.PROJECTS if p["name"] == project), None)
        if not project_data:
            logger.error(f"Project {project} not found")
            return []
        
        project_name = f"{project_data['name']}-{project_data['version']}"
        function_paths = [os.path.join(config.FUNCTION_DIR, project_name, architecture, compiler, opt_level, "functions.json")]
    else:
        # Load functions from all projects
        function_paths = glob.glob(os.path.join(config.FUNCTION_DIR, "*", architecture, compiler, opt_level, "functions.json"))
    
    # Load functions from each file
    for function_path in function_paths:
        if os.path.exists(function_path):
            functions.extend(load_json(function_path))
    
    logger.info(f"Loaded {len(functions)} functions for {architecture} with {compiler} -{opt_level}")
    return functions

def create_function_pairs(source_functions, target_functions=None, similar_ratio=0.5, max_pairs=10000):
    """Create function pairs for training/testing"""
    function_pairs = []
    
    # If target functions are not provided, use source functions for both
    if target_functions is None:
        target_functions = source_functions
    
    # Identify similar function pairs (from cross-compilation)
    similar_pairs = []
    for source_func in source_functions:
        source_name = source_func["name"]
        for target_func in target_functions:
            target_name = target_func["name"]
            if source_name == target_name:
                similar_pairs.append((source_func, target_func))
    
    # Create similar pairs up to the desired number
    num_similar = int(max_pairs * similar_ratio)
    similar_pairs = similar_pairs[:num_similar]
    
    # Add similar pairs to the result
    for source_func, target_func in similar_pairs:
        function_pairs.append({
            "func1": source_func,
            "func2": target_func,
            "label": 1.0
        })
    
    # Create dissimilar pairs
    num_dissimilar = max_pairs - len(similar_pairs)
    dissimilar_pairs = []
    
    # Randomly select source and target functions
    source_indices = random.sample(range(len(source_functions)), min(num_dissimilar, len(source_functions)))
    target_indices = random.sample(range(len(target_functions)), min(num_dissimilar, len(target_functions)))
    
    # Create dissimilar pairs
    for i, source_idx in enumerate(source_indices):
        if i >= len(target_indices):
            break
        
        source_func = source_functions[source_idx]
        target_func = target_functions[target_indices[i]]
        
        # Skip if the functions are actually similar
        if source_func["name"] == target_func["name"]:
            continue
        
        dissimilar_pairs.append({
            "func1": source_func,
            "func2": target_func,
            "label": 0.0
        })
    
    # Add dissimilar pairs to the result
    function_pairs.extend(dissimilar_pairs)
    
    # Shuffle the result
    random.shuffle(function_pairs)
    
    logger.info(f"Created {len(function_pairs)} function pairs "
                f"({len(similar_pairs)} similar, {len(dissimilar_pairs)} dissimilar)")
    
    return function_pairs

def create_model(vocab_size, embedding_dim=100, max_length=150):
    """Create a Siamese network model for function similarity detection"""
    # Input layers
    func1_input = layers.Input(shape=(max_length,), name="func1_input")
    func2_input = layers.Input(shape=(max_length,), name="func2_input")
    
    # Shared embedding layer
    embedding_layer = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        name="instruction_embedding"
    )
    
    # Shared LSTM layer
    lstm_layer = layers.LSTM(128, name="lstm_encoder")
    
    # Process each function
    func1_embedding = embedding_layer(func1_input)
    func2_embedding = embedding_layer(func2_input)
    
    func1_encoded = lstm_layer(func1_embedding)
    func2_encoded = lstm_layer(func2_embedding)
    
    # Calculate similarity
    distance = layers.Lambda(
        lambda x: tf.keras.backend.abs(x[0] - x[1]),
        name="distance"
    )([func1_encoded, func2_encoded])
    
    # Dense layers for classification
    dense1 = layers.Dense(64, activation='relu', name="dense1")(distance)
    dropout1 = layers.Dropout(0.3, name="dropout1")(dense1)
    output = layers.Dense(1, activation='sigmoid', name="similarity")(dropout1)
    
    # Create model
    model = models.Model(inputs=[func1_input, func2_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model_on_source(source_arch, source_compiler, source_opt, output_dir, args):
    """Train a model on source architecture"""
    # Set random seed
    set_random_seed()
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load functions
    source_functions = load_functions(source_arch, source_compiler, source_opt)
    
    if len(source_functions) == 0:
        logger.error(f"No functions found for {source_arch} with {source_compiler} -{source_opt}")
        return None, None
    
    # Create function pairs
    function_pairs = create_function_pairs(
        source_functions,
        similar_ratio=config.SIMILAR_PAIRS_RATIO,
        max_pairs=config.MAX_FUNCTIONS_TOTAL
    )
    
    # Split into training and validation sets
    train_pairs, val_pairs = train_test_split(
        function_pairs,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    # Collect all instructions for tokenizer
    all_instructions = []
    for pair in function_pairs:
        all_instructions.extend(pair["func1"]["instructions"])
        all_instructions.extend(pair["func2"]["instructions"])
    
    # Create and fit tokenizer
    tokenizer = InstructionTokenizer()
    tokenizer.fit(all_instructions)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Create data generators
    train_generator = FunctionPairGenerator(
        train_pairs,
        tokenizer,
        batch_size=args.batch_size,
        max_length=config.SEQUENCE_LENGTH,
        training=True
    )
    
    val_generator = FunctionPairGenerator(
        val_pairs,
        tokenizer,
        batch_size=args.batch_size,
        max_length=config.SEQUENCE_LENGTH,
        training=False
    )
    
    # Create model
    model = create_model(
        vocab_size=tokenizer.num_words,
        embedding_dim=config.EMBEDDING_DIM,
        max_length=config.SEQUENCE_LENGTH
    )
    
    # Create callbacks
    checkpoint_path = os.path.join(output_dir, "model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=os.path.join(output_dir, "logs"),
        histogram_freq=1
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, tensorboard],
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    # Load the best model
    model.load_weights(checkpoint_path)
    
    # Save model summary
    model_summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(model_summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Evaluate model on validation set
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save final model and metadata
    metadata = {
        "source_arch": source_arch,
        "source_compiler": source_compiler,
        "source_opt": source_opt,
        "embedding_dim": config.EMBEDDING_DIM,
        "sequence_length": config.SEQUENCE_LENGTH,
        "vocab_size": tokenizer.num_words,
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss)
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return model, tokenizer

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train a binary similarity model')
    parser.add_argument('--source-arch', type=str, default='x86_64',
                        help='Source architecture to train on')
    parser.add_argument('--source-compiler', type=str, default='gcc',
                        help='Source compiler to train on')
    parser.add_argument('--source-opt', type=str, default='O2',
                        help='Source optimization level to train on')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs for training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source_arch not in config.ARCHITECTURES:
        logger.error(f"Architecture {args.source_arch} not supported")
        return
    
    if args.source_compiler not in config.COMPILERS:
        logger.error(f"Compiler {args.source_compiler} not supported")
        return
    
    if args.source_opt not in config.OPTIMIZATION_LEVELS:
        logger.error(f"Optimization level {args.source_opt} not supported")
        return
    
    # Create directories
    create_directories()
    
    # Define model output directory
    model_id = f"{args.source_arch}_{args.source_compiler}_{args.source_opt}"
    output_dir = os.path.join(config.MODEL_DIR, model_id)
    
    logger.info(f"Training model on {args.source_arch} with {args.source_compiler} -{args.source_opt}")
    
    # Train model
    model, tokenizer = train_model_on_source(
        args.source_arch,
        args.source_compiler,
        args.source_opt,
        output_dir,
        args
    )
    
    if model is None:
        logger.error("Model training failed")
        return
    
    logger.info(f"Model trained successfully. Saved to {output_dir}")

if __name__ == "__main__":
    main()