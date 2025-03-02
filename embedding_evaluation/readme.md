# Instruction Embedding Evaluation Framework

This repository contains a comprehensive framework for evaluating instruction embedding techniques in the context of binary analysis, focusing on cybersecurity applications like Binary Function Similarity Detection.

## Overview

Binary code analysis is a fundamental task in cybersecurity, and deep learning approaches have shown promising results in tasks like function similarity detection. This framework aims to systematically evaluate the quality of instruction embeddings (vector representations of assembly instructions) by measuring their performance on tasks requiring semantic understanding:

1. **Instruction Synonym Detection**: Identifying different instructions with the same effect
2. **Semantic Block Equivalence**: Recognizing different instruction sequences with the same outcome
3. **Dead Code Detection**: Identifying instructions with no semantic impact
4. **Function Boundary Detection**: Identifying function boundaries in binary code
5. **Vulnerability Pattern Detection**: Detecting code similar to known vulnerabilities

## Research Novelty

This framework introduces several novel contributions:

1. A standardized evaluation methodology for comparing instruction embeddings in binary analysis
2. Task-specific metrics that align with real-world binary analysis challenges
3. Comparative analysis of multiple embedding techniques on semantically meaningful tasks
4. Integration of NLP-inspired evaluation metrics into binary code analysis
5. Support for real-world binaries and cross-architecture analysis
6. Enhanced visualization and error analysis tools

## Supported Embedding Models

The framework includes a variety of embedding techniques:

- **Word2Vec-based**: Classical word embedding approach adapted for assembly
- **PalmTree**: Specialized contextual embedding for assembly instructions
- **BERT-based**: Transformer-based contextual embeddings
- **Graph-based**: Embedding model using control-flow and data-flow information
- **Baselines**: TF-IDF, One-hot encodings, and N-gram models

## Data Sources

The framework supports multiple data sources:

- **Synthetic Data**: Procedurally generated instruction sequences
- **Real Binary Data**: Extracted from actual compiled binaries
- **Cross-Architecture Data**: Instructions from multiple CPU architectures (x86-64, ARM, MIPS)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/instruction-embedding-evaluation.git
cd instruction-embedding-evaluation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if available)
python scripts/download_models.py
```

## Basic Usage

```bash
# Run with default settings (synthetic data, word2vec and palmtree)
python scripts/run_experiment.py

# Run enhanced experiment script with configuration file
python scripts/enhanced_run_experiment.py --config configs/real_binary_experiment.json

# Run with specific embedding models
python scripts/enhanced_run_experiment.py --embeddings word2vec,bert,graph

# Run with specific tasks
python scripts/enhanced_run_experiment.py --tasks enhanced_synonym,enhanced_block,function_boundary
```

## Configuration System

The framework includes a configuration system for reproducible experiments:

```bash
# Use a predefined configuration
python scripts/enhanced_run_experiment.py --config configs/real_binary_experiment.json

# Override config settings via command line
python scripts/enhanced_run_experiment.py --config configs/real_binary_experiment.json --seed 123 --embedding_dim 256
```

Configuration files can specify:
- Data sources and parameters
- Embedding models and hyperparameters
- Evaluation tasks
- Visualization settings
- Output directories

## Extending the Framework

### Adding New Embedding Techniques

1. Create a new class in `src/embeddings/` that inherits from `BaseEmbedding`
2. Implement the required methods: `fit()`, `transform()`, `save()`, `load()`
3. Register your model in `src/embeddings/__init__.py`

### Adding New Evaluation Tasks

1. Create a new class in `src/tasks/` that inherits from `BaseTask`
2. Implement the required methods: `evaluate()` and `score()`
3. Register your task in `src/tasks/__init__.py`

## Framework Structure

```
instruction-embedding-evaluation/
├── configs/                    # Experiment configuration files
├── data/                       # Data storage
│   ├── raw/                    # Raw binary files
│   ├── processed/              # Processed instruction data
│   ├── real/                   # Real binary data
│   └── cross_arch/             # Cross-architecture data
├── models/                     # Model storage
├── results/                    # Experiment results
├── visualizations/             # Visualizations output
├── scripts/                    # Utility scripts
│   ├── download_models.py      # Download pre-trained models
│   ├── preprocess.py           # Process binary files
│   ├── run_experiment.py       # Basic experiment script
│   ├── enhanced_run_experiment.py # Enhanced experiment script
│   └── visualize_results.py    # Generate visualizations
├── src/                        # Source code
│   ├── analysis/               # Analysis utilities
│   │   ├── visualization.py    # Visualization tools
│   │   └── error_analysis.py   # Error analysis tools
│   ├── data/                   # Data processing modules
│   │   ├── generator.py        # Synthetic data generation
│   │   ├── preprocessing.py    # Binary preprocessing
│   │   ├── real_binary_loader.py # Real binary processing
│   │   └── cross_architecture.py # Cross-architecture support
│   ├── embeddings/             # Embedding models
│   │   ├── base.py             # Base embedding class
│   │   ├── word2vec.py         # Word2Vec implementation
│   │   ├── palmtree.py         # PalmTree implementation
│   │   ├── bert_assembly.py    # BERT-based embeddings
│   │   ├── graph_embedding.py  # Graph-based embeddings
│   │   └── baseline_embeddings.py # Baseline models
│   ├── tasks/                  # Evaluation tasks
│   │   ├── base.py             # Base task class
│   │   ├── dead_code.py        # Dead code detection
│   │   ├── instruction_synonym.py # Synonym detection
│   │   ├── semantic_block.py   # Block equivalence
│   │   ├── enhanced_*.py       # Enhanced tasks
│   │   ├── function_boundary.py # Function boundary detection
│   │   └── vulnerability.py    # Vulnerability detection
│   ├── evaluation/             # Evaluation utilities
│   │   └── scorer.py           # Scoring and metrics
│   └── config.py               # Configuration system
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Advanced Features

### Real Binary Analysis

Analyze real binary files:

```bash
# Analyze binaries from GitHub repositories
python scripts/enhanced_run_experiment.py --config configs/real_binary_experiment.json
```

### Cross-Architecture Analysis

Analyze binaries across multiple CPU architectures:

```bash
# Compare x86-64, ARM, and MIPS
python scripts/enhanced_run_experiment.py --config configs/cross_architecture_experiment.json
```

### Enhanced Visualization

Generate comprehensive visualizations of embedding spaces and task performance:

```bash
# Visualize experiment results
python scripts/visualize_results.py --results_dir results/experiment_20250302_123456
```

### Error Analysis

Perform detailed error analysis to understand model weaknesses:

```bash
# Include error analysis in your experiment
python scripts/enhanced_run_experiment.py --config configs/your_config.json
```

## Citation

If you use this framework in your research, please cite:

```
@article{author2023instruction,
  title={Instruction Embedding Evaluation Framework for Binary Analysis},
  author={Author, A.},
  journal={arXiv preprint arXiv:2023.xxxx},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.