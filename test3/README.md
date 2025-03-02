# Binary Similarity Robustness Assessment Framework

This framework allows you to assess the robustness of binary code similarity detection models against various obfuscation techniques. The implementation evaluates how well different similarity detection approaches (traditional graph-based, embedding-based, and neural network-based) can handle obfuscated code.

## Overview

The framework consists of several components:

1. **Obfuscator** - Generates obfuscated binaries from source code
2. **Feature Extractor** - Extracts features from binaries for similarity analysis
3. **Model Manager** - Implements and manages different similarity detection models
4. **Evaluator** - Measures how well each model handles obfuscation
5. **Experiment Runner** - Coordinates the complete workflow

## Requirements

- Python 3.7+
- Radare2 for binary analysis
- Tigress or OLLVM for code obfuscation (optional but recommended)
- Python packages: numpy, pandas, matplotlib, seaborn, networkx, tensorflow

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn networkx tensorflow r2pipe