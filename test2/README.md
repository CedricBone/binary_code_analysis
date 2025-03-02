# Compiler Optimization Impact Analysis

This repository contains an experimental framework to analyze how compiler optimization levels affect binary code similarity across different architectures. The experiment evaluates whether a machine learning model trained on one architecture can effectively transfer knowledge to other architectures, and how compiler optimization settings affect this transferability.

## Overview

The core research question this experiment addresses is:

**How do compiler optimization levels affect cross-architecture binary code similarity detection?**

For security applications like vulnerability detection and malware analysis, understanding how well deep learning models transfer across architectures and compiler settings is crucial. This experiment provides empirical evidence on:

1. How well a model trained on one architecture (e.g., x86) transfers to other architectures (ARM, MIPS, PowerPC)
2. Which optimization levels preserve semantic similarities best across architectures
3. How different compilers (gcc, clang) affect the transferability of knowledge

## Methodology

The experiment follows these steps:

1. **Compilation Phase**: Compiles several open-source projects with different compilers and optimization levels for multiple architectures
2. **Function Extraction**: Disassembles compiled binaries and extracts functions
3. **Model Training**: Trains a deep learning model on functions from a source architecture
4. **Cross-Architecture Testing**: Tests the model on functions from other architectures
5. **Analysis**: Visualizes and analyzes the impact of optimization levels on model performance

## File Structure

| File                  | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `config.py`           | Central configuration for the experiment                     |
| `utils.py`            | Utility functions used across scripts                        |
| `compile_projects.py` | Compiles projects with different configurations              |
| `extract_functions.py`| Extracts functions from compiled binaries                    |
| `train_model.py`      | Trains a binary similarity model on source architecture      |
| `test_model.py`       | Tests trained model on different architectures and optimizations |
| `analyze_results.py`  | Analyzes and visualizes the experiment results              |
| `run_experiment.sh`   | Shell script to run the complete experiment                  |

The experiment creates these directories:

| Directory      | Description                                         |
|----------------|-----------------------------------------------------|
| `build/`       | Contains compiled binaries                          |
| `downloads/`   | Downloaded source code                              |
| `functions/`   | Extracted functions as JSON files                   |
| `models/`      | Trained models and tokenizers                       |
| `results/`     | Evaluation results and visualizations               |
| `logs/`        | Log files                                           |

## Setup

### Prerequisites

1. **Python 3.7+** with these packages:
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn pandas requests tqdm
   ```

2. **Compilers**:
   - GCC with cross-compilation support for ARM, MIPS, and PowerPC
   - Clang with cross-compilation support

   On Ubuntu/Debian, you can install these with:
   ```bash
   sudo apt update
   sudo apt install gcc g++ clang 
   sudo apt install gcc-arm-linux-gnueabi gcc-mips-linux-gnu gcc-powerpc-linux-gnu
   ```

3. **Disassemblers and Build Tools**:
   ```bash
   sudo apt install binutils-arm-linux-gnueabi binutils-mips-linux-gnu binutils-powerpc-linux-gnu
   sudo apt install build-essential make
   ```

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/compiler-optimization-analysis.git
   cd compiler-optimization-analysis
   ```

2. Make scripts executable:
   ```bash
   chmod +x *.py run_experiment.sh
   ```

## Usage

### Running the Complete Experiment

To run the complete experiment with default settings:

```bash
./run_experiment.sh
```

The script will:
1. Compile projects for all architectures with different optimization levels
2. Extract functions from compiled binaries
3. Train a model on x86_64 with gcc -O2
4. Test the model on all architectures, compilers, and optimization levels
5. Analyze and visualize the results

### Running Individual Steps

You can also run each step individually:

1. **Compile projects**:
   ```bash
   python compile_projects.py --parallel 4
   ```

2. **Extract functions**:
   ```bash
   python extract_functions.py --parallel 4
   ```

3. **Train model**:
   ```bash
   python train_model.py --source-arch x86_64 --source-compiler gcc --source-opt O2
   ```

4. **Test model**:
   ```bash
   python test_model.py --source-arch x86_64 --source-compiler gcc --source-opt O2
   ```

5. **Analyze results**:
   ```bash
   python analyze_results.py --source-arch x86_64 --source-compiler gcc --source-opt O2
   ```

### Customizing the Experiment

Edit `config.py` to change:
- Projects to compile
- Target architectures
- Compilers to use
- Optimization levels
- Model hyperparameters

You can also pass command-line arguments to individual scripts. For example:

```bash
python train_model.py --source-arch x86_64 --source-compiler clang --source-opt O3 --epochs 20 --batch-size 128
```

## Expected Outputs

The experiment produces these key outputs in the `results/` directory:

1. **optimization_matrix.png**: Heatmap showing how different optimization levels affect model performance across architectures
2. **architecture_transfer.png**: Bar chart showing how well the model transfers from x86_64 to other architectures
3. **compiler_impact.png**: Bar chart showing how different compilers affect model performance
4. **optimization_trends.png**: Line chart showing performance trends across optimization levels
5. **relative_performance.png**: Bar chart showing relative performance compared to the source configuration
6. **results_summary.xlsx**: Excel file with detailed metrics
7. **results_summary.csv**: CSV file with detailed metrics

## Result Interpretation

The visualizations help identify several key insights:

1. **Optimization Level Impact**: Compare how different optimization levels affect cross-architecture transferability. Is O0 (no optimization) better for transfer learning than highly optimized O3 code?

2. **Architecture Transferability**: See which architectures are most similar to the source architecture in terms of learned embeddings. 

3. **Compiler Impact**: Compare how different compilers affect the model's ability to transfer knowledge.

## Troubleshooting

### Compilation Issues

If you encounter compilation errors:

1. Check that all required cross-compilers are installed
2. Verify the build environment has all necessary dependencies
3. Check the logs in the `logs/` directory for specific error messages
4. Try compiling with a single project first to isolate issues

### Function Extraction Issues

If function extraction produces too few results:

1. Check that the appropriate disassemblers are installed
2. Try adjusting the extraction parameters in `config.py` (e.g., `MIN_FUNCTION_SIZE` and `MAX_FUNCTION_SIZE`)
3. Verify that the binaries were compiled correctly

### Model Training Issues

If model training fails or produces poor results:

1. Try increasing the number of epochs
2. Adjust the learning rate or batch size
3. Check that enough functions were extracted to create a sufficient training dataset

## Citation

If you use this code in academic work, please cite:

```
@misc{compiler_optimization_analysis,
  author = {Your Name},
  title = {Compiler Optimization Impact Analysis on Cross-Architecture Binary Similarity},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/compiler-optimization-analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.