# Binary Code Similarity and Description Generation

This project implements state-of-the-art deep learning models for binary code similarity detection and automatically generating descriptions of binary functions.

## Features

- **Binary Code Similarity Detection**: 
  - Multiple architecture options (SAFE, Gemini, Hybrid, Transformer)
  - Support for both sequence-based and graph-based approaches
  - Handles cross-architecture binary similarity

- **Function Description Generation**:
  - Transformer-based encoder-decoder architecture
  - Generates human-readable descriptions of binary functions
  - Support for beam search decoding

- **Preprocessing Pipeline**:
  - Disassembly of binary functions
  - Control Flow Graph (CFG) extraction
  - Feature normalization and tokenization

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/username/binary-code-analysis.git
   cd binary-code-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Dependencies

- Python 3.8+
- PyTorch 1.8+
- NetworkX for graph processing
- NLTK for evaluation metrics
- scikit-learn for metrics calculation
- tqdm for progress bars

For a full list of dependencies, see `requirements.txt`.

## Usage

### Quick Start with Example Data

Run the example script to see how the system works:

```
python example_usage.py --demo full --data_dir ./sample_data --output_dir ./output
```

This will:
1. Create sample assembly functions
2. Train both similarity and description models on the sample data
3. Run inference to demonstrate the system's capabilities

### Training Similarity Models

```
python train_similarity.py \
  --train_data path/to/training/data \
  --val_data path/to/validation/data \
  --output_dir ./output/similarity \
  --model_type safe \
  --embedding_dim 256 \
  --hidden_dim 512 \
  --epochs 10
```

Model types:
- `safe`: Self-Attentive Function Embedding (sequence-based)
- `gemini`: Graph-based model using Structure2Vec
- `hybrid`: Combination of sequence and graph approaches
- `transformer`: Transformer-based encoder (similar to jTrans)

### Training Description Generation Models

```
python train_description.py \
  --train_data path/to/training/data \
  --val_data path/to/validation/data \
  --output_dir ./output/description \
  --encoder_type transformer \
  --embedding_dim 256 \
  --hidden_dim 512 \
  --epochs 10
```

Encoder types:
- `transformer`: Transformer-based encoder
- `safe`: RNN-based encoder with self-attention

### Running Similarity Detection

To compute similarity between two binary functions:

```
python inference_similarity.py \
  --model_path ./output/similarity/best_model.pt \
  --vocab_dir ./output/similarity \
  --func1_path path/to/function1.asm \
  --func2_path path/to/function2.asm
```

### Generating Descriptions

To generate a description for a binary function:

```
python inference_description.py \
  --model_path ./output/description/best_model.pt \
  --vocab_dir ./output/description \
  --func_path path/to/function.asm \
  --beam_size 3
```

### Full Analysis

To run both similarity and description generation on a directory of functions:

```
python run_analysis.py \
  --mode both \
  --similarity_model ./output/similarity/best_model.pt \
  --description_model ./output/description/best_model.pt \
  --vocab_dir ./output/similarity \
  --input_dir ./input/functions \
  --output_file ./output/results.json
```

## Data Format

### Similarity Data

The system expects training data in one of these formats:

1. Directory structure:
   ```
   training/
     similar_pairs/
       pair1/
         func1.asm
         func2.asm
       pair2/
         ...
     dissimilar_pairs/
       pair1/
         func1.asm
         func2.asm
       pair2/
         ...
   ```

2. Single JSON file:
   ```json
   [
     {"func1": "assembly code...", "func2": "assembly code...", "label": 1},
     {"func1": "assembly code...", "func2": "assembly code...", "label": 0},
     ...
   ]
   ```

### Description Data

The system expects description data in one of these formats:

1. Directory structure:
   ```
   training/
     samples/
       sample1/
         func.asm
         desc.txt
       sample2/
         ...
   ```

2. Single JSON file:
   ```json
   [
     {"code": "assembly code...", "description": "Description text..."},
     {"code": "assembly code...", "description": "Description text..."},
     ...
   ]
   ```

## Model Architecture

### Similarity Detection

The system implements several architectures for binary similarity detection:

1. **SAFE**: Self-Attentive Function Embedding using bidirectional LSTM with attention
2. **Gemini**: Graph Neural Network model using Structure2Vec on Control Flow Graphs
3. **Hybrid**: Combination of sequence and graph-based approaches
4. **Transformer**: Transformer-based encoder (similar to jTrans approach)

All models are trained using a Siamese network architecture with contrastive loss.

### Description Generation

The description generation model uses:

1. Either a Transformer encoder or SAFE encoder for the assembly code
2. A Transformer decoder for generating natural language descriptions
3. Beam search for improved generation quality

## References

This implementation is based on research from the following papers:

- Xu et al., "Gemini: Learning Binary Code Similarity with Graph Embedding" (NDSS 2017)
- Massarelli et al., "SAFE: Self-Attentive Function Embeddings for Binary Similarity" (DIMVA 2019)
- Li et al., "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" (ICML 2019)
- Quader et al., "AsmDocGen: Generating Functional NL Descriptions for Assembly Code" (ICSOFT 2024)
- Jiang et al., "Nova: Generative Language Models for Assembly Code..." (ArXiv 2023)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
