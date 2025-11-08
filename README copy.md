#  BioTransformer: A Modular Transformer Framework for Biological Sequence Modeling

**BioTransformer** is a flexible and extensible framework for building, training, and evaluating Transformer-based architectures on biological sequence data.  
It enables researchers to easily specify and experiment with **different attention mechanisms** and **model configurations**, facilitating systematic exploration of sequence modeling approaches in bioinformatics.

---

## Key Features

- **Configurable Transformer Architecture**  
  Define custom model depth, embedding dimensions, number of heads, feed-forward size, and activation functions.

- **Customizable Attention Mechanisms**  
  Choose from a wide collection of baseline and novel attention formulations, or plug in your own research variant via a unified interface.

- **Dataset-Agnostic Design**  
  Works with DNA, RNA, or protein sequences. Users can preprocess their own datasets and train models end-to-end.

- **Benchmarking and Comparison**  
  Evaluate multiple Transformer configurations within a consistent experimental framework to compare accuracy, loss, and efficiency metrics.

- **Research-Oriented Codebase**  
  Built for transparency and reproducibility — easily adaptable for papers, ablation studies, and method development.

---

## Implemented Attention Mechanisms

BioTransformer provides a rich suite of attention variants that can be specified directly in the configuration:

### Baseline Attentions
- **Scaled Dot-Product Attention** — the standard attention mechanism introduced by Vaswani et al. (2017).  
- **Linformer Attention** — efficient linear attention using low-rank projections.  
- **Block Attention** — divides sequences into fixed blocks to reduce quadratic complexity.

### 3D and Hybrid Attention Mechanisms
- **3D Novel Attention** — a higher-order attention mechanism modeling tri-token (3-way) interactions across sequence dimensions.  
- **3D Linformer Attention** — combines 3D attention with Linformer-style projection for efficiency.  
- **3D Block Attention** — integrates 3D token interactions within localized blocks.  
- **Multiped Linformer Attention** — a novel combination of Linformer and Block Attention for structured sparsity and efficiency.  
- **Combined 2D and 3D Attentions** — merges planar and volumetric attention dynamics for richer representation.  
- **Optimized Dot-Product with Overlapping Blocks & 3-Way Interaction Attention** —  
  the top-performing configuration combining standard dot-product attention with overlapping local blocks and selective 3D token interactions, achieving high performance while significantly reducing computational complexity.

---

## Example Configuration

config = {
    "model_name": "BioTransformer_v1",
    "embedding_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "attention_type": "3D_NovelAttention",
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "sequence_type": "protein",
    "epochs": 50
}


## License

This project is distributed under a strict proprietary license (see LICENSE file).

