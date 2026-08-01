"""
Centralized configuration for the TAPE BioTransformer package.

All hyperparameters live here. Pass these dataclass instances to model,
data, and training constructors rather than scattering magic numbers through
the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared across all tasks.

    Attributes:
        vocab_size: Number of tokens in the amino-acid vocabulary (TAPE IUPAC tokenizer).
        d_model: Embedding / hidden dimension throughout the transformer.
        num_layers: Number of stacked encoder layers.
        num_heads: Number of attention heads (must divide d_model evenly).
        dim_feedforward: Hidden width of the position-wise feed-forward network.
        dropout: Dropout probability applied after attention and FFN sub-layers.
        max_seq_length: Maximum padded sequence length fed to the model.
    """
    vocab_size: int = 30
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.4
    max_seq_length: Optional[int] = None


@dataclass
class AttentionConfig:
    """Configuration for the attention mechanism.

    The ``type`` field selects which attention class is instantiated via
    ``get_attention()``.  Only the parameters relevant to the chosen type
    need to be set; unused ones are silently ignored.

    Supported types
    ---------------
    ``"plain2d"``
        Standard multi-head scaled dot-product attention.
    ``"blockwise2d"``
        Sliding-window 2D attention (block_size, stride).
    ``"linformer2d"``
        Low-rank 2D attention (linformer_k, max_seq_length from ModelConfig).
    ``"homa"``  ← **main contribution**
        HOMA (Higher-Order MultiHead Attention) with low-rank U-matrix and optional
        pretrained-2D transfer (block_size, stride, window_size, rank_3d,
        pretrained_ckpt, freeze_2d).

    Attributes:
        type: One of the strings listed above.
        block_size: Block length for sliding-window attention variants.
        stride: Step size between consecutive blocks.
        linformer_k: Low-rank projection dimension for Linformer2D.
        window_size: Local context window for HOMA attention.
        rank_3d: Rank of the low-rank U-matrix decomposition in homa.
        pretrained_ckpt: Path to a checkpoint whose W_q/W_k/W_v weights are
            loaded into the 2D projections of homa.
        freeze_2d: If True, freeze the loaded 2D projection weights so only
            the 3D-specific parameters (W_u_u, W_u_v, and whichever
            combination parameters ``combine`` creates) are trained.
        combine: How homa merges its 2D and 3D branches — ``"fusion"``
            (concat + MLP, published), ``"add"`` (plain sum), or ``"gated"``
            (sum with one learnable scalar).  Ablation knob: the fusion MLP
            can rescale or discard a branch, so ``"add"`` / ``"gated"``
            measure the triadic contribution without that confound.
        gate_init: Initial value of the ``gated`` scalar.
    """
    type: str = "plain2d"

    # Sliding-window (blockwise2d / homa)
    block_size: int = 30
    stride: int = 15

    # Linformer
    linformer_k: int = 50

    # 3D sliding-window specific
    window_size: int = 7
    rank_3d: int = 8
    tie_u_to_k: bool = False  # homa: reuse K as the third (U) factor -> score = Q·K·K (no separate U)
    uniform_pool_3d: bool = False  # homa: ablate triadic attention -> uniform V⊙V pooling (no scores)

    # homa: how the 2D and 3D branch outputs are merged.  Both branches are
    # head_dim-wide, so all three are shape-compatible drop-ins:
    #   "fusion" — concat -> MLP(2*head_dim -> 128 -> head_dim)   [published]
    #   "add"    — attn_2d + res_3d                               [no parameters]
    #   "gated"  — attn_2d + gate * res_3d, gate a learnable scalar
    combine: str = "fusion"
    gate_init: float = 1.0  # init of `gate` when combine="gated" (1.0 == "add";
                            # 0.0 starts identical to the 2D branch alone)

    # Transfer-learning for homa
    pretrained_ckpt: Optional[str] = None
    freeze_2d: bool = False


@dataclass
class TrainingConfig:
    """Optimisation and bookkeeping settings.

    Attributes:
        batch_size: Mini-batch size for the data loaders.
        learning_rate: Initial Adam learning rate (peak LR after warmup).
        epochs: Total number of training epochs.
        warmup_ratio: Fraction of total training steps used for linear LR
            warmup (e.g. 0.06 = 6 %).  0.0 disables warmup.
        lr_scheduler: LR schedule applied after warmup.  One of
            ``"cosine"`` (cosine decay to 0), ``"linear"`` (linear decay
            to 0), or ``"none"`` (constant LR after warmup).
        grad_clip: Maximum global gradient norm for clipping.  0.0
            disables clipping.  A value of 1.0 is a safe default.
        warmup_steps: Steps excluded from efficiency timing at the start
            of each epoch (avoids measuring JIT-compilation overhead).
        checkpoint_dir: Directory where ``*.pt`` checkpoints are saved.
        num_workers: DataLoader worker processes.
        device: ``"cuda"`` or ``"cpu"`` (auto-detected if not set).
    """
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 20
    warmup_ratio: float = 0.0
    lr_scheduler: str = "none"
    grad_clip: float = 0.0
    warmup_steps: int = 5
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0
    device: Optional[str] = None
    u_entropy_lambda: float = 0.0  # weight of the HOMA U-axis entropy penalty (0.0 = off)
