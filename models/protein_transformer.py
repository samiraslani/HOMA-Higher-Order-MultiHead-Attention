"""
Unified ProteinTransformer backbone with swappable task heads.

Two task head classes are provided:

* ``PerResidueHead``      — per-position classification (e.g. SS3 prediction).
* ``GlobalRegressionHead`` — sequence-level regression via mean pooling
                             (e.g. fluorescence / stability prediction).

Typical usage::

    from tape_biotransformer.config import ModelConfig, AttentionConfig
    from tape_biotransformer.models.protein_transformer import (
        ProteinTransformer, PerResidueHead, GlobalRegressionHead
    )

    model_cfg  = ModelConfig()
    attn_cfg   = AttentionConfig(type="homa", block_size=40, stride=15)

    # Secondary structure
    head = PerResidueHead(d_model=model_cfg.d_model, num_classes=3)
    model = ProteinTransformer(model_cfg, attn_cfg, head)

    # Fluorescence / stability
    head = GlobalRegressionHead(d_model=model_cfg.d_model, d_ff=128)
    model = ProteinTransformer(model_cfg, attn_cfg, head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from config import AttentionConfig, ModelConfig
from models.encoder import Encoder, _SLIDING_ATTENTION_TYPES

# ---------------------------------------------------------------------------
# Padding utility (Minimum Safe Padding Lemma)
# ---------------------------------------------------------------------------

def min_safe_length(L_real: int, block_size: int, stride: int) -> int:
    """Return the minimum effective padded length guaranteeing every real
    position appears in exactly ``block_size // stride`` overlapping blocks.

    From the Minimum Safe Padding Lemma:

        L_eff >= stride * floor((L_real - 1) / stride) + block_size

    Block k starts at k*stride and covers positions [k*stride, k*stride + block_size - 1].
    The last real position (L_real - 1) needs block k* = floor((L_real-1)/stride)
    to exist, which requires L_eff >= k* * stride + block_size.
    """
    return stride * ((L_real - 1) // stride) + block_size


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class PerResidueHead(nn.Module):
    """Linear classifier applied independently to each sequence position.

    Used for per-residue prediction tasks such as secondary structure (SS3).

    Args:
        d_model: Encoder output dimension.
        num_classes: Number of output classes (e.g. 3 for H/E/C).
    """

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, L, num_classes)``
        """
        return self.classifier(x)


class GlobalRegressionHead(nn.Module):
    """Flat classifier for sequence-level regression.

    Flattens the full encoder output and passes it through a two-layer MLP.
    A LayerNorm is applied before the first linear layer to stabilise
    initialisation and eliminate seed-dependent collapse of the large linear.

    Architecture::

        Flatten  →  LayerNorm(len_seq * d_model)
                 →  Linear(len_seq * d_model, d_ff)  →  ReLU  →  Dropout
                 →  Linear(d_ff, 1)

    Args:
        d_model: Encoder output dimension.
        len_seq: Padded sequence length seen by the encoder.
        d_ff: Hidden dimension of the regression MLP.
        dropout: Dropout probability applied inside the MLP.
    """

    def __init__(self, d_model: int, len_seq: int, d_ff: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len_seq * d_model),
            nn.Linear(len_seq * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, 1)``
        """
        return self.regressor(x)


# ---------------------------------------------------------------------------
# Unified backbone
# ---------------------------------------------------------------------------

class ProteinTransformer(nn.Module):
    """Shared transformer backbone with a pluggable task head.

    The backbone consists of:
      - Token embedding:     (B, L) → (B, L, d_model)
      - Positional embedding (learnable): (B, L, d_model)
      - Embedding layer norm
      - ``num_layers`` encoder layers (attention + FFN + residual norms)

    The output of the encoder stack is passed to the task ``head`` module.

    Mask generation
    ---------------
    Padding masks are generated internally from ``input_ids`` (padding token
    index = 0).  The mask shape is adjusted based on the attention type:

    - ``plain2d``    → ``(B, 1, 1, L)``
    - ``linformer2d`` → ``(B, 1, L, 1)``
    - ``blockwise2d`` / ``homa`` / ``blockwise3d`` → ``(B, Blk, L_b)``

    Sequence alignment for sliding-window types
    --------------------------------------------
    When a sliding-window attention is used, ``input_ids`` are zero-padded to
    the nearest length satisfying ``(L - block_size) % stride == 0`` before
    any embedding lookup.  If ``max_seq_length`` is set, sequences longer than
    that limit are truncated first. For the SS3 task the corresponding
    ``labels`` tensor is also padded with ``-100`` (the ``ignore_index`` in
    ``CrossEntropyLoss``) or truncated to match.

    Args:
        model_cfg: Architecture hyperparameters.
        attn_cfg: Attention type and its parameters.
        head: Task head module (``PerResidueHead`` or ``GlobalRegressionHead``).
        load_ffn_pretrained: Load FFN weights from checkpoint as well.
        freeze_ffn: Freeze FFN weights after loading.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        attn_cfg: AttentionConfig,
        head: nn.Module,
        pretrained_2d_ckpt: Optional[str] = None,
        load_ffn_pretrained: bool = False,
        freeze_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.attn_cfg = attn_cfg
        self.head = head

        # --- resolve effective sequence length ---
        is_sliding = attn_cfg.type.lower() in _SLIDING_ATTENTION_TYPES
        self._is_sliding = is_sliding

        if model_cfg.max_seq_length is not None:
            # Fixed mode: user supplied an explicit maximum; use it directly.
            self._fixed_len_seq = model_cfg.max_seq_length
            len_seq = model_cfg.max_seq_length
            print(f"  Padding sequences to fixed length {len_seq} (user-specified max_seq_length).")
        else:
            # Dynamic mode: pad each batch to the minimum safe length derived
            # from the Minimum Safe Padding Lemma at forward time.
            self._fixed_len_seq = None
            # Allocate position embeddings up to a generous maximum so that
            # any protein sequence encountered at runtime is covered.
            len_seq = 4096

        self._last_reported_target = -1  # used to avoid duplicate print messages

        # --- resolve pretrained checkpoint (explicit arg overrides attn_cfg) ---
        resolved_ckpt = pretrained_2d_ckpt or attn_cfg.pretrained_ckpt

        # --- build common attn_kwargs dict ---
        attn_kwargs = dict(
            len_seq=len_seq,
            block_size=attn_cfg.block_size,
            stride=attn_cfg.stride,
            linformer_k=attn_cfg.linformer_k,
            window_size=attn_cfg.window_size,
            rank=attn_cfg.rank_3d,
            tie_u_to_k=attn_cfg.tie_u_to_k,
            uniform_pool_3d=attn_cfg.uniform_pool_3d,
            combine=attn_cfg.combine,
            gate_init=attn_cfg.gate_init,
            load_from_pretrained_2d=(resolved_ckpt is not None),
            pretrained_2d_ckpt=resolved_ckpt,
            freeze_2d=attn_cfg.freeze_2d,
        )

        # --- encoder stack ---
        self.encoder_layers = nn.ModuleList([
            Encoder(
                attn_type=attn_cfg.type,
                d_model=model_cfg.d_model,
                num_heads=model_cfg.num_heads,
                d_ff=model_cfg.dim_feedforward,
                dropout=model_cfg.dropout,
                load_ffn_pretrained=load_ffn_pretrained,
                freeze_ffn=freeze_ffn,
                **attn_kwargs,
            )
            for _ in range(model_cfg.num_layers)
        ])

        if resolved_ckpt is not None:
            print(f"  Transfer learning : blockwise 2D parameters (W_q, W_k, W_v) loaded from: {resolved_ckpt}")
        if attn_cfg.freeze_2d:
            combine_params = {"fusion": "fusion_layer", "gated": "gate", "add": ""}[
                attn_cfg.combine
            ]
            trainable = "W_u_u, W_u_v" + (f", {combine_params}" if combine_params else "")
            print(f"  Frozen layers     : W_q, W_k, W_v  (only {trainable} will be trained)")

        # --- embeddings ---
        self.token_embedding = nn.Embedding(
            model_cfg.vocab_size, model_cfg.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(len_seq, model_cfg.d_model)
        self.embedding_norm = nn.LayerNorm(model_cfg.d_model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_to_blocks(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Truncate or zero-pad ``input_ids`` (and ``labels``) to the target length.

        In fixed mode (``max_seq_length`` was set), pads every batch to that
        length.  In dynamic mode, computes the minimum safe length for the
        current batch via the Minimum Safe Padding Lemma:

            L_eff >= stride * floor((L_real - 1) / stride) + block_size
        """
        B, L = input_ids.shape

        if self._fixed_len_seq is not None:
            target = self._fixed_len_seq
            if L > target:
                input_ids = input_ids[:, :target]
                if labels is not None:
                    labels = labels[:, :target]
                L = target
        else:
            target = min_safe_length(L, self.attn_cfg.block_size, self.attn_cfg.stride)

        if target != self._last_reported_target:
            print(f"  Padding sequences to length {target} "
                  f"(L_real={L}, block_size={self.attn_cfg.block_size}, stride={self.attn_cfg.stride}).")
            self._last_reported_target = target

        pad_len = target - L
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            if labels is not None:
                labels = F.pad(labels, (0, pad_len), value=-100)
        return input_ids, labels

    def _generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a padding mask suitable for the configured attention type.

        Args:
            input_ids: ``(B, L)`` — already padded if sliding.

        Returns:
            Mask tensor.  Shape and dtype depend on attention type.
        """
        attn_type = self.attn_cfg.type.lower()
        mask = (input_ids != 0)  # (B, L), bool

        if attn_type == "plain2d":
            return mask.unsqueeze(1).unsqueeze(1)                  # (B, 1, 1, L)

        if attn_type == "linformer2d":
            return mask.unsqueeze(1).unsqueeze(3)                  # (B, 1, L, 1)

        if attn_type == "blockwise2d":
            block_size = self.attn_cfg.block_size
            stride = self.attn_cfg.stride
            B, L = input_ids.shape
            num_blocks = (L - block_size) // stride + 1
            shape = (B, num_blocks, block_size)
            strides_ = (
                mask.stride(0),
                stride * mask.stride(1),
                mask.stride(1),
            )
            block_mask = mask.as_strided(shape, strides_).contiguous()  # (B, Blk, L_b)
            return block_mask.unsqueeze(2).unsqueeze(4)                  # (B, Blk, 1, L_b, 1)

        if attn_type == "homa":
            block_size = self.attn_cfg.block_size
            stride = self.attn_cfg.stride
            B, L = input_ids.shape
            num_blocks = (L - block_size) // stride + 1
            shape = (B, num_blocks, block_size)
            strides_ = (
                mask.stride(0),
                stride * mask.stride(1),
                mask.stride(1),
            )
            return mask.as_strided(shape, strides_).contiguous()         # (B, Blk, L_b)

        if attn_type == "blockwise3d":
            block_size = self.attn_cfg.block_size
            stride = self.attn_cfg.stride
            B, L = input_ids.shape
            num_blocks = (L - block_size) // stride + 1
            shape = (B, num_blocks, block_size)
            strides_ = (
                mask.stride(0),
                stride * mask.stride(1),
                mask.stride(1),
            )
            return mask.as_strided(shape, strides_).contiguous()         # (B, Blk, L_b)

        # Fallback: standard 2D mask
        return mask.unsqueeze(1).unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the full forward pass.

        Args:
            input_ids: ``(B, L)`` token index tensor.
            labels: Optional target tensor.  For SS3 this is ``(B, L)``; for
                regression tasks it is ``(B,)`` or ``(B, 1)``.  When provided,
                the method returns ``(logits, labels)`` so that padded labels
                are updated to match any sequence-length padding applied
                internally.

        Returns:
            * If ``labels`` is ``None``: output tensor from the head.
            * If ``labels`` is provided: ``(head_output, updated_labels)``
              tuple so the training loop can use the up-to-date labels.
        """
        # Pad/truncate sequences: sliding types use the lemma after optional
        # truncation to max_seq_length; fixed max_seq_length non-sliding
        # types pad or truncate directly to the declared fixed length.
        if self._is_sliding:
            input_ids, labels = self._pad_to_blocks(input_ids, labels)
        elif self._fixed_len_seq is not None:
            if input_ids.shape[1] > self._fixed_len_seq:
                input_ids = input_ids[:, :self._fixed_len_seq]
                if labels is not None:
                    labels = labels[:, :self._fixed_len_seq]
            pad_len = self._fixed_len_seq - input_ids.shape[1]
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=0)
                if labels is not None:
                    labels = F.pad(labels, (0, pad_len), value=-100)

        B, L = input_ids.shape

        # Padding mask
        masks = self._generate_mask(input_ids)

        # Positional ids
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Embed tokens + positions
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.embedding_norm(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=masks)

        # Task head
        out = self.head(x)

        if labels is not None:
            return out, labels
        return out

