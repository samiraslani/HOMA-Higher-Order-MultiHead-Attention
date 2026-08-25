"""Synthetic interaction-order tasks: PARITY-k and MAJORITY-k.

These are the controlled tasks used to measure what interaction order an
attention mechanism can represent, independently of any protein dataset.

    PARITY-k    y_i = (sum of k bits at fixed offsets around i) mod 2
                irreducible order k: any strict subset of the k bits carries
                *no* information about the label.
    MAJORITY-k  y_i = 1 if that same sum exceeds k/2
                order 1: a threshold on a sum, so it is separable by a pairwise
                mechanism at every k.

Running both is what makes a result interpretable.  MAJORITY-k is solvable
pairwise at every k, so any effect that shows up on majority as well as parity
is not about interaction order.

The model is deliberately impoverished: embedding, attention layers, linear
readout, and **no feed-forward sublayer**.  An FFN can compute parity on its own
and would mask the attention's contribution; with a linear readout, a mechanism
that can only build degree-2 features cannot separate parity-3 in a single
layer.  That is what makes the order comparison a statement about attention.

Example
-------
    from tasks.synthetic_order import run_one, DEFAULT_CFG

    rec = run_one("homa", family="parity", k=4, d_model=32, seed=0,
                  n_layers=1, cfg=DEFAULT_CFG, device="cuda")
    print(rec["final"])
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.attention import attention_2d as a2
from models.attention import attention_3d as a3

__all__ = ["DEFAULT_CFG", "make_offsets", "centers_for", "make_data",
           "chance_level", "TinyModel", "build_model", "run_one",
           "n_params", "set_seed", "epochs_to"]

DEFAULT_CFG = dict(
    seq_len=16,      # sequence length
    reach=3,         # max |offset|; held constant as k varies
    heads=4,
    rank=8,          # low-rank U in the triadic mechanisms
    stride=8,
    train_n=3000,
    test_n=800,
    epochs=40,
    batch_size=128,
    lr=2e-3,
)

def make_offsets(k: int, reach: int) -> tuple[int, ...]:
    """``k`` offsets spread evenly over the fixed interval [-reach, +reach].

    Both the reach (max |offset|) and the span (max - min) are held constant as
    k grows, so the order sweep varies only k.  Constant reach also means one
    centred triadic window of size 2*reach+1 covers every k, keeping triadic
    compute constant across the sweep.

    Odd k includes the query position 0; even k straddles it.  At reach=3 this
    gives (-3, 0, 3) for k=3, matching the original XOR experiment exactly.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return (0,)
    raw = np.linspace(-reach, reach, k)
    offs = tuple(sorted({int(round(float(x))) for x in raw}))
    if len(offs) < k:
        raise SystemExit(
            f"reach={reach} cannot hold {k} distinct integer offsets "
            f"(got {offs}). Raise --reach to at least {(k - 1 + 1) // 2}."
        )
    return offs


def centers_for(offsets: tuple[int, ...], L: int) -> list[int]:
    """Positions whose full offset pattern lies inside the sequence."""
    return [i for i in range(L) if all(0 <= i + o < L for o in offsets)]


def make_data(n: int, family: str, k: int, reach: int, L: int, seed: int,
              centers_override=None):
    """Random bits plus per-position labels; -100 marks ignored boundaries.

    ``centers_override`` pins the labelled positions to a fixed set.  Sweep C
    varies the reach, and the number of valid centres shrinks as the pattern
    widens (L - 2R for k=3), which would confound "wider interaction" with
    "fewer labelled positions per sequence".  Pinning the centres to the set
    valid at the largest reach in the sweep holds the label count constant.
    """
    offsets = make_offsets(k, reach)
    centers = centers_for(offsets, L)
    if centers_override is not None:
        bad = [i for i in centers_override if i not in set(centers)]
        if bad:
            raise ValueError(
                f"centers_override {bad} invalid for k={k}, reach={reach}, L={L}")
        centers = list(centers_override)
    if not centers:
        raise ValueError(f"No valid centres for k={k}, reach={reach}, L={L}; raise --seq-len.")

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, L))
    y = np.full((n, L), -100, dtype=np.int64)
    for i in centers:
        total = sum(bits[:, i + o] for o in offsets)
        if family == "parity":
            y[:, i] = (total % 2).astype(np.int64)
        elif family == "majority":
            y[:, i] = (total > k / 2).astype(np.int64)
        else:
            raise ValueError(f"unknown family {family!r}")
    return (
        torch.tensor(bits, dtype=torch.long),
        torch.tensor(y),
        offsets,
        centers,
    )


def chance_level(Y: torch.Tensor) -> float:
    """Accuracy of always predicting the most frequent class."""
    lab = Y[Y != -100]
    if lab.numel() == 0:
        return float("nan")
    p = lab.float().mean().item()
    return max(p, 1.0 - p)


class TinyModel(nn.Module):
    """Embedding(+positional) -> N attention modules -> Linear(d_model, 2).

    There is no feed-forward sublayer: an FFN could compute parity on its own
    and mask the attention module's contribution.  With a linear readout, a
    mechanism that can only build degree-2 features cannot separate parity-3.

    With ``n_layers == 1`` the attention is applied directly, which is the
    configuration used by Sweeps A and B.  With ``n_layers > 1`` the layers are
    stacked pre-norm with residual connections, so a windowed mechanism can
    compose across layers and reach beyond a single window.  Sweep C uses that
    to ask whether depth recovers an interaction the window cannot see in one
    layer.  Note the single-layer point of a depth sweep is the residual
    variant, so it is comparable within Sweep C but not identical to Sweep A.
    """

    def __init__(self, attns, d_model: int, L: int, residual: bool = False):
        super().__init__()
        self.tok = nn.Embedding(2, d_model)
        self.pos = nn.Embedding(L, d_model)
        self.attns = nn.ModuleList(attns)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in attns)
        self.residual = residual
        self.final_norm = nn.LayerNorm(d_model) if residual else None
        self.readout = nn.Linear(d_model, 2)
        self.register_buffer("pos_ids", torch.arange(L).unsqueeze(0))

    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for attn, norm in zip(self.attns, self.norms):
            if self.residual:
                h = h + attn(norm(h), None)
            else:
                h = attn(norm(h), None)
        if self.final_norm is not None:
            h = self.final_norm(h)
        return self.readout(h)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def epochs_to(curve, thresh: float):
    """1-indexed first epoch reaching ``thresh``; None if never reached."""
    for i, v in enumerate(curve):
        if v >= thresh:
            return i + 1
    return None


def build_model(mech: str, d_model: int, heads: int, L: int, window: int,
                rank: int, stride: int, n_layers: int = 1,
                residual: bool = True) -> TinyModel:
    """Stack ``n_layers`` copies of one mechanism.

    ``block_size = L`` gives a single full-sequence block, so blocking is not a
    confound at this sequence length and the comparison isolates the attention
    order.

    ``residual`` should be left True whenever depth is being varied: TinyModel
    only stacks pre-norm with residual connections when depth > 1, so a depth-1
    point built without them would differ in architecture as well as depth.
    """
    def one():
        if mech == "blockwise2d":
            return a2.Attn2DBlockwise(heads, d_model, stride=stride, block_size=L)
        if mech == "blockwise3d":
            return a3.MultiHeadAttn3D(heads, d_model, block_size=L, stride=stride,
                                      window_size=window, rank=rank)
        if mech == "homa":
            return a3.HOMA(heads, d_model, stride=stride, block_size=L,
                           window_size=window, rank=rank)
        if mech == "homa_add":
            # HOMA with the fusion MLP replaced by a plain sum of the two
            # branches.  This is the control that keeps the comparison honest:
            # the published fusion layer is Linear -> ReLU -> Linear, an
            # internal MLP that no other mechanism here has.  TinyModel omits
            # the feed-forward sublayer precisely so that attention is the only
            # nonlinear stage, so with "fusion" HOMA carries a nonlinearity its
            # baselines lack -- 3,208 parameters per layer at d_model=32, and
            # twelve of them in a twelve-layer stack.  Any advantage could then
            # be that MLP rather than the fusion of interaction orders.
            # "add" is parameter-matched to Blockwise-3D exactly.
            return a3.HOMA(heads, d_model, stride=stride, block_size=L,
                           window_size=window, rank=rank, combine="add")
        raise ValueError(f"unknown mechanism {mech!r}")

    return TinyModel([one() for _ in range(n_layers)], d_model, L,
                     residual=residual)


def run_one(mech: str, *, family: str, k: int, d_model: int, seed: int,
            n_layers: int = 1, cfg: Optional[dict] = None,
            device: str = "cpu", residual: bool = True) -> dict:
    """Train one (mechanism, task, capacity, depth, seed) and return its record.

    The triadic window is sized once as ``2 * reach + 1`` rather than per k, so
    triadic compute is constant across an order sweep and k is the only thing
    that varies.
    """
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    set_seed(seed)
    L, R = cfg["seq_len"], cfg["reach"]

    Xtr, Ytr, offsets, centers = make_data(cfg["train_n"], family, k, R, L, seed)
    Xte, Yte, _, _ = make_data(cfg["test_n"], family, k, R, L, seed + 999)
    Xtr, Ytr, Xte, Yte = (t.to(device) for t in (Xtr, Ytr, Xte, Yte))

    window = 2 * R + 1
    model = build_model(mech, d_model, cfg["heads"], L, window, cfg["rank"],
                        cfg["stride"], n_layers=n_layers,
                        residual=residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, curve = time.time(), []
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for s in range(0, Xtr.shape[0], cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            loss = lossf(model(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte).argmax(-1)
            m = Yte != -100
            curve.append((pred[m] == Yte[m]).float().mean().item())

    return {
        "curve": curve,
        "final": curve[-1],
        "best": max(curve),
        "params": n_params(model),
        "offsets": list(offsets),
        "span": max(offsets) - min(offsets),
        "window": window,
        "n_centers": len(centers),
        "n_layers": n_layers,
        "chance": chance_level(Yte),
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                        if device == "cuda" else None),
    }
