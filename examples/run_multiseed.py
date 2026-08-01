"""
Multi-seed experiment runner.

Trains a BioTransformer for each requested (task, attention) combination
across multiple random seeds, evaluates on all relevant test sets, then
reports mean ± std of the primary metric across seeds.

Usage
-----
From the repository root::

    python examples/run_multiseed.py \\
        --task secondary_structure fluorescence \\
        --attention plain2d blockwise2d blockwise3d homa \\
        --seeds 42 123 456 \\
        --ss3_data_dir  /path/to/tape/secondary_structure \\
        --fl_data_dir   /path/to/tape/fluorescence \\
        --stab_data_dir /path/to/tape/stability \\
        --checkpoint_dir checkpoints_multiseed \\
        --epochs 20

Primary metrics
---------------
- secondary_structure : per-residue accuracy  (best model chosen by val loss)
- fluorescence        : Spearman ρ            (best model chosen by val Spearman ρ)
- stability           : Spearman ρ            (best model chosen by val Spearman ρ)

Secondary structure test sets: cb513, casp12, ts115
Fluorescence / stability test sets: test
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer

from config import AttentionConfig, ModelConfig, TrainingConfig
from data.collate import collate_regression, collate_ss3
from data.datasets import (
    FluorescenceDataset,
    SecondaryStructureDataset,
    StabilityDataset,
)
from evaluation.metrics import accuracy_per_position, spearman_correlation
from tasks.fluorescence import FluorescenceTask
from tasks.secondary_structure import SecondaryStructureTask
from tasks.stability import StabilityTask
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# SS3 test-set names (TAPE provides three held-out splits)
# ---------------------------------------------------------------------------
SS3_TEST_SPLITS = ["cb513", "casp12", "ts115"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ss3(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Accuracy over non-padding positions on a secondary-structure test set."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, labels = model(input_ids, labels)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return accuracy_per_position(torch.cat(all_logits), torch.cat(all_labels))


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Spearman ρ on a regression test set (fluorescence or stability)."""
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        preds = model(input_ids)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    return spearman_correlation(torch.cat(all_preds), torch.cat(all_targets))


# ---------------------------------------------------------------------------
# Attention config factory
# ---------------------------------------------------------------------------

def make_attn_cfg(attn_type: str, args: argparse.Namespace) -> AttentionConfig:
    return AttentionConfig(
        type=attn_type,
        block_size=args.block_size,
        stride=args.stride,
        linformer_k=args.linformer_k,
        window_size=args.window_size,
        rank_3d=args.rank_3d,
        combine=getattr(args, "combine", "fusion"),
        gate_init=getattr(args, "gate_init", 1.0),
    )


# ---------------------------------------------------------------------------
# Per-task runners
# ---------------------------------------------------------------------------

def run_secondary_structure(
    attn_cfg: AttentionConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    tokenizer,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    """Train on SS3 and evaluate on all three test splits. Returns metric dict."""
    set_seed(seed)

    task = SecondaryStructureTask(model_cfg, attn_cfg, train_cfg)
    train_lmdb = LMDBDataset(os.path.join(args.ss3_data_dir, "train.lmdb"))
    val_lmdb   = LMDBDataset(os.path.join(args.ss3_data_dir, "valid.lmdb"))

    model, _ = task.train(
        train_lmdb=train_lmdb,
        val_lmdb=val_lmdb,
        tokenizer=tokenizer,
        attn_name_suffix=f"_seed{seed}",
    )

    device = next(model.parameters()).device
    results = {}
    for split in SS3_TEST_SPLITS:
        lmdb_path = os.path.join(args.ss3_data_dir, f"{split}.lmdb")
        if not os.path.exists(lmdb_path):
            print(f"  [warn] {lmdb_path} not found — skipping {split}")
            continue
        test_lmdb = LMDBDataset(lmdb_path)
        test_loader = DataLoader(
            SecondaryStructureDataset(test_lmdb, tokenizer),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_ss3,
            num_workers=train_cfg.num_workers,
        )
        results[split] = evaluate_ss3(model, test_loader, device)
        print(f"    [{split}] accuracy = {results[split]:.4f}")

    return results


def run_fluorescence(
    attn_cfg: AttentionConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    tokenizer,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    """Train on fluorescence and evaluate on the test split."""
    set_seed(seed)

    task = FluorescenceTask(model_cfg, attn_cfg, train_cfg)
    train_lmdb = LMDBDataset(os.path.join(args.fl_data_dir, "train.lmdb"))
    val_lmdb   = LMDBDataset(os.path.join(args.fl_data_dir, "valid.lmdb"))

    model, _ = task.train(
        train_lmdb=train_lmdb,
        val_lmdb=val_lmdb,
        tokenizer=tokenizer,
        attn_name_suffix=f"_seed{seed}",
    )

    device = next(model.parameters()).device
    test_lmdb = LMDBDataset(os.path.join(args.fl_data_dir, "test.lmdb"))
    test_loader = DataLoader(
        FluorescenceDataset(test_lmdb, tokenizer),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_regression,
        num_workers=train_cfg.num_workers,
    )
    rho = evaluate_regression(model, test_loader, device)
    print(f"    [test] Spearman ρ = {rho:.4f}")
    return {"test": rho}


def run_stability(
    attn_cfg: AttentionConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    tokenizer,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    """Train on stability and evaluate on the test split."""
    set_seed(seed)

    task = StabilityTask(model_cfg, attn_cfg, train_cfg)
    train_lmdb = LMDBDataset(os.path.join(args.stab_data_dir, "train.lmdb"))
    val_lmdb   = LMDBDataset(os.path.join(args.stab_data_dir, "valid.lmdb"))

    model, _ = task.train(
        train_lmdb=train_lmdb,
        val_lmdb=val_lmdb,
        tokenizer=tokenizer,
        attn_name_suffix=f"_seed{seed}",
    )

    device = next(model.parameters()).device
    test_lmdb = LMDBDataset(os.path.join(args.stab_data_dir, "test.lmdb"))
    test_loader = DataLoader(
        StabilityDataset(test_lmdb, tokenizer),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_regression,
        num_workers=train_cfg.num_workers,
    )
    rho = evaluate_regression(model, test_loader, device)
    print(f"    [test] Spearman ρ = {rho:.4f}")
    return {"test": rho}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(
    task: str,
    attn_type: str,
    seed_results: List[Dict[str, float]],
) -> None:
    """Print mean ± std across seeds for every test split."""
    # Collect all split names seen across seeds
    all_splits = sorted({s for r in seed_results for s in r})

    print(f"\n  {'─'*54}")
    print(f"  Task: {task}  |  Attention: {attn_type}  |  Seeds: {len(seed_results)}")
    print(f"  {'─'*54}")

    for split in all_splits:
        vals = [r[split] for r in seed_results if split in r]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=0))
        per_seed = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  [{split:>6}]  mean={mean:.4f}  std={std:.4f}   per-seed: {per_seed}")

    print(f"  {'─'*54}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed BioTransformer experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- What to run ---
    parser.add_argument(
        "--task", nargs="+",
        choices=["secondary_structure", "fluorescence", "stability"],
        required=True,
        help="Task(s) to run.",
    )
    parser.add_argument(
        "--attention", nargs="+",
        choices=["plain2d", "blockwise2d", "linformer2d", "homa", "blockwise3d"],
        required=True,
        help="Attention type(s) to evaluate.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Random seeds (default: 3 seeds).",
    )

    # --- Data paths ---
    parser.add_argument("--ss3_data_dir",  default=None,
                        help="Directory containing SS3 LMDB files.")
    parser.add_argument("--fl_data_dir",   default=None,
                        help="Directory containing fluorescence LMDB files.")
    parser.add_argument("--stab_data_dir", default=None,
                        help="Directory containing stability LMDB files.")

    # --- Checkpointing ---
    parser.add_argument("--checkpoint_dir", default="checkpoints_multiseed",
                        help="Root directory for checkpoints.")

    # --- Model hyperparameters ---
    parser.add_argument("--d_model",        type=int, default=512)
    parser.add_argument("--num_layers",     type=int, default=12)
    parser.add_argument("--num_heads",      type=int, default=8)
    parser.add_argument("--dim_feedforward",type=int, default=1024)
    parser.add_argument("--dropout",        type=float, default=0.4)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # --- Attention hyperparameters ---
    parser.add_argument("--block_size",   type=int,   default=30)
    parser.add_argument("--stride",       type=int,   default=15)
    parser.add_argument("--linformer_k",  type=int,   default=50)
    parser.add_argument("--window_size",  type=int,   default=7)
    parser.add_argument("--rank_3d",      type=int,   default=8)
    parser.add_argument("--combine", choices=("fusion", "add", "gated"), default="fusion",
                        help="homa: how the 2D and 3D branches are merged. "
                             "'fusion' = concat + MLP (published); 'add' = plain sum; "
                             "'gated' = sum with one learnable scalar.")
    parser.add_argument("--gate_init", type=float, default=1.0,
                        help="init of the gate scalar when --combine gated "
                             "(1.0 matches 'add'; 0.0 starts as the 2D branch alone)")

    # --- Training hyperparameters ---
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate that required data dirs are provided for requested tasks
    required = {
        "secondary_structure": ("ss3_data_dir",  "--ss3_data_dir"),
        "fluorescence":        ("fl_data_dir",   "--fl_data_dir"),
        "stability":           ("stab_data_dir", "--stab_data_dir"),
    }
    for task in args.task:
        attr, flag = required[task]
        if getattr(args, attr) is None:
            raise ValueError(f"Task '{task}' requires {flag} to be set.")

    tokenizer = TAPETokenizer(vocab="iupac")

    model_cfg = ModelConfig(
        vocab_size=30,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    )

    # Results store: results[task][attn_type] = list of per-seed dicts
    all_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        t: {a: [] for a in args.attention} for t in args.task
    }

    total_runs = len(args.task) * len(args.attention) * len(args.seeds)
    run_idx = 0

    for task in args.task:
        for attn_type in args.attention:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"  Run {run_idx}/{total_runs} | task={task}  attn={attn_type}  seed={seed}")
                print(f"{'='*60}")

                # Per-run checkpoint subdir to keep seeds isolated
                ckpt_dir = os.path.join(
                    args.checkpoint_dir, task, attn_type, f"seed{seed}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                train_cfg = TrainingConfig(
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    epochs=args.epochs,
                    checkpoint_dir=ckpt_dir,
                    num_workers=args.num_workers,
                )

                attn_cfg = make_attn_cfg(attn_type, args)

                if task == "secondary_structure":
                    seed_result = run_secondary_structure(
                        attn_cfg, model_cfg, train_cfg, tokenizer, args, seed
                    )
                elif task == "fluorescence":
                    seed_result = run_fluorescence(
                        attn_cfg, model_cfg, train_cfg, tokenizer, args, seed
                    )
                else:  # stability
                    seed_result = run_stability(
                        attn_cfg, model_cfg, train_cfg, tokenizer, args, seed
                    )

                all_results[task][attn_type].append(seed_result)

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'#'*60}")
    print("  MULTI-SEED RESULTS SUMMARY")
    print(f"{'#'*60}")

    for task in args.task:
        print(f"\n{'━'*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'━'*60}")
        for attn_type in args.attention:
            seed_results = all_results[task][attn_type]
            if not seed_results:
                continue
            report(task, attn_type, seed_results)

    print()


if __name__ == "__main__":
    main()
