"""
Training script for Infrastructure Impact GNN.

Implements:
- Synthetic data generation (1,000 graphs by default)
- BFS cascade labels with edge-quality modulation
- Weighted BCE training with gradient clipping
- 5-fold cross validation and best-model selection
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import ImpactPredictor


SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
CV_DIR = MODELS_DIR / "cv"
DEFAULT_MODEL_PATH = MODELS_DIR / "gnn_model.pt"

NUM_OUTPUTS = 12
NUM_FEATURES = 24
TYPE_DIMS = 12
CRITICAL_TYPES = {2, 3, 9, 10}  # Power, Tank, School, Hospital


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_node_features(num_nodes: int, rng: np.random.Generator):
    """Generate 24-dim node features: [12 one-hot type | 12 operational]."""
    type_indices = rng.integers(0, TYPE_DIMS, size=num_nodes)
    type_one_hot = np.eye(TYPE_DIMS, dtype=np.float32)[type_indices]

    # Operational features [capacity, level, flow, status, criticality, population,
    # economic, connectivity, maintenance, weather, failure_history, reserved]
    operational = rng.uniform(0.0, 1.0, size=(num_nodes, 12)).astype(np.float32)

    # Basic interdependence constraints for realism
    status = operational[:, 3]
    operational[:, 2] = operational[:, 2] * status  # flow drops when unhealthy
    operational[:, 1] = np.minimum(operational[:, 1], operational[:, 0])  # level <= capacity

    x = np.concatenate([type_one_hot, operational], axis=1).astype(np.float32)
    return x, type_indices


def _generate_sparse_graph(num_nodes: int, rng: np.random.Generator):
    """Generate sparse random directed/partially-undirected topology."""
    edge_map = {}

    for src in range(num_nodes):
        degree_target = int(rng.integers(3, 6))  # 3-5 avg neighbors
        candidates = [idx for idx in range(num_nodes) if idx != src]
        if not candidates:
            continue

        chosen = rng.choice(candidates, size=min(degree_target, len(candidates)), replace=False)
        for dst in chosen:
            w_forward = float(rng.uniform(0.5, 1.0))
            edge_map[(src, int(dst))] = max(edge_map.get((src, int(dst)), 0.0), w_forward)

            # Optionally add reverse edge to mimic mixed directed/undirected systems.
            if rng.random() < 0.5:
                w_reverse = float(rng.uniform(0.5, 1.0))
                edge_map[(int(dst), src)] = max(edge_map.get((int(dst), src), 0.0), w_reverse)

    if not edge_map and num_nodes >= 2:
        edge_map[(0, 1)] = 0.8
        edge_map[(1, 0)] = 0.8

    edge_list = list(edge_map.keys())
    edge_weights = [edge_map[e] for e in edge_list]

    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_weight = np.array(edge_weights, dtype=np.float32)
    return edge_index, edge_weight


def _build_adjacency(edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int):
    adjacency = {node: [] for node in range(num_nodes)}
    for edge_id in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        w = float(edge_weight[edge_id])
        adjacency[src].append((dst, w))
    return adjacency


def generate_training_data(
    num_samples: int = 1000,
    num_nodes_range: tuple[int, int] = (10, 30),
    seed: int = 42,
):
    """
    Generate synthetic graph samples with 24-dim node features and 12-dim labels.
    """
    print(f"Generating synthetic training data: {num_samples} graphs")
    rng = np.random.default_rng(seed)
    data_list = []

    for sample_id in range(num_samples):
        num_nodes = int(rng.integers(num_nodes_range[0], num_nodes_range[1] + 1))

        x, node_types = _generate_node_features(num_nodes, rng)
        edge_index_np, edge_weight_np = _generate_sparse_graph(num_nodes, rng)
        adjacency = _build_adjacency(edge_index_np, edge_weight_np, num_nodes)

        # Ground truth cascade labels (num_nodes x 12)
        y = np.zeros((num_nodes, NUM_OUTPUTS), dtype=np.float32)

        failure_node = int(rng.integers(0, num_nodes))
        if int(node_types[failure_node]) in CRITICAL_TYPES:
            y[failure_node] = rng.uniform(0.7, 0.95, size=NUM_OUTPUTS)
        else:
            y[failure_node] = rng.uniform(0.5, 0.85, size=NUM_OUTPUTS)

        # BFS propagation with depth-limited decay and edge quality modulation.
        visited = {failure_node}
        current_layer = [failure_node]
        decay_factor = 0.7

        for depth in range(3):
            next_layer = []
            for node in current_layer:
                for neighbor, edge_quality in adjacency.get(node, []):
                    if neighbor in visited:
                        continue

                    quality_factor = 0.5 + 0.5 * edge_quality
                    propagation_factor = (decay_factor ** depth) * quality_factor
                    randomness = rng.uniform(0.5, 1.0, size=NUM_OUTPUTS)
                    y[neighbor] = y[node] * propagation_factor * randomness

                    visited.add(neighbor)
                    next_layer.append(neighbor)

            current_layer = next_layer
            decay_factor *= 0.7
            if not current_layer:
                break

        y += rng.normal(0.0, 0.05, size=y.shape).astype(np.float32)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        # Per-graph z-score normalization and clipping.
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True) + 1e-6
        x = (x - x_mean) / x_std
        x = np.clip(x, -3.0, 3.0).astype(np.float32)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight_np, dtype=torch.float32).unsqueeze(-1),
            y=torch.tensor(y, dtype=torch.float32),
        )
        data_list.append(data)

        if (sample_id + 1) % 100 == 0:
            print(f"  Generated {sample_id + 1}/{num_samples} samples")

    print(f"Done: generated {num_samples} graphs.")
    return data_list


def _run_train_epoch(predictor: ImpactPredictor, loader: DataLoader) -> float:
    losses = []
    for batch in loader:
        losses.append(predictor.train_step(batch))
    return float(np.mean(losses)) if losses else float("inf")


def _run_val_epoch(predictor: ImpactPredictor, loader: DataLoader) -> float:
    losses = []
    for batch in loader:
        losses.append(predictor.validation_step(batch))
    return float(np.mean(losses)) if losses else float("inf")


def train_model_with_five_fold_cv(
    num_epochs: int = 50,
    batch_size: int = 32,
    num_samples: int = 1000,
    save_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 42,
):
    """Train with 5-fold cross validation and keep the globally best fold model."""
    set_seed(seed)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    CV_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("Infrastructure Impact GNN Training (5-Fold Cross Validation)")
    print("=" * 72)
    print(f"Epochs={num_epochs}, Batch Size={batch_size}, Samples={num_samples}")

    dataset = generate_training_data(num_samples=num_samples, seed=seed)

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_global_val = float("inf")
    best_global_path = None
    fold_summaries = []
    all_indices = np.arange(len(dataset))

    for fold_id, (train_idx, val_idx) in enumerate(kfold.split(all_indices), start=1):
        print("\n" + "-" * 72)
        print(f"Fold {fold_id}/5 | train={len(train_idx)} val={len(val_idx)}")
        print("-" * 72)

        train_split = [dataset[i] for i in train_idx]
        val_split = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)

        predictor = ImpactPredictor(learning_rate=0.001, dropout=0.3)
        print(f"Device: {predictor.device}")
        print(f"Model params: {sum(p.numel() for p in predictor.model.parameters()):,}")

        best_fold_val = float("inf")
        best_fold_epoch = 0
        fold_model_path = CV_DIR / f"gnn_model_fold{fold_id}.pt"

        for epoch in range(num_epochs):
            train_loss = _run_train_epoch(predictor, train_loader)
            val_loss = _run_val_epoch(predictor, val_loader)
            predictor.scheduler.step(val_loss)

            current_lr = predictor.optimizer.param_groups[0]["lr"]
            print(
                f"Fold {fold_id} | Epoch {epoch + 1:02d}/{num_epochs} "
                f"| Train={train_loss:.4f} | Val={val_loss:.4f} | LR={current_lr:.6f}"
            )

            if val_loss < best_fold_val:
                best_fold_val = val_loss
                best_fold_epoch = epoch + 1
                predictor.save_model(
                    str(fold_model_path),
                    metadata={
                        "fold": fold_id,
                        "epoch": best_fold_epoch,
                        "best_val_loss": best_fold_val,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "num_samples": num_samples,
                        "seed": seed,
                    },
                )

        fold_summaries.append(
            {
                "fold": fold_id,
                "best_val_loss": best_fold_val,
                "best_epoch": best_fold_epoch,
                "checkpoint": str(fold_model_path),
            }
        )

        print(
            f"Fold {fold_id} complete | best val loss={best_fold_val:.4f} "
            f"at epoch {best_fold_epoch}"
        )

        if best_fold_val < best_global_val:
            best_global_val = best_fold_val
            best_global_path = fold_model_path

    if best_global_path is None:
        raise RuntimeError("Cross-validation did not produce any checkpoint.")

    shutil.copyfile(best_global_path, save_path)

    report = {
        "best_global_val_loss": best_global_val,
        "best_global_checkpoint": str(best_global_path),
        "exported_model_path": str(save_path),
        "folds": fold_summaries,
        "trained_at": int(time.time()),
    }

    report_path = CV_DIR / "cv_results.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("Cross-validation complete")
    print(f"Best val loss: {best_global_val:.4f}")
    print(f"Best fold checkpoint: {best_global_path}")
    print(f"Exported model: {save_path}")
    print(f"CV report: {report_path}")
    print("=" * 72 + "\n")

    best_predictor = ImpactPredictor(model_path=str(save_path))
    return best_predictor, report


def train_model(
    num_epochs: int = 50,
    batch_size: int = 32,
    save_path: str | Path = DEFAULT_MODEL_PATH,
    num_samples: int = 1000,
    seed: int = 42,
):
    """Backward-compatible entry point; now uses 5-fold cross validation."""
    predictor, _ = train_model_with_five_fold_cv(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_samples=num_samples,
        save_path=save_path,
        seed=seed,
    )
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train Infrastructure GNN with 5-fold CV")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic graphs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for best model checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictor, report = train_model_with_five_fold_cv(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        save_path=args.save_path,
        seed=args.seed,
    )

    print("Running quick inference sanity check on a synthetic graph...")
    rng = np.random.default_rng(args.seed)
    num_nodes = 12
    x, _ = _generate_node_features(num_nodes, rng)
    edge_index, edge_weight = _generate_sparse_graph(num_nodes, rng)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x_norm = np.clip((x - x_mean) / x_std, -3.0, 3.0)

    predictions = predictor.predict(x_norm.astype(np.float32), edge_index, edge_weight)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Best CV validation loss: {report['best_global_val_loss']:.4f}")
    print("Model is ready for Gradio inference.")
