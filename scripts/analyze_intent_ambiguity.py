#!/usr/bin/env python3
"""Analyze intent ambiguity with CLIP features and error buckets.

This script:
1. loads a CLIP-based Intentonomy checkpoint,
2. runs validation + test inference,
3. derives TP / FN / high-score FP samples per intent,
4. clusters same-intent positive samples in CLIP feature space,
5. visualizes selected intents from high / mid / tail frequency buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hydra
import matplotlib
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.metrics import get_best_f1_scores  # noqa: E402
from src.utils.visualization import get_class_names  # noqa: E402


matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass
class SplitArtifacts:
    probs: np.ndarray
    targets: np.ndarray
    features: np.ndarray
    image_ids: np.ndarray


@dataclass
class IntentSelection:
    bucket: str
    intent_idx: int
    intent_name: str
    positive_count: int
    tp_count: int
    fn_count: int
    fp_count: int
    best_k: int
    silhouette: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    default_hparams = project_root / "logs/train/runs/2026-03-07_04-33-17/tensorboard/Intentonomy-CLIP-ViT-L14-L24-SUIL/version_0/hparams.yaml"
    default_ckpt = project_root / "logs/train/runs/2026-03-07_04-33-17/checkpoints/epoch_042.ckpt"
    default_output = project_root / "output/ambiguity_analysis/suil_l14_l24"

    parser = argparse.ArgumentParser(description="Analyze intent ambiguity with CLIP features.")
    parser.add_argument("--hparams", type=Path, default=default_hparams)
    parser.add_argument("--ckpt", type=Path, default=default_ckpt)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per-bucket", type=int, default=2, help="Selected intents per frequency bucket.")
    parser.add_argument("--top-fp", type=int, default=24, help="High-score FP samples kept for detailed plots.")
    parser.add_argument("--max-k", type=int, default=4, help="Max k for KMeans on positive samples.")
    parser.add_argument("--min-positive-samples", type=int, default=8, help="Minimum positives required for cluster diagnostics.")
    parser.add_argument("--min-cluster-size", type=int, default=4, help="Minimum cluster size when evaluating k > 1.")
    parser.add_argument("--silhouette-threshold", type=float, default=0.15, help="Minimum silhouette to call a class multi-cluster.")
    parser.add_argument("--examples-per-cluster", type=int, default=12)
    parser.add_argument("--tsne-perplexity", type=float, default=20.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_model(hparams_path: Path, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, DictConfig]:
    run_cfg = OmegaConf.load(hparams_path)
    if "model" not in run_cfg:
        raise ValueError(f"Missing model config in {hparams_path}")

    model_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg.model, resolve=False))
    if "compile" in model_cfg:
        model_cfg.compile = False

    model = hydra.utils.instantiate(model_cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalized_state_dict = {
        key.replace("._orig_mod", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    normalized_state_dict = {
        key: value
        for key, value in normalized_state_dict.items()
        if not key.startswith("ema_model.")
    }
    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    ignorable_missing = {"semantic_weight"}
    ignorable_unexpected_prefixes = ("ema_model.",)
    filtered_missing = [key for key in missing if key not in ignorable_missing]
    filtered_unexpected = [
        key
        for key in unexpected
        if not any(key.startswith(prefix) for prefix in ignorable_unexpected_prefixes)
    ]
    if filtered_missing or filtered_unexpected:
        raise RuntimeError(
            f"State dict mismatch when loading {ckpt_path}\nmissing={missing}\nunexpected={unexpected}"
        )
    model.eval()
    model.to(device)
    return model, run_cfg


def load_data_module(batch_size: int, num_workers: int):
    project_root = Path(__file__).resolve().parents[1]
    data_cfg = OmegaConf.load(project_root / "configs/data/intentonomy.yaml")
    merged_cfg = OmegaConf.create(
        {
            "paths": {"root_dir": str(project_root)},
            "data": data_cfg,
        }
    )
    OmegaConf.resolve(merged_cfg)

    merged_cfg.data.batch_size = batch_size
    merged_cfg.data.num_workers = num_workers
    datamodule = hydra.utils.instantiate(merged_cfg.data)
    datamodule.setup("fit")
    datamodule.setup("test")
    return datamodule


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, dim=1)


def extract_clip_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    net = model.net
    if hasattr(net, "_extract_cls_and_patch_mean_from_layer"):
        layer_idx = getattr(model, "layer_idx", getattr(net, "layer_idx", None))
        if layer_idx is None:
            raise ValueError("Could not determine layer_idx for CLIP feature extraction.")
        return net._extract_cls_and_patch_mean_from_layer(images, int(layer_idx))
    if hasattr(model, "_extract_all_features"):
        all_features = model._extract_all_features(images)
        cls_token = all_features["cls_token"].squeeze(1)
        patch_mean = all_features["patch_tokens"].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=1)
    raise TypeError(f"Unsupported model type for feature extraction: {type(model).__name__}")


@torch.no_grad()
def collect_split_artifacts(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> SplitArtifacts:
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_features: List[np.ndarray] = []
    all_image_ids: List[np.ndarray] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        step_outputs = model.model_step(
            batch,
            use_ema_model=False,
            intent_loss_weight=1.0,
        )
        preds = step_outputs[1]
        targets = step_outputs[2]
        features = extract_clip_features(model, batch["image"])
        features = l2_normalize(features)

        all_probs.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_features.append(features.detach().cpu().numpy())
        image_ids = batch["image_id"]
        if torch.is_tensor(image_ids):
            image_ids_np = image_ids.detach().cpu().numpy()
        else:
            image_ids_np = np.asarray(image_ids)
        all_image_ids.append(image_ids_np)

    return SplitArtifacts(
        probs=np.concatenate(all_probs, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        features=np.concatenate(all_features, axis=0),
        image_ids=np.concatenate(all_image_ids, axis=0),
    )


def build_image_lookup(dataset) -> Dict[str, Path]:
    return {str(image_id): Path(image_path) for image_id, image_path in dataset.images}


def choose_threshold(model: torch.nn.Module, val_artifacts: SplitArtifacts) -> float:
    use_half = bool(getattr(model, "use_learned_thresholds_for_eval", False))
    if use_half:
        return 0.5
    f1_dict = get_best_f1_scores(val_artifacts.targets, val_artifacts.probs)
    return float(f1_dict["threshold"])


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs > threshold).astype(np.int32)


def evaluate_best_k(
    features: np.ndarray,
    max_k: int,
    min_cluster_size: int,
    silhouette_threshold: float,
    seed: int,
) -> tuple[int, float]:
    sample_count = features.shape[0]
    if sample_count < max(2 * min_cluster_size, 8):
        return 1, 0.0

    best_k = 1
    best_silhouette = 0.0
    upper_k = min(max_k, sample_count // min_cluster_size)
    for k in range(2, upper_k + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(features)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < min_cluster_size:
            continue
        silhouette = float(silhouette_score(features, labels, metric="cosine"))
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    if best_silhouette < silhouette_threshold:
        return 1, best_silhouette
    return best_k, best_silhouette


def assign_clusters(
    positive_features: np.ndarray,
    all_features: np.ndarray,
    best_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if best_k <= 1:
        centroids = positive_features.mean(axis=0, keepdims=True)
        positive_labels = np.zeros(positive_features.shape[0], dtype=np.int32)
    else:
        kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
        positive_labels = kmeans.fit_predict(positive_features)
        centroids = kmeans.cluster_centers_

    similarity = all_features @ centroids.T
    nearest = similarity.argmax(axis=1).astype(np.int32)
    return positive_labels, nearest


def project_2d(features: np.ndarray, seed: int, tsne_perplexity: float) -> np.ndarray:
    if features.shape[0] <= 2:
        padded = np.zeros((features.shape[0], 2), dtype=np.float32)
        padded[:, : min(2, features.shape[1])] = features[:, : min(2, features.shape[1])]
        return padded

    if features.shape[0] < 8:
        return PCA(n_components=2, random_state=seed).fit_transform(features)

    reduced_dim = min(32, features.shape[0] - 1, features.shape[1])
    reduced = PCA(n_components=reduced_dim, random_state=seed).fit_transform(features)
    perplexity = max(5.0, min(tsne_perplexity, float(features.shape[0] - 1) / 3.0))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
    ).fit_transform(reduced)
    return coords.astype(np.float32)


def save_scatter_plot(
    output_path: Path,
    intent_name: str,
    coords: np.ndarray,
    sample_types: Sequence[str],
    cluster_ids: Sequence[int],
    scores: Sequence[float],
) -> None:
    color_map = {"TP": "#1f77b4", "FN": "#d62728", "FP": "#ff7f0e"}
    marker_cycle = ["o", "s", "^", "D", "P", "X"]

    plt.figure(figsize=(8, 6))
    unique_types = ["TP", "FN", "FP"]
    unique_clusters = sorted(set(int(x) for x in cluster_ids))

    for cluster_id in unique_clusters:
        marker = marker_cycle[cluster_id % len(marker_cycle)]
        for sample_type in unique_types:
            mask = [
                idx
                for idx, (kind, cid) in enumerate(zip(sample_types, cluster_ids))
                if kind == sample_type and int(cid) == int(cluster_id)
            ]
            if not mask:
                continue
            x = coords[mask, 0]
            y = coords[mask, 1]
            label = f"{sample_type} / C{cluster_id}"
            plt.scatter(
                x,
                y,
                s=42,
                alpha=0.85,
                marker=marker,
                c=color_map[sample_type],
                edgecolors="white",
                linewidths=0.5,
                label=label,
            )

    if coords.shape[0] <= 24:
        for idx, (x_coord, y_coord) in enumerate(coords):
            plt.text(
                x_coord,
                y_coord,
                f"{scores[idx]:.2f}",
                fontsize=7,
                alpha=0.7,
            )

    plt.title(f"{intent_name}: TP / FN / high-score FP in CLIP feature space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def ranked_indices_by_similarity(features: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    similarity = features @ centroid
    return np.argsort(-similarity)


def get_default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        return ImageFont.load_default()


def render_tile(image_path: Path, text: str, size: int = 180) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 30), color=(255, 255, 255))
    x_offset = (size - image.width) // 2
    y_offset = (size - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))

    draw = ImageDraw.Draw(canvas)
    font = get_default_font()
    draw.rectangle([0, size, size, size + 30], fill=(245, 245, 245))
    draw.text((6, size + 7), text[:22], fill=(0, 0, 0), font=font)
    return canvas


def save_contact_sheet(
    output_path: Path,
    title: str,
    records: Sequence[dict],
    columns: int = 4,
) -> None:
    if not records:
        return

    tile_size = 180
    tiles = [
        render_tile(Path(record["image_path"]), record["caption"], size=tile_size)
        for record in records
    ]
    rows = math.ceil(len(tiles) / columns)
    header_h = 42
    canvas = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + 30) + header_h),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(canvas)
    font = get_default_font()
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x_offset = col * tile_size
        y_offset = header_h + row * (tile_size + 30)
        canvas.paste(tile, (x_offset, y_offset))

    canvas.save(output_path)


def select_bucket_intents(
    summary_rows: List[dict],
    class_names: Sequence[str],
    per_bucket: int,
) -> List[IntentSelection]:
    positive_rows = [row for row in summary_rows if row["positive_count"] > 0]
    positive_rows.sort(key=lambda row: row["positive_count"], reverse=True)
    if not positive_rows:
        return []

    bucket_names = ["high", "mid", "tail"]
    bucketed: Dict[str, List[dict]] = {name: [] for name in bucket_names}
    for idx, row in enumerate(positive_rows):
        bucket_id = min((idx * 3) // len(positive_rows), 2)
        bucketed[bucket_names[bucket_id]].append(row)

    selected: List[IntentSelection] = []
    for bucket in bucket_names:
        rows = bucketed[bucket]
        rows.sort(
            key=lambda row: (
                row["fn_count"] + row["fp_count"],
                row["best_k"],
                row["silhouette"],
                row["positive_count"],
            ),
            reverse=True,
        )
        for row in rows[:per_bucket]:
            selected.append(
                IntentSelection(
                    bucket=bucket,
                    intent_idx=row["intent_idx"],
                    intent_name=class_names[row["intent_idx"]],
                    positive_count=row["positive_count"],
                    tp_count=row["tp_count"],
                    fn_count=row["fn_count"],
                    fp_count=row["fp_count"],
                    best_k=row["best_k"],
                    silhouette=row["silhouette"],
                )
            )
    return selected


def describe_cluster_labels(
    positive_targets: np.ndarray,
    cluster_labels: np.ndarray,
    target_idx: int,
    class_names: Sequence[str],
    top_n: int = 3,
) -> Dict[int, str]:
    descriptions: Dict[int, str] = {}
    for cluster_id in sorted(set(int(x) for x in cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_targets = positive_targets[mask]
        if cluster_targets.shape[0] == 0:
            descriptions[cluster_id] = "-"
            continue
        co_occurrence = cluster_targets.sum(axis=0)
        co_occurrence[target_idx] = 0
        ranked = np.argsort(-co_occurrence)
        pairs = []
        for idx in ranked[:top_n]:
            if co_occurrence[idx] <= 0:
                continue
            ratio = float(co_occurrence[idx] / cluster_targets.shape[0])
            pairs.append(f"{class_names[idx]}({ratio:.2f})")
        descriptions[cluster_id] = ", ".join(pairs) if pairs else "-"
    return descriptions


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_detailed_intent_analysis(
    selection: IntentSelection,
    test_artifacts: SplitArtifacts,
    test_preds: np.ndarray,
    image_lookup: Dict[str, Path],
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    intent_idx = selection.intent_idx
    intent_name = selection.intent_name

    positive_mask = test_artifacts.targets[:, intent_idx] > 0.5
    negative_mask = ~positive_mask
    tp_mask = positive_mask & (test_preds[:, intent_idx] == 1)
    fn_mask = positive_mask & (test_preds[:, intent_idx] == 0)
    fp_mask = negative_mask & (test_preds[:, intent_idx] == 1)

    positive_indices = np.where(positive_mask)[0]
    tp_indices = np.where(tp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    fp_candidates = np.where(fp_mask)[0]

    if positive_indices.size == 0:
        return {"intent_name": intent_name, "note": "no positive samples"}

    fp_scores = test_artifacts.probs[fp_candidates, intent_idx]
    ranked_fp = fp_candidates[np.argsort(-fp_scores)]
    ranked_fp = ranked_fp[: args.top_fp]

    positive_features = test_artifacts.features[positive_indices]
    best_k, silhouette = evaluate_best_k(
        positive_features,
        max_k=args.max_k,
        min_cluster_size=args.min_cluster_size,
        silhouette_threshold=args.silhouette_threshold,
        seed=args.seed,
    )

    combined_indices = np.concatenate([positive_indices, ranked_fp], axis=0)
    combined_features = test_artifacts.features[combined_indices]
    positive_cluster_labels, combined_nearest = assign_clusters(
        positive_features=positive_features,
        all_features=combined_features,
        best_k=best_k,
        seed=args.seed,
    )
    positive_cluster_map = {
        int(sample_idx): int(cluster_id)
        for sample_idx, cluster_id in zip(positive_indices, positive_cluster_labels)
    }
    assigned_cluster_map = {
        int(sample_idx): int(cluster_id)
        for sample_idx, cluster_id in zip(combined_indices, combined_nearest)
    }

    coords = project_2d(combined_features, seed=args.seed, tsne_perplexity=args.tsne_perplexity)
    sample_types = [
        "TP" if int(sample_idx) in set(tp_indices.tolist()) else "FN"
        for sample_idx in positive_indices
    ] + ["FP"] * len(ranked_fp)
    cluster_ids = [positive_cluster_map[int(sample_idx)] for sample_idx in positive_indices] + [
        assigned_cluster_map[int(sample_idx)] for sample_idx in ranked_fp
    ]
    scores = test_artifacts.probs[combined_indices, intent_idx].tolist()

    scatter_path = output_dir / f"{selection.bucket}_{intent_name}_scatter.png"
    save_scatter_plot(
        output_path=scatter_path,
        intent_name=f"{selection.bucket.upper()} / {intent_name}",
        coords=coords,
        sample_types=sample_types,
        cluster_ids=cluster_ids,
        scores=scores,
    )

    positive_targets = test_artifacts.targets[positive_indices]
    cluster_descriptions = describe_cluster_labels(
        positive_targets=positive_targets,
        cluster_labels=positive_cluster_labels,
        target_idx=intent_idx,
        class_names=class_names,
    )

    cluster_rows: List[dict] = []
    for cluster_id in sorted(set(cluster_ids)):
        cluster_positive_indices = [
            int(sample_idx)
            for sample_idx in positive_indices
            if positive_cluster_map[int(sample_idx)] == int(cluster_id)
        ]
        cluster_tp = [idx for idx in cluster_positive_indices if idx in set(tp_indices.tolist())]
        cluster_fn = [idx for idx in cluster_positive_indices if idx in set(fn_indices.tolist())]
        cluster_fp = [int(sample_idx) for sample_idx in ranked_fp if assigned_cluster_map[int(sample_idx)] == int(cluster_id)]

        fn_rate = float(len(cluster_fn) / max(len(cluster_positive_indices), 1))
        fp_mean = (
            float(test_artifacts.probs[cluster_fp, intent_idx].mean()) if cluster_fp else 0.0
        )
        cluster_rows.append(
            {
                "bucket": selection.bucket,
                "intent_idx": intent_idx,
                "intent_name": intent_name,
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster_positive_indices),
                "tp_count": len(cluster_tp),
                "fn_count": len(cluster_fn),
                "fn_rate": round(fn_rate, 4),
                "assigned_fp_count": len(cluster_fp),
                "assigned_fp_mean_score": round(fp_mean, 4),
                "top_cooccurring_intents": cluster_descriptions[int(cluster_id)],
            }
        )

        centroid = positive_features[positive_cluster_labels == int(cluster_id)].mean(axis=0)
        per_type_cap = max(1, math.ceil(args.examples_per_cluster / 3))

        selected_records: List[dict] = []
        for sample_type, sample_indices in [("TP", cluster_tp), ("FN", cluster_fn), ("FP", cluster_fp)]:
            if not sample_indices:
                continue
            sample_features = test_artifacts.features[sample_indices]
            order = ranked_indices_by_similarity(sample_features, centroid)
            for order_idx in order[:per_type_cap]:
                sample_idx = int(sample_indices[int(order_idx)])
                image_id = str(test_artifacts.image_ids[sample_idx])
                image_path = image_lookup[image_id]
                score = float(test_artifacts.probs[sample_idx, intent_idx])
                selected_records.append(
                    {
                        "image_path": str(image_path),
                        "caption": f"{sample_type} {score:.2f}",
                    }
                )

        sheet_path = output_dir / f"{selection.bucket}_{intent_name}_cluster_{cluster_id}.png"
        sheet_title = (
            f"{selection.bucket.upper()} / {intent_name} / cluster {cluster_id} "
            f"(co-occur: {cluster_descriptions[int(cluster_id)]})"
        )
        save_contact_sheet(
            output_path=sheet_path,
            title=sheet_title,
            records=selected_records,
        )

    return {
        "bucket": selection.bucket,
        "intent_idx": intent_idx,
        "intent_name": intent_name,
        "positive_count": selection.positive_count,
        "tp_count": selection.tp_count,
        "fn_count": selection.fn_count,
        "fp_count": selection.fp_count,
        "best_k": int(best_k),
        "silhouette": round(float(silhouette), 4),
        "scatter_path": str(scatter_path),
        "cluster_rows": cluster_rows,
    }


def write_markdown_report(
    output_path: Path,
    ckpt_path: Path,
    threshold: float,
    selected: Sequence[IntentSelection],
    detailed_results: Sequence[dict],
) -> None:
    lines = [
        "# Intent Ambiguity Analysis",
        "",
        f"- checkpoint: `{ckpt_path}`",
        f"- decision threshold: `{threshold:.4f}`",
        f"- selected intents: `{len(selected)}`",
        "",
        "## Selected Intents",
        "",
    ]

    for item in selected:
        lines.append(
            f"- `{item.bucket}` | `{item.intent_name}` | pos={item.positive_count} | TP={item.tp_count} | FN={item.fn_count} | FP={item.fp_count} | best_k={item.best_k} | silhouette={item.silhouette:.4f}"
        )

    lines.extend(["", "## Cluster Diagnostics", ""])
    for result in detailed_results:
        if "cluster_rows" not in result:
            continue
        lines.append(
            f"### {result['bucket'].upper()} / {result['intent_name']} (best_k={result['best_k']}, silhouette={result['silhouette']})"
        )
        lines.append(f"- scatter: `{result['scatter_path']}`")
        for row in result["cluster_rows"]:
            lines.append(
                f"- cluster {row['cluster_id']}: size={row['cluster_size']}, FN={row['fn_count']}, FN_rate={row['fn_rate']}, assigned_FP={row['assigned_fp_count']}, co-occur={row['top_cooccurring_intents']}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, run_cfg = load_run_model(args.hparams, args.ckpt, device=device)
    datamodule = load_data_module(batch_size=args.batch_size, num_workers=args.num_workers)

    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    annotation_file = Path(datamodule.hparams.annotation_dir) / datamodule.hparams.test_annotation
    class_names = get_class_names(str(annotation_file))

    print(f"Collecting validation artifacts on {device} ...")
    val_artifacts = collect_split_artifacts(model, val_loader, device=device)
    print(f"Collecting test artifacts on {device} ...")
    test_artifacts = collect_split_artifacts(model, test_loader, device=device)

    threshold = choose_threshold(model, val_artifacts)
    test_preds = predict_with_threshold(test_artifacts.probs, threshold)
    image_lookup = build_image_lookup(datamodule.data_test)

    summary_rows: List[dict] = []
    for intent_idx, intent_name in enumerate(class_names):
        positive_mask = test_artifacts.targets[:, intent_idx] > 0.5
        negative_mask = ~positive_mask
        tp_mask = positive_mask & (test_preds[:, intent_idx] == 1)
        fn_mask = positive_mask & (test_preds[:, intent_idx] == 0)
        fp_mask = negative_mask & (test_preds[:, intent_idx] == 1)

        positive_indices = np.where(positive_mask)[0]
        positive_features = test_artifacts.features[positive_indices]
        if positive_features.shape[0] >= args.min_positive_samples:
            best_k, silhouette = evaluate_best_k(
                positive_features,
                max_k=args.max_k,
                min_cluster_size=args.min_cluster_size,
                silhouette_threshold=args.silhouette_threshold,
                seed=args.seed,
            )
        else:
            best_k, silhouette = 1, 0.0

        summary_rows.append(
            {
                "intent_idx": intent_idx,
                "intent_name": intent_name,
                "positive_count": int(positive_mask.sum()),
                "tp_count": int(tp_mask.sum()),
                "fn_count": int(fn_mask.sum()),
                "fp_count": int(fp_mask.sum()),
                "best_k": int(best_k),
                "silhouette": round(float(silhouette), 4),
                "likely_multimodal": bool(best_k > 1 and silhouette >= args.silhouette_threshold),
                "avg_positive_score": round(
                    float(test_artifacts.probs[positive_mask, intent_idx].mean()) if positive_mask.any() else 0.0,
                    4,
                ),
                "avg_negative_score": round(
                    float(test_artifacts.probs[negative_mask, intent_idx].mean()) if negative_mask.any() else 0.0,
                    4,
                ),
            }
        )

    write_csv(args.output_dir / "all_intent_summary.csv", summary_rows)
    selected = select_bucket_intents(summary_rows, class_names=class_names, per_bucket=args.per_bucket)

    detailed_results: List[dict] = []
    cluster_rows: List[dict] = []
    for selection in selected:
        print(
            f"Analyzing {selection.bucket} intent: {selection.intent_name} "
            f"(pos={selection.positive_count}, TP={selection.tp_count}, FN={selection.fn_count}, FP={selection.fp_count})"
        )
        result = run_detailed_intent_analysis(
            selection=selection,
            test_artifacts=test_artifacts,
            test_preds=test_preds,
            image_lookup=image_lookup,
            class_names=class_names,
            output_dir=args.output_dir,
            args=args,
        )
        detailed_results.append(result)
        cluster_rows.extend(result.get("cluster_rows", []))

    write_csv(args.output_dir / "selected_intent_cluster_stats.csv", cluster_rows)
    write_markdown_report(
        output_path=args.output_dir / "report.md",
        ckpt_path=args.ckpt,
        threshold=threshold,
        selected=selected,
        detailed_results=detailed_results,
    )

    metadata = {
        "hparams": str(args.hparams),
        "ckpt": str(args.ckpt),
        "output_dir": str(args.output_dir),
        "decision_threshold": threshold,
        "device": str(device),
        "run_model_target": str(getattr(run_cfg.model, "_target_", "")),
        "selected_intents": [selection.__dict__ for selection in selected],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved ambiguity analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
