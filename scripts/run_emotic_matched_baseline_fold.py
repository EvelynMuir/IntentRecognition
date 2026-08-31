#!/usr/bin/env python3
"""Run one or all EMOTIC 5x5 folds for a matched-backbone baseline."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_privileged_distillation import _evaluate_score_bundle, _json_ready, compute_mAP

SEEDS = [20260625, 20260626, 20260627, 20260628, 20260629]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["cocoer", "emotionclip"])
    p.add_argument("--feature-dir", default="logs/analysis/emotic_matched_vitl14_features")
    p.add_argument("--output-dir", default="logs/analysis/emotic_matched_baselines_vitl14")
    p.add_argument("--seed-index", type=int, default=None, choices=range(5))
    p.add_argument("--fold", type=int, default=None, choices=range(5))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(n: int, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = np.array_split(np.random.RandomState(seed).permutation(n), 5)
    test_idx = np.asarray(parts[fold], dtype=np.int64)
    val_idx = np.asarray(parts[(fold + 1) % 5], dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(n), np.concatenate([test_idx, val_idx]))
    return train_idx, val_idx, test_idx


class TokenDataset(Dataset):
    def __init__(self, feature_dir: Path, labels: np.ndarray, indices: np.ndarray):
        self.context = np.load(feature_dir / "context_tokens.npy", mmap_mode="r")
        self.body = np.load(feature_dir / "body_tokens.npy", mmap_mode="r")
        self.head = np.load(feature_dir / "head_tokens.npy", mmap_mode="r")
        self.vi = np.load(feature_dir / "vi_logits.npy", mmap_mode="r")
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(np.array(self.context[idx], dtype=np.float32)),
            torch.from_numpy(np.array(self.body[idx], dtype=np.float32)),
            torch.from_numpy(np.array(self.head[idx], dtype=np.float32)),
            torch.from_numpy(np.array(self.vi[idx], dtype=np.float32)),
            torch.from_numpy(self.labels[idx]),
        )


class CocoERMatchedHead(nn.Module):
    """Frozen-token CocoER adaptation retaining alignment, competition and coordination."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, classes: int = 26):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim)) for _ in range(3)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True) for _ in range(3)])
        self.stream_cls = nn.ModuleList([nn.Linear(hidden_dim, classes) for _ in range(3)])
        self.vi_proj = nn.Sequential(nn.Linear(classes, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1), nn.Linear(hidden_dim, classes)
        )

    def forward(self, context: torch.Tensor, body: torch.Tensor, head: torch.Tensor, vi: torch.Tensor):
        raw = [head, body, context]
        streams = [proj(x) for proj, x in zip(self.proj, raw)]
        ctx = streams[2]
        aligned = []
        for stream, attn in zip(streams, self.cross):
            delta, _ = attn(stream, ctx, ctx, need_weights=False)
            aligned.append(stream + delta)
        pooled = [x.mean(1) for x in aligned]
        stream_logits = torch.stack([head(x) for head, x in zip(self.stream_cls, pooled)], dim=1)

        # Competition: downweight levels whose categorical evidence conflicts with VI.
        vi_prob = torch.sigmoid(vi).unsqueeze(1)
        disagreement = F.binary_cross_entropy_with_logits(stream_logits, vi_prob.expand_as(stream_logits), reduction="none").mean(-1)
        weights = torch.softmax(-disagreement, dim=1)
        competitive = sum(weights[:, i : i + 1] * pooled[i] for i in range(3))

        # Coordination: reinforce evidence shared by all aligned levels and VI.
        coordinated = torch.stack(pooled, dim=1).mean(1) + self.vi_proj(vi)
        final_logits = self.final(torch.cat([competitive, coordinated], dim=-1))
        return final_logits, stream_logits


@torch.inference_mode()
def predict_cocoer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, targets = [], []
    for batch in loader:
        context, body, head, vi, labels = [x.to(device, non_blocking=True) for x in batch]
        logits, _ = model(context, body, head, vi)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())
    return np.concatenate(scores), np.concatenate(targets)


def run_cocoer(args: argparse.Namespace, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    feature_dir = Path(args.feature_dir)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TokenDataset(feature_dir, labels, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(TokenDataset(feature_dir, labels, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(TokenDataset(feature_dir, labels, test_idx), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = CocoERMatchedHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, best_epoch = None, 0, -1
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            context, body, head, vi, target = [x.to(device, non_blocking=True) for x in batch]
            logits, stream_logits = model(context, body, head, vi)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            stream_target = target[:, None, :].expand_as(stream_logits)
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(stream_logits, stream_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_scores, val_targets = predict_cocoer(model, val_loader, device)
        val_map = compute_mAP(val_scores, val_targets)
        print(f"[cocoer] seed={seed} epoch={epoch:02d} val_mAP={val_map:.4f}", flush=True)
        if best is None or val_map > best[0] + 1e-6:
            best = (val_map, copy.deepcopy(model.state_dict()))
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    model.load_state_dict(best[1])
    val_scores, val_targets = predict_cocoer(model, val_loader, device)
    test_scores, test_targets = predict_cocoer(model, test_loader, device)
    return val_scores, val_targets, test_scores, test_targets, best_epoch, best[1]


def run_emotionclip(args: argparse.Namespace, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    features = np.load(Path(args.feature_dir) / "emotionclip_features.npy", mmap_mode="r")
    clf = OneVsRestClassifier(LogisticRegression(random_state=seed, max_iter=2000, C=2.5, solver="sag", n_jobs=1), n_jobs=1)
    clf.fit(np.asarray(features[train_idx], dtype=np.float32), labels[train_idx])
    val_scores = clf.predict_proba(np.asarray(features[val_idx], dtype=np.float32)).astype(np.float32)
    test_scores = clf.predict_proba(np.asarray(features[test_idx], dtype=np.float32)).astype(np.float32)
    return val_scores, labels[val_idx], test_scores, labels[test_idx], None, None


def run_one(args: argparse.Namespace, seed_index: int, fold: int, labels: np.ndarray) -> None:
    seed = SEEDS[seed_index]
    out = Path(args.output_dir) / args.method / "folds" / f"seed_{seed}_fold_{fold}.json"
    pred_path = out.with_suffix(".npz")
    if out.exists() and pred_path.exists() and not args.overwrite:
        print(f"[resume] {out}")
        return
    seed_everything(seed + fold)
    train_idx, val_idx, test_idx = split_indices(len(labels), seed, fold)
    runner = run_cocoer if args.method == "cocoer" else run_emotionclip
    val_scores, val_targets, test_scores, test_targets, best_epoch, state = runner(
        args, labels, train_idx, val_idx, test_idx, seed + fold
    )
    bundle = _evaluate_score_bundle(val_scores, val_targets, test_scores, test_targets)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, val_idx=val_idx, test_idx=test_idx, val_scores=val_scores, test_scores=test_scores)
    if state is not None:
        torch.save(state, out.with_suffix(".pt"))
    payload = {
        "method": args.method,
        "display_name": "CocoER (CLIP ViT-L/14 matched adaptation)" if args.method == "cocoer" else "EmotionCLIP (CLIP ViT-L/14 mask-aware linear-probe adaptation)",
        "seed": seed,
        "fold": fold,
        "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_epoch": best_epoch,
        "bundle": _json_ready(bundle),
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    test = bundle["classwise"]["test"]
    print(f"[{args.method}] seed={seed} fold={fold} mAP={test['mAP']:.4f} macro={test['macro']*100:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = np.asarray(np.load(feature_dir / "labels.npy"), dtype=np.float32)
    if metadata.get("clip_model") != "ViT-L/14" or labels.shape != (10613, 26):
        raise RuntimeError(f"Unexpected matched feature protocol: {metadata}, labels={labels.shape}")
    seed_indices = [args.seed_index] if args.seed_index is not None else list(range(5))
    folds = [args.fold] if args.fold is not None else list(range(5))
    for seed_index in seed_indices:
        for fold in folds:
            run_one(args, int(seed_index), int(fold), labels)


if __name__ == "__main__":
    main()
