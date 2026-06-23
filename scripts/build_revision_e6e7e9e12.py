#!/usr/bin/env python3
"""Revision figures/tables: E6 (t-SNE), E7 (ROC/PR), E9 (prior-source ablation),
E12 (per-class attribution).

Reuses the deterministic score reconstruction of build_e13 (no retraining) from
the healthy E2 seed-20260616 checkpoints on the frozen CLIP ViT-L/14 cache.

Outputs (under paper/revision_1 for figures, logs/analysis for tables):
  - 4_roc_pr.pdf            (E7) micro-averaged ROC + PR curves, 4 methods
  - 5_tsne.pdf              (E6) t-SNE of decision-space embeddings, baseline vs FDIL
  - e9_prior_ablation.csv   (E9) SLR-C reranking-only metrics for 7 prior subsets
  - e12_per_class_gain.csv  (E12) per-class F1 (baseline vs FDIL) + difficulty subset

Run with the `s2d` env.
"""
from __future__ import annotations

# --- fabricate training-only deps so pure helpers import in a clip-only env ----
import sys, types, importlib.abc, importlib.machinery

_STUB_ROOTS = set(
    "lightning pytorch_lightning rich hydra omegaconf rootutils "
    "lightning_utilities torchmetrics wandb tensorboard".split()
)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return _Dummy()

    def __getattr__(self, _n):
        return _Dummy()


class _AutoModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (_Dummy,), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _AutoModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

# --- real imports -------------------------------------------------------------
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_privileged_distillation import StudentMLP  # noqa: E402
from scripts.analyze_distillation_slrc import (  # noqa: E402
    ResidualStudent,
    _apply_slr,
    _load_cache_bundle,
    _predict_residual_student,
    _sigmoid_np,
    _slr_feature_view,
    _text_logits_from_features,
)
from scripts.analyze_text_prior_boundary import (  # noqa: E402
    _build_text_pools,
    _encode_text_pool,
    _load_class_names,
)
from src.utils.decision_rule_calibration import search_classwise_thresholds  # noqa: E402
from src.utils.metrics import SUBSET2IDS  # noqa: E402

import clip  # type: ignore  # noqa: E402

CACHE = (
    PROJECT_ROOT / "logs" / "analysis"
    / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
TEACHER_RUN = PROJECT_ROOT / "logs" / "analysis" / "e2_privileged_distillation_seed20260616"
FDIL_RUN = PROJECT_ROOT / "logs" / "analysis" / "e2_distillation_slrc_lcs_topk10_seed20260616"
ANNOTATION = PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
GEMINI = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"
PAPER = PROJECT_ROOT / "paper" / "revision_1"
OUT = PROJECT_ROOT / "logs" / "analysis" / "e6e7e9e12_revision_20260623"
TOPK = 10  # adopted default candidate size
ALPHA = 0.3


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_student(state, image_dim, num_classes, device) -> StudentMLP:
    hidden_dim = int(state["encoder.0.weight"].shape[0])
    feature_proj_dim = int(state["feature_proj.0.weight"].shape[0])
    m = StudentMLP(image_dim, hidden_dim, num_classes, dropout=0.1, feature_proj_dim=feature_proj_dim).to(device)
    m.load_state_dict(state, strict=True)
    return m.eval()


def _build_residual(state, image_dim, num_classes, device) -> ResidualStudent:
    hidden_dim = int(state["net.0.weight"].shape[0])
    m = ResidualStudent(image_dim, hidden_dim, num_classes, dropout=0.1).to(device)
    m.load_state_dict(state, strict=True)
    return m.eval()


@torch.inference_mode()
def _logits(model: StudentMLP, feats, device, batch=256) -> np.ndarray:
    out = []
    for s in range(0, feats.shape[0], batch):
        x = torch.as_tensor(feats[s:s + batch], dtype=torch.float32, device=device)
        out.append(model(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


@torch.inference_mode()
def _encoder_feats(model: StudentMLP, feats, device, batch=256) -> np.ndarray:
    out = []
    for s in range(0, feats.shape[0], batch):
        x = torch.as_tensor(feats[s:s + batch], dtype=torch.float32, device=device)
        out.append(model.encoder(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


# ---- F1 metric helpers (calibrated, val-only thresholds) ---------------------
def _f1_suite(val_s, val_y, test_s, test_y) -> Dict[str, float]:
    thr = search_classwise_thresholds(val_s, val_y)
    pred = (test_s > thr[None, :]).astype(np.int32)
    tp = (pred * test_y).sum(0)
    fp = (pred * (1 - test_y)).sum(0)
    fn = ((1 - pred) * test_y).sum(0)
    denom = 2 * tp + fp + fn
    per_class = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-9), 0.0)
    macro = float(per_class.mean() * 100)
    # micro
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    micro = float(2 * TP / max(2 * TP + FP + FN, 1e-9) * 100)
    # samples F1
    s_tp = (pred * test_y).sum(1)
    s_fp = (pred * (1 - test_y)).sum(1)
    s_fn = ((1 - pred) * test_y).sum(1)
    s_den = 2 * s_tp + s_fp + s_fn
    samples = float(np.where(s_den > 0, 2 * s_tp / np.maximum(s_den, 1e-9), 0.0).mean() * 100)
    hard = float(per_class[SUBSET2IDS["hard"]].mean() * 100)
    return {
        "macro": macro, "micro": micro, "samples": samples,
        "avg": (macro + micro + samples) / 3.0, "hard": hard,
        "per_class": per_class * 100,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = _device()
    val = _load_cache_bundle(CACHE / "val_clip.npz")
    test = _load_cache_bundle(CACHE / "test_clip.npz")
    val_feat = np.asarray(val["features"], np.float32)
    test_feat = np.asarray(test["features"], np.float32)
    val_y = np.asarray(val["labels"], np.float32)
    test_y = np.asarray(test["labels"], np.float32)
    image_dim, num_classes = test_feat.shape[1], test_y.shape[1]

    # --- semantic prior (CLIP text) ---
    class_names = _load_class_names(ANNOTATION)
    pools = _build_text_pools(class_names, GEMINI)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lex = _encode_text_pool(clip_model, pools["lexical"], wrap_prompt=True)
    can = _encode_text_pool(clip_model, pools["canonical"], wrap_prompt=True)
    scen = _encode_text_pool(clip_model, pools["scenario"], wrap_prompt=False)

    def src_logits(bundle, emb):
        return _text_logits_from_features(_slr_feature_view(bundle), emb, logit_scale)

    val_src = {"lex": src_logits(val, lex), "can": src_logits(val, can), "scen": src_logits(val, scen)}
    test_src = {"lex": src_logits(test, lex), "can": src_logits(test, can), "scen": src_logits(test, scen)}

    # --- baseline / UTD / SLR-C / FDIL reconstruction ---
    base_state = torch.load(TEACHER_RUN / "baseline_best.pt", map_location="cpu", weights_only=True)
    base_model = _build_student(base_state, image_dim, num_classes, device)
    val_base_logits = _logits(base_model, val_feat, device)
    test_base_logits = _logits(base_model, test_feat, device)
    val_baseline, test_baseline = _sigmoid_np(val_base_logits), _sigmoid_np(test_base_logits)

    utd_state = torch.load(TEACHER_RUN / "dynamic_gated_kd_best.pt", map_location="cpu", weights_only=True)
    utd_model = _build_student(utd_state, image_dim, num_classes, device)
    val_utd, test_utd = _sigmoid_np(_logits(utd_model, val_feat, device)), _sigmoid_np(_logits(utd_model, test_feat, device))

    def full_prior(src):
        return (src["lex"] + src["can"] + src["scen"]) / 3.0

    val_slr_logits = _apply_slr(val_base_logits, full_prior(val_src), TOPK, ALPHA)
    test_slr_logits = _apply_slr(test_base_logits, full_prior(test_src), TOPK, ALPHA)
    val_slr, test_slr = _sigmoid_np(val_slr_logits), _sigmoid_np(test_slr_logits)

    fdil_state = torch.load(FDIL_RUN / "slr_c_residual_dynamic_kd_best.pt", map_location="cpu", weights_only=True)
    fdil_model = _build_residual(fdil_state, image_dim, num_classes, device)
    val_fdil = _predict_residual_student(fdil_model, val_feat, val_slr_logits, device, 256)
    test_fdil = _predict_residual_student(fdil_model, test_feat, test_slr_logits, device, 256)

    methods = {
        "CLIP baseline": (val_baseline, test_baseline),
        "UTD only": (val_utd, test_utd),
        "SLR-C only": (val_slr, test_slr),
        "FDIL": (val_fdil, test_fdil),
    }

    # ============================ E7: ROC / PR ============================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = {"CLIP baseline": "#888888", "UTD only": "#2c7fb8",
              "SLR-C only": "#41ab5d", "FDIL": "#e6550d"}
    for name, (_, ts) in methods.items():
        y = test_y.ravel().astype(int)
        s = ts.ravel()
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                     color=colors[name], lw=2 if name == "FDIL" else 1.4)
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                     color=colors[name], lw=2 if name == "FDIL" else 1.4)
    axes[0].plot([0, 1], [0, 1], ls="--", color="lightgray", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("(a) Micro-averaged ROC"); axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Micro-averaged Precision-Recall"); axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(PAPER / "4_roc_pr.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[E7] wrote", PAPER / "4_roc_pr.pdf")

    # ============================ E6: t-SNE ===============================
    # Decision-space embedding: penultimate encoder features for baseline,
    # SLR-C-fused residual representation for FDIL. Restrict to test samples
    # whose ground truth contains exactly one of a few semantically adjacent
    # hard classes, colour by that class.
    focus = SUBSET2IDS["hard"][:6]
    single = []
    labels_single = []
    for i in range(test_y.shape[0]):
        pos = np.where(test_y[i] > 0)[0]
        inter = [c for c in pos if c in focus]
        if len(inter) == 1 and len(pos) <= 2:
            single.append(i); labels_single.append(inter[0])
    single = np.asarray(single); labels_single = np.asarray(labels_single)

    base_emb = _encoder_feats(base_model, test_feat, device)

    @torch.inference_mode()
    def _residual_penult(model, feats, slr, device, batch=256):
        outs = []
        for s in range(0, feats.shape[0], batch):
            x = torch.as_tensor(feats[s:s + batch], dtype=torch.float32, device=device)
            h = x
            for layer in list(model.net)[:-1]:
                h = layer(h)
            outs.append(h.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, 0)

    fdil_emb = _residual_penult(fdil_model, test_feat, test_slr_logits, device)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    cmap = plt.get_cmap("tab10")
    name_map = {c: class_names[c] for c in focus}
    for ax, emb, title in [(axes[0], base_emb, "(a) CLIP baseline"),
                           (axes[1], fdil_emb, "(b) FDIL")]:
        Z = TSNE(n_components=2, perplexity=20, init="pca",
                 random_state=0, learning_rate="auto").fit_transform(emb[single])
        for j, c in enumerate(focus):
            m = labels_single == c
            ax.scatter(Z[m, 0], Z[m, 1], s=16, color=cmap(j % 10),
                       label=name_map[c], alpha=0.8, edgecolors="none")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER / "5_tsne.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[E6] wrote", PAPER / "5_tsne.pdf", "| n_single =", len(single))

    # ====================== E9: prior-source ablation =====================
    combos = {
        "Lexical": ["lex"], "Canonical": ["can"], "Scenario": ["scen"],
        "Lexical+Canonical": ["lex", "can"],
        "Lexical+Scenario": ["lex", "scen"],
        "Canonical+Scenario": ["can", "scen"],
        "Lexical+Canonical+Scenario": ["lex", "can", "scen"],
    }
    rows9 = []
    for name, keys in combos.items():
        vp = sum(val_src[k] for k in keys) / len(keys)
        tp = sum(test_src[k] for k in keys) / len(keys)
        vlog = _apply_slr(val_base_logits, vp, TOPK, ALPHA)
        tlog = _apply_slr(test_base_logits, tp, TOPK, ALPHA)
        r = _f1_suite(_sigmoid_np(vlog), val_y, _sigmoid_np(tlog), test_y)
        rows9.append({"combo": name, **{k: round(r[k], 2) for k in ["macro", "micro", "samples", "avg", "hard"]}})
        print(f"[E9] {name:30s} Macro {r['macro']:.2f} Micro {r['micro']:.2f} "
              f"Samples {r['samples']:.2f} Avg {r['avg']:.2f} Hard {r['hard']:.2f}")
    with (OUT / "e9_prior_ablation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows9[0].keys())); w.writeheader(); w.writerows(rows9)

    # ====================== E12: per-class attribution ====================
    base_suite = _f1_suite(val_baseline, val_y, test_baseline, test_y)
    fdil_suite = _f1_suite(val_fdil, val_y, test_fdil, test_y)
    diff = {c: "easy" for c in SUBSET2IDS["easy"]}
    diff.update({c: "medium" for c in SUBSET2IDS["medium"]})
    diff.update({c: "hard" for c in SUBSET2IDS["hard"]})
    rows12 = []
    for c in range(num_classes):
        rows12.append({
            "class_id": c, "class_name": class_names[c] if c < len(class_names) else str(c),
            "difficulty": diff.get(c, "?"),
            "baseline_f1": round(float(base_suite["per_class"][c]), 2),
            "fdil_f1": round(float(fdil_suite["per_class"][c]), 2),
            "gain": round(float(fdil_suite["per_class"][c] - base_suite["per_class"][c]), 2),
        })
    rows12.sort(key=lambda r: r["gain"], reverse=True)
    with (OUT / "e12_per_class_gain.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows12[0].keys())); w.writeheader(); w.writerows(rows12)
    print(f"[E12] baseline macro {base_suite['macro']:.2f} -> FDIL macro {fdil_suite['macro']:.2f}")
    print("[E12] top gains:", [(r["class_name"], r["gain"]) for r in rows12[:5]])
    print("[E12] top losses:", [(r["class_name"], r["gain"]) for r in rows12[-5:]])
    print("[done] artifacts in", OUT)


if __name__ == "__main__":
    main()
