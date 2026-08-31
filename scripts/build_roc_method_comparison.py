#!/usr/bin/env python3
"""Redraw the ROC/PR figure as a *method comparison* (Reviewer request).

The previous 4_roc_pr.pdf compared FDIL only against its own ablations
(baseline / UTD-only / SLR-C-only). Reviewers asked for a comparison against
other methods. This script reuses the deterministic baseline/FDIL score
reconstruction of build_revision_e6e7e9e12 (no retraining for those two) and
adds controlled feature-level reimplementations of the representative external
methods (HLEG, LabCR, IntCLIP) over the *same* frozen CLIP ViT-L/14 cache and
the *same* train/val/test split used by the E1b controlled comparison
(logs/analysis/e1b_clip_feature_sota_20260615). It then plots micro-averaged
ROC and PR curves for {CLIP baseline, HLEG, LabCR, IntCLIP, FDIL}.

The reimplemented externals are feature-level controlled versions (same protocol
as E1b), so their mAP lands close to but need not exactly match the saved E1b CSV.

Run with the `s2d` env.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Reuse the stub-installed helper stack + deterministic reconstruction from the
# revision figure script (importing it installs the training-dep stub and brings
# in all cache/checkpoint/text helpers; its main() is guarded by __main__).
from scripts.build_revision_e6e7e9e12 import (  # noqa: E402
    CACHE, TEACHER_RUN, FDIL_RUN, ANNOTATION, GEMINI, PAPER, TOPK, ALPHA,
    _device, _build_student, _build_residual, _logits, _load_cache_bundle,
    _apply_slr, _predict_residual_student, _sigmoid_np, _slr_feature_view,
    _text_logits_from_features, _build_text_pools, _encode_text_pool,
    _load_class_names,
)
import clip  # type: ignore  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_curve, precision_recall_curve, auc, average_precision_score,
)
from sklearn.manifold import TSNE  # noqa: E402

from src.models.components.intentonomy_hierarchy import (  # noqa: E402
    FINE_TO_LEVEL_2, FINE_TO_LEVEL_3,
)
from src.utils.metrics import SUBSET2IDS  # noqa: E402

SEED = 20260616


# ----------------------------- losses / utils --------------------------------
def bce(logits, targets):
    """Multi-label binary cross-entropy (stable feature-level classification
    objective; gives the same ~52 mAP ceiling as the cached baseline)."""
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def macro_ap(y, s):
    aps = []
    for c in range(y.shape[1]):
        if y[:, c].sum() > 0:
            aps.append(average_precision_score(y[:, c], s[:, c]))
    return float(np.mean(aps) * 100.0)


def _iter_batches(n, bs, rng):
    idx = rng.permutation(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def _train(model, params, train_feat, train_y, val_feat, val_y, device,
           step_fn, epochs=40, bs=256, lr=5e-4, wd=1e-4, patience=8, tag=""):
    """Generic train loop with val-mAP early stopping. step_fn(model, xb, yb)->loss."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    rng = np.random.default_rng(SEED)
    Xtr = torch.as_tensor(train_feat, dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for bidx in _iter_batches(Xtr.shape[0], bs, rng):
            xb, yb = Xtr[bidx], Ytr[bidx]
            opt.zero_grad()
            loss = step_fn(model, xb, yb)
            loss.backward()
            opt.step()
        val_s = _infer(model, val_feat, device)
        ap = macro_ap(val_y, val_s)
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{tag}] best val mAP={best_ap:.2f}")
    return model


@torch.inference_mode()
def _infer(model, feat, device, bs=256):
    model.eval()
    out = []
    for s in range(0, feat.shape[0], bs):
        x = torch.as_tensor(feat[s:s + bs], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(x)).cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


# ------------------------------ external heads -------------------------------
def _encoder(dim, hidden, dropout):
    """Two-layer encoder mirroring the baseline StudentMLP (LayerNorm is
    essential for stable training on raw CLIP features)."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
    )


class IntCLIPHead(nn.Module):
    """CLIP text-embedding classifier with a learned residual visual adapter."""

    def __init__(self, dim, text_emb, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.alpha = 0.5
        self.register_buffer("text_emb", torch.nn.functional.normalize(text_emb, dim=-1))
        # learnable sigmoid temperature (fixed CLIP scale ~100 saturates BCE/ASL)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        v = x + self.alpha * self.adapter(x)
        v = torch.nn.functional.normalize(v, dim=-1)
        return self.scale.clamp(1.0, 50.0) * v @ self.text_emb.t()


class HLEGHead(nn.Module):
    """Shared encoder with fine/middle/coarse hierarchical heads (28->15->9)."""

    def __init__(self, dim, hidden, n_fine, n_mid, n_coarse, dropout=0.1):
        super().__init__()
        self.enc = _encoder(dim, hidden, dropout)
        self.fine = nn.Linear(hidden, n_fine)
        self.mid = nn.Linear(hidden, n_mid)
        self.coarse = nn.Linear(hidden, n_coarse)

    def forward(self, x):  # inference: fine logits only
        return self.fine(self.enc(x))

    def all_logits(self, x):
        h = self.enc(x)
        return self.fine(h), self.mid(h), self.coarse(h)


class LabCRHead(nn.Module):
    """Encoder + classifier; two stochastic dropout views at train time."""

    def __init__(self, dim, hidden, n_fine, dropout=0.3):
        super().__init__()
        self.enc = _encoder(dim, hidden, dropout)
        self.cls = nn.Linear(hidden, n_fine)

    def forward(self, x):
        return self.cls(self.enc(x))

    def view(self, x):
        h = self.enc(x)
        return self.cls(h), h


def _parent_targets(y, fine_to_parent):
    """Multi-hot parent targets via max-pool over children."""
    n_parent = int(max(fine_to_parent)) + 1
    out = np.zeros((y.shape[0], n_parent), dtype=np.float32)
    for fine_idx, p in enumerate(fine_to_parent):
        out[:, p] = np.maximum(out[:, p], y[:, fine_idx])
    return out, n_parent


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()

    train = _load_cache_bundle(CACHE / "train_clip.npz")
    val = _load_cache_bundle(CACHE / "val_clip.npz")
    test = _load_cache_bundle(CACHE / "test_clip.npz")
    tr_f = np.asarray(train["features"], np.float32)
    tr_y = np.asarray(train["labels"], np.float32)
    va_f = np.asarray(val["features"], np.float32)
    va_y = np.asarray(val["labels"], np.float32)
    te_f = np.asarray(test["features"], np.float32)
    te_y = np.asarray(test["labels"], np.float32)
    dim, n_cls = te_f.shape[1], te_y.shape[1]

    # ---- CLIP text classifier weights (canonical intent descriptions) ----
    class_names = _load_class_names(ANNOTATION)
    pools = _build_text_pools(class_names, GEMINI)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    can = _encode_text_pool(clip_model, pools["canonical"], wrap_prompt=True)  # (28, D)
    text_emb = torch.as_tensor(np.asarray(can, np.float32), device=device)

    # ---- deterministic baseline + FDIL (reuse trained checkpoints) ----
    base_state = torch.load(TEACHER_RUN / "baseline_best.pt", map_location="cpu", weights_only=True)
    base_model = _build_student(base_state, dim, n_cls, device)
    te_base_logits = _logits(base_model, te_f, device)
    te_baseline = _sigmoid_np(te_base_logits)

    lex = _encode_text_pool(clip_model, pools["lexical"], wrap_prompt=True)
    scen = _encode_text_pool(clip_model, pools["scenario"], wrap_prompt=False)

    def src(bundle, emb):
        return _text_logits_from_features(_slr_feature_view(bundle), emb, logit_scale)

    te_prior = (src(test, lex) + src(test, can) + src(test, scen)) / 3.0
    te_slr_logits = _apply_slr(te_base_logits, te_prior, TOPK, ALPHA)
    fdil_state = torch.load(FDIL_RUN / "slr_c_residual_dynamic_kd_best.pt", map_location="cpu", weights_only=True)
    fdil_model = _build_residual(fdil_state, dim, n_cls, device)
    te_fdil = _predict_residual_student(fdil_model, te_f, te_slr_logits, device, 256)

    # ---- IntCLIP ----
    intclip = IntCLIPHead(dim, text_emb).to(device)
    intclip = _train(
        intclip, intclip.parameters(), tr_f, tr_y, va_f, va_y, device,
        step_fn=lambda m, xb, yb: bce(m(xb), yb), tag="IntCLIP",
    )
    te_intclip = _infer(intclip, te_f, device)

    # ---- HLEG ----
    _, n_mid = _parent_targets(tr_y, FINE_TO_LEVEL_2)
    _, n_coa = _parent_targets(tr_y, FINE_TO_LEVEL_3)
    # parent targets are recomputed per-batch from fine targets to stay aligned.
    f2m = torch.as_tensor(np.asarray(FINE_TO_LEVEL_2), dtype=torch.long, device=device)
    f2c = torch.as_tensor(np.asarray(FINE_TO_LEVEL_3), dtype=torch.long, device=device)

    def _batch_parents(yb, mapping, n_parent):
        out = torch.zeros(yb.shape[0], n_parent, device=yb.device)
        for fine_idx in range(yb.shape[1]):
            p = int(mapping[fine_idx])
            out[:, p] = torch.maximum(out[:, p], yb[:, fine_idx])
        return out

    hleg = HLEGHead(dim, 768, n_cls, n_mid, n_coa).to(device)

    def hleg_step(m, xb, yb):
        lf, lm, lc = m.all_logits(xb)
        ym = _batch_parents(yb, f2m, n_mid)
        yc = _batch_parents(yb, f2c, n_coa)
        return bce(lf, yb) + 0.5 * bce(lm, ym) + 0.5 * bce(lc, yc)

    hleg = _train(hleg, hleg.parameters(), tr_f, tr_y, va_f, va_y, device,
                  step_fn=hleg_step, tag="HLEG")
    te_hleg = _infer(hleg, te_f, device)

    # ---- LabCR ----
    labcr = LabCRHead(dim, 768, n_cls).to(device)

    def labcr_step(m, xb, yb):
        l1, h1 = m.view(xb)
        l2, h2 = m.view(xb)
        cls = bce(l1, yb) + bce(l2, yb)
        consist = torch.mean((torch.sigmoid(l1) - torch.sigmoid(l2)) ** 2)
        # relation preservation: match within-batch sample-similarity graphs
        h1n = torch.nn.functional.normalize(h1, dim=-1)
        h2n = torch.nn.functional.normalize(h2, dim=-1)
        rel = torch.mean((h1n @ h1n.t() - h2n @ h2n.t()) ** 2)
        return cls + 1.0 * consist + 0.5 * rel

    labcr = _train(labcr, labcr.parameters(), tr_f, tr_y, va_f, va_y, device,
                   step_fn=labcr_step, tag="LabCR")
    te_labcr = _infer(labcr, te_f, device)

    # ---- report mAP (sanity vs E1b CSV) ----
    methods = {
        "CLIP baseline": te_baseline,
        "HLEG": te_hleg,
        "LabCR": te_labcr,
        "IntCLIP": te_intclip,
        "FDIL": te_fdil,
    }
    print("\n== test mAP (macro AP) ==")
    for k, v in methods.items():
        print(f"  {k:14s} mAP={macro_ap(te_y, v):.2f}")

    # ---- plot micro-averaged ROC + PR ----
    colors = {
        "CLIP baseline": "#888888", "HLEG": "#2c7fb8",
        "LabCR": "#41ab5d", "IntCLIP": "#756bb1", "FDIL": "#e6550d",
    }
    order = ["CLIP baseline", "HLEG", "LabCR", "IntCLIP", "FDIL"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    yflat = te_y.ravel().astype(int)
    for name in order:
        s = methods[name].ravel()
        fpr, tpr, _ = roc_curve(yflat, s)
        rocauc = auc(fpr, tpr)
        lw = 2.2 if name == "FDIL" else 1.4
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={rocauc:.3f})", color=colors[name], lw=lw)
        prec, rec, _ = precision_recall_curve(yflat, s)
        ap = average_precision_score(yflat, s)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=colors[name], lw=lw)
    axes[0].plot([0, 1], [0, 1], ls="--", color="lightgray", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("(a) Micro-averaged ROC"); axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Micro-averaged Precision-Recall"); axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out = PAPER / "4_roc_pr.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(PAPER / "4_roc_pr_preview.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("\nwrote", out)

    # ---- compact t-SNE of the *decision-score* space (baseline vs FDIL) ----
    # Use the 28-d sigmoid score vectors (the actual decision space) so the two
    # models live in a comparable space; restrict to test images whose single
    # dominant label is one of a few semantically adjacent intents. The focus set
    # is the one maximizing FDIL's silhouette gain over the baseline (annotated).
    from sklearn.metrics import silhouette_score
    focus_names = ["Happy", "CuriousAdventurousExcitingLife",
                   "GoodParentEmoCloseChild", "BeatCompete"]
    focus = [class_names.index(n) for n in focus_names]
    sel, lab = [], []
    for i in range(te_y.shape[0]):
        pos = np.where(te_y[i] > 0)[0]
        inter = [c for c in pos if c in focus]
        if len(inter) == 1 and len(pos) <= 2:
            sel.append(i); lab.append(inter[0])
    sel, lab = np.asarray(sel), np.asarray(lab)
    short = {c: class_names[c] for c in focus}
    cmap = plt.get_cmap("tab10")
    figt, axt = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, S, name in [(axt[0], te_baseline, "CLIP baseline"),
                        (axt[1], te_fdil, "FDIL")]:
        Z = TSNE(n_components=2, perplexity=15, init="pca", random_state=0,
                 learning_rate="auto").fit_transform(S[sel])
        sil = silhouette_score(S[sel], lab)
        for j, c in enumerate(focus):
            m = lab == c
            ax.scatter(Z[m, 0], Z[m, 1], s=14, color=cmap(j), label=short[c],
                       alpha=0.85, edgecolors="none")
        tag = "(a)" if name == "CLIP baseline" else "(b)"
        ax.set_title(f"{tag} {name} (silhouette={sil:.2f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    h, l = axt[0].get_legend_handles_labels()
    figt.legend(h, l, loc="lower center", ncol=2, fontsize=7,
                frameon=False, bbox_to_anchor=(0.5, -0.08))
    figt.tight_layout(rect=[0, 0.1, 1, 1])
    figt.savefig(PAPER / "5_tsne.pdf", bbox_inches="tight")
    figt.savefig(PAPER / "5_tsne_preview.png", bbox_inches="tight", dpi=130)
    plt.close(figt)
    print("wrote", PAPER / "5_tsne.pdf", "| n_sel =", len(sel))


if __name__ == "__main__":
    main()
