#!/usr/bin/env python3
"""E16 — seed-level statistical validation against matched-protocol SOTA.

Reviewer 6 asked for an explicit *paired t-test against the strongest
state-of-the-art methods*, arguing that mean+-std over three runs does not
establish statistical superiority. E13 answers a different question: it is a
paired bootstrap over *test samples* (n=1216) and only covers FDIL's own
ablations. This script supplies the missing evidence:

  * retrain the controlled feature-level reimplementations of HLEG, LabCR and
    IntCLIP (head classes reused verbatim from build_roc_method_comparison) once
    per seed, over the same frozen CLIP ViT-L/14 cache, the same split, and the
    same validation-only class-wise thresholding as FDIL;
  * pair every comparator with FDIL *on the seed*, and report mean, 95% CI
    (t-interval), the paired two-sided t-test, and Holm-corrected p-values.

The CLIP baseline / UTD-only / FDIL per-seed values are read from the saved E2
run summaries -- those are the exact runs the manuscript already reports, so the
significance test and the headline table stay consistent.

Note on provenance: the script that produced logs/analysis/
e1b_clip_feature_sota_20260615 (source of the manuscript's 54.09 / 53.94 / 52.77
reproduced-SOTA numbers) is no longer in the repository. Its protocol survives
only as the head classes + _train loop in build_roc_method_comparison.py, which
is what we reuse here. Regenerated numbers should land near, but need not equal,
that CSV; when they differ the manuscript should quote *these* numbers so the
table and the test come from one run.

Run with the `s2d` env (torch + clip + sklearn); a meta-path stub fabricates the
project's training-only deps so the pure helpers import without the full stack.
"""

from __future__ import annotations

# --- fabricate training-only deps so pure helpers import in a clip-only env ----
import sys
import types
import importlib.abc
import importlib.machinery

_STUB_ROOTS = set(
    "lightning pytorch_lightning rich hydra omegaconf rootutils "
    "lightning_utilities torchmetrics wandb tensorboard".split()
)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]  # behave as a no-op decorator
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
        return None


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip  # type: ignore  # noqa: E402
from scipy import stats as sps  # noqa: E402

import scripts.build_roc_method_comparison as roc  # noqa: E402
from scripts.build_revision_e6e7e9e12 import (  # noqa: E402
    ANNOTATION,
    GEMINI,
    _apply_slr,
    _build_residual,
    _build_student,
    _build_text_pools,
    _encode_text_pool,
    _load_cache_bundle,
    _load_class_names,
    _logits,
    _predict_residual_student,
    _sigmoid_np,
    _slr_feature_view,
    _text_logits_from_features,
)
from src.models.components.intentonomy_hierarchy import (  # noqa: E402
    FINE_TO_LEVEL_2,
    FINE_TO_LEVEL_3,
)
from src.utils.decision_rule_calibration import search_classwise_thresholds  # noqa: E402
from src.utils.metrics import SUBSET2IDS  # noqa: E402

DEFAULT_CACHE = (
    PROJECT_ROOT / "logs" / "analysis"
    / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
# Seeds with saved E2 runs for baseline / UTD-only / FDIL.
DEFAULT_SEEDS = (20260317, 20260615, 20260616)
# Additional seeds: no saved E2 run, so the *reproduced* comparators are trained
# here but FDIL cannot be paired on them. Kept configurable; see --extra-seeds.
DEFAULT_EXTRA_SEEDS: tuple = ()

TEACHER_RUN_TMPL = "e2_privileged_distillation_seed{seed}"
FDIL_RUN_TMPL = "e2_distillation_slrc_lcs_topk{topk}_seed{seed}"

# Guards against the known checkpoint-reload bug (see memory: SLR-C untrained
# base). A healthy SLR-C-only run reaches ~mAP 53 / macro-F1 49; the broken run
# collapsed to ~18.7. Anything below this floor means the base classifier failed
# to load and the seed must not be used.
SLRC_MAP_FLOOR = 40.0

METRICS = ("macro", "micro", "samples", "avg_f1", "mAP", "hard")
COMPARATORS = ("CLIP baseline", "UTD only", "HLEG", "LabCR", "IntCLIP")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E16 seed-level paired significance vs matched SOTA.")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--annotation-file", type=Path, default=ANNOTATION)
    p.add_argument("--gemini-file", type=Path, default=GEMINI)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--extra-seeds", type=int, nargs="*", default=list(DEFAULT_EXTRA_SEEDS),
        help="Seeds for which only the reproduced comparators are trained (no saved FDIL run).",
    )
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--slr-alpha", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- metrics -----------------------------------------------------------------
def _f1_suite(
    val_s: np.ndarray, val_y: np.ndarray, test_s: np.ndarray, test_y: np.ndarray
) -> Dict[str, float]:
    """Macro / Micro / Samples / Hard F1 under validation-only class-wise thresholds."""
    thr = search_classwise_thresholds(val_s, val_y)
    pred = (test_s > thr[None, :]).astype(np.float32)

    tp = (pred * test_y).sum(axis=0)
    fp = (pred * (1 - test_y)).sum(axis=0)
    fn = ((1 - pred) * test_y).sum(axis=0)
    denom = 2 * tp + fp + fn
    per_class = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-9), 0.0)

    micro_denom = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = 2 * tp.sum() / micro_denom if micro_denom > 0 else 0.0

    s_tp = (pred * test_y).sum(axis=1)
    s_denom = 2 * s_tp + (pred * (1 - test_y)).sum(axis=1) + ((1 - pred) * test_y).sum(axis=1)
    samples = float(np.mean(np.where(s_denom > 0, 2 * s_tp / np.maximum(s_denom, 1e-9), 0.0)))

    macro = float(np.mean(per_class))
    hard = float(np.mean(per_class[SUBSET2IDS["hard"]]))
    return {
        "macro": macro * 100.0,
        "micro": float(micro) * 100.0,
        "samples": samples * 100.0,
        "hard": hard * 100.0,
        "avg_f1": (macro + float(micro) + samples) / 3.0 * 100.0,
    }


def _metrics_row(
    val_s: np.ndarray, val_y: np.ndarray, test_s: np.ndarray, test_y: np.ndarray
) -> Dict[str, float]:
    row = _f1_suite(val_s, val_y, test_s, test_y)
    row["mAP"] = roc.macro_ap(test_y, test_s)
    return row


# ---- comparator training -----------------------------------------------------
def _batch_parents(yb: torch.Tensor, mapping: Sequence[int], n_parent: int) -> torch.Tensor:
    out = torch.zeros(yb.shape[0], n_parent, device=yb.device)
    for fine_idx in range(yb.shape[1]):
        out[:, int(mapping[fine_idx])] = torch.maximum(
            out[:, int(mapping[fine_idx])], yb[:, fine_idx]
        )
    return out


def _train_comparators(
    seed: int,
    data: Mapping[str, np.ndarray],
    text_emb: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Train IntCLIP / HLEG / LabCR heads for one seed; return val+test scores."""
    # roc._train seeds its batch-order RNG from the module-level SEED; rebind it
    # so each seed genuinely differs in both init and batch order.
    roc.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_f, tr_y = data["tr_f"], data["tr_y"]
    va_f, va_y = data["va_f"], data["va_y"]
    te_f = data["te_f"]
    dim, n_cls = te_f.shape[1], tr_y.shape[1]
    n_mid = int(max(FINE_TO_LEVEL_2)) + 1
    n_coa = int(max(FINE_TO_LEVEL_3)) + 1

    out: Dict[str, Dict[str, np.ndarray]] = {}

    intclip = roc.IntCLIPHead(dim, text_emb).to(device)
    intclip = roc._train(
        intclip, intclip.parameters(), tr_f, tr_y, va_f, va_y, device,
        step_fn=lambda m, xb, yb: roc.bce(m(xb), yb),
        epochs=epochs, patience=patience, tag=f"IntCLIP/{seed}",
    )
    out["IntCLIP"] = {
        "val": roc._infer(intclip, va_f, device),
        "test": roc._infer(intclip, te_f, device),
    }

    f2m = torch.as_tensor(np.asarray(FINE_TO_LEVEL_2), dtype=torch.long, device=device)
    f2c = torch.as_tensor(np.asarray(FINE_TO_LEVEL_3), dtype=torch.long, device=device)
    hleg = roc.HLEGHead(dim, 768, n_cls, n_mid, n_coa).to(device)

    def hleg_step(m, xb, yb):
        lf, lm, lc = m.all_logits(xb)
        return (
            roc.bce(lf, yb)
            + 0.5 * roc.bce(lm, _batch_parents(yb, f2m, n_mid))
            + 0.5 * roc.bce(lc, _batch_parents(yb, f2c, n_coa))
        )

    hleg = roc._train(
        hleg, hleg.parameters(), tr_f, tr_y, va_f, va_y, device,
        step_fn=hleg_step, epochs=epochs, patience=patience, tag=f"HLEG/{seed}",
    )
    out["HLEG"] = {
        "val": roc._infer(hleg, va_f, device),
        "test": roc._infer(hleg, te_f, device),
    }

    labcr = roc.LabCRHead(dim, 768, n_cls).to(device)

    def labcr_step(m, xb, yb):
        l1, h1 = m.view(xb)
        l2, h2 = m.view(xb)
        cls = roc.bce(l1, yb) + roc.bce(l2, yb)
        consist = torch.mean((torch.sigmoid(l1) - torch.sigmoid(l2)) ** 2)
        h1n = torch.nn.functional.normalize(h1, dim=-1)
        h2n = torch.nn.functional.normalize(h2, dim=-1)
        rel = torch.mean((h1n @ h1n.t() - h2n @ h2n.t()) ** 2)
        return cls + 1.0 * consist + 0.5 * rel

    labcr = roc._train(
        labcr, labcr.parameters(), tr_f, tr_y, va_f, va_y, device,
        step_fn=labcr_step, epochs=epochs, patience=patience, tag=f"LabCR/{seed}",
    )
    out["LabCR"] = {
        "val": roc._infer(labcr, va_f, device),
        "test": roc._infer(labcr, te_f, device),
    }
    return out


# ---- FDIL family reconstruction ---------------------------------------------
def _reconstruct_fdil_family(
    seed: int,
    topk: int,
    alpha: float,
    data: Mapping[str, np.ndarray],
    priors: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Rebuild CLIP-baseline / UTD-only / SLR-C-only / FDIL scores from saved E2
    checkpoints for one seed. Raises if a run directory is missing."""
    teacher_run = PROJECT_ROOT / "logs" / "analysis" / TEACHER_RUN_TMPL.format(seed=seed)
    fdil_run = PROJECT_ROOT / "logs" / "analysis" / FDIL_RUN_TMPL.format(topk=topk, seed=seed)
    for d in (teacher_run, fdil_run):
        if not d.is_dir():
            raise FileNotFoundError(f"missing saved E2 run for seed {seed}: {d}")

    va_f, te_f = data["va_f"], data["te_f"]
    dim, n_cls = te_f.shape[1], data["tr_y"].shape[1]

    base_state = torch.load(teacher_run / "baseline_best.pt", map_location="cpu", weights_only=True)
    base_model = _build_student(base_state, dim, n_cls, device)
    va_base_logits = _logits(base_model, va_f, device)
    te_base_logits = _logits(base_model, te_f, device)

    utd_state = torch.load(teacher_run / "dynamic_gated_kd_best.pt", map_location="cpu", weights_only=True)
    utd_model = _build_student(utd_state, dim, n_cls, device)

    va_slr_logits = _apply_slr(va_base_logits, priors["val"], topk, alpha)
    te_slr_logits = _apply_slr(te_base_logits, priors["test"], topk, alpha)

    fdil_state = torch.load(
        fdil_run / "slr_c_residual_dynamic_kd_best.pt", map_location="cpu", weights_only=True
    )
    fdil_model = _build_residual(fdil_state, dim, n_cls, device)

    return {
        "CLIP baseline": {
            "val": _sigmoid_np(va_base_logits),
            "test": _sigmoid_np(te_base_logits),
        },
        "UTD only": {
            "val": _sigmoid_np(_logits(utd_model, va_f, device)),
            "test": _sigmoid_np(_logits(utd_model, te_f, device)),
        },
        "SLR-C only": {
            "val": _sigmoid_np(va_slr_logits),
            "test": _sigmoid_np(te_slr_logits),
        },
        "FDIL": {
            "val": _predict_residual_student(fdil_model, va_f, va_slr_logits, device, 256),
            "test": _predict_residual_student(fdil_model, te_f, te_slr_logits, device, 256),
        },
    }


# ---- statistics --------------------------------------------------------------
def _mean_ci(values: Sequence[float]) -> Dict[str, float]:
    """Mean with a two-sided 95% t-interval (n-1 df)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(arr.std(ddof=1))
    half = float(sps.t.ppf(0.975, n - 1)) * std / np.sqrt(n)
    return {"n": n, "mean": mean, "std": std,
            "ci_low": mean - half, "ci_high": mean + half}


def _paired_ttest(fdil: Sequence[float], other: Sequence[float]) -> Dict[str, float]:
    """Paired two-sided t-test on the per-seed deltas, plus the delta's 95% CI."""
    a = np.asarray(fdil, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    d = a - b
    n = d.size
    out: Dict[str, float] = {"n": n, "mean_delta": float(d.mean())}
    if n < 2 or np.allclose(d, d[0]):
        out.update({"t_stat": float("nan"), "p_value": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan")})
        return out
    res = sps.ttest_rel(a, b)
    std = float(d.std(ddof=1))
    half = float(sps.t.ppf(0.975, n - 1)) * std / np.sqrt(n)
    out.update({
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "ci_low": float(d.mean()) - half,
        "ci_high": float(d.mean()) + half,
    })
    return out


def _holm(pvals: Mapping[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment over one comparator family."""
    items = [(k, v) for k, v in pvals.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {k: float("nan") for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ---- reporting ---------------------------------------------------------------
def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(_fmt(r.get(c, "")) for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def _p_str(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def main() -> None:
    args = _parse_args()
    device = _device(args.device)

    val = _load_cache_bundle(args.cache_dir / "val_clip.npz")
    test = _load_cache_bundle(args.cache_dir / "test_clip.npz")
    train = _load_cache_bundle(args.cache_dir / "train_clip.npz")
    data = {
        "tr_f": np.asarray(train["features"], np.float32),
        "tr_y": np.asarray(train["labels"], np.float32),
        "va_f": np.asarray(val["features"], np.float32),
        "va_y": np.asarray(val["labels"], np.float32),
        "te_f": np.asarray(test["features"], np.float32),
        "te_y": np.asarray(test["labels"], np.float32),
    }
    val_y, test_y = data["va_y"], data["te_y"]

    # ---- CLIP text pools: LCS prior (FDIL) + canonical classifier (IntCLIP) ----
    class_names = _load_class_names(args.annotation_file)
    pools = _build_text_pools(class_names, args.gemini_file)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lex = _encode_text_pool(clip_model, pools["lexical"], wrap_prompt=True)
    can = _encode_text_pool(clip_model, pools["canonical"], wrap_prompt=True)
    scen = _encode_text_pool(clip_model, pools["scenario"], wrap_prompt=False)
    text_emb = torch.as_tensor(np.asarray(can, np.float32), device=device)

    def prior(bundle: Mapping[str, Any]) -> np.ndarray:
        f = _slr_feature_view(bundle)
        return (
            _text_logits_from_features(f, lex, logit_scale)
            + _text_logits_from_features(f, can, logit_scale)
            + _text_logits_from_features(f, scen, logit_scale)
        ) / 3.0

    priors = {"val": prior(val), "test": prior(test)}

    # ---- per-seed evaluation --------------------------------------------------
    seed_rows: List[Dict[str, Any]] = []
    per_method: Dict[str, Dict[int, Dict[str, float]]] = {}
    integrity: List[Dict[str, Any]] = []

    fdil_seeds = list(args.seeds)
    all_seeds = fdil_seeds + [s for s in args.extra_seeds if s not in fdil_seeds]

    for seed in all_seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        scores: Dict[str, Dict[str, np.ndarray]] = {}
        if seed in fdil_seeds:
            scores.update(
                _reconstruct_fdil_family(seed, args.topk, args.slr_alpha, data, priors, device)
            )
        scores.update(
            _train_comparators(seed, data, text_emb, device, args.epochs, args.patience)
        )

        for method, s in scores.items():
            row = _metrics_row(s["val"], val_y, s["test"], test_y)
            per_method.setdefault(method, {})[seed] = row
            seed_rows.append({"method": method, "seed": seed, **row})
            print(f"  {method:14s} " + "  ".join(f"{k}={row[k]:.2f}" for k in METRICS), flush=True)

        # Guard against the known untrained-SLR-C-base checkpoint bug.
        if "SLR-C only" in scores:
            slrc_map = per_method["SLR-C only"][seed]["mAP"]
            ok = slrc_map >= SLRC_MAP_FLOOR
            integrity.append({"seed": seed, "slrc_only_mAP": slrc_map, "healthy": ok})
            if not ok:
                raise RuntimeError(
                    f"seed {seed}: SLR-C-only mAP={slrc_map:.2f} < {SLRC_MAP_FLOOR} -- the "
                    "baseline checkpoint almost certainly failed to load into the student "
                    "(untrained base). Refusing to report significance from this seed."
                )

    # ---- aggregate + paired tests --------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for method, by_seed in per_method.items():
        for metric in METRICS:
            vals = [by_seed[s][metric] for s in sorted(by_seed)]
            summary_rows.append({
                "method": method, "metric": metric,
                "seeds": " ".join(str(s) for s in sorted(by_seed)),
                **_mean_ci(vals),
            })

    test_rows: List[Dict[str, Any]] = []
    for metric in METRICS:
        raw: Dict[str, float] = {}
        staged: Dict[str, Dict[str, Any]] = {}
        for comp in COMPARATORS:
            if comp not in per_method:
                continue
            shared = sorted(set(per_method["FDIL"]) & set(per_method[comp]))
            if len(shared) < 2:
                continue
            res = _paired_ttest(
                [per_method["FDIL"][s][metric] for s in shared],
                [per_method[comp][s][metric] for s in shared],
            )
            staged[comp] = {"comparison": f"FDIL - {comp}", "metric": metric,
                            "seeds": " ".join(str(s) for s in shared), **res}
            raw[comp] = res["p_value"]
        adj = _holm(raw)
        for comp, row in staged.items():
            row["p_holm"] = adj.get(comp, float("nan"))
            test_rows.append(row)

    # ---- write ----------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (PROJECT_ROOT / "logs" / "analysis" / f"e16_seed_significance_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "e16_seed_level.csv", seed_rows, ["method", "seed", *METRICS])
    _write_csv(
        out_dir / "e16_mean_ci.csv", summary_rows,
        ["method", "metric", "seeds", "n", "mean", "std", "ci_low", "ci_high"],
    )
    _write_csv(
        out_dir / "e16_paired_ttests.csv", test_rows,
        ["comparison", "metric", "seeds", "n", "mean_delta", "ci_low", "ci_high",
         "t_stat", "p_value", "p_holm"],
    )

    (out_dir / "summary.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "cache_dir": str(args.cache_dir),
            "seeds_with_fdil": fdil_seeds,
            "extra_seeds": list(args.extra_seeds),
            "topk": args.topk,
            "slr_alpha": args.slr_alpha,
            "epochs": args.epochs,
            "patience": args.patience,
            "device": str(device),
            "thresholding": "validation-only class-wise",
        },
        "integrity": integrity,
        "seed_level": seed_rows,
        "mean_ci": summary_rows,
        "paired_ttests": test_rows,
    }, indent=2), encoding="utf-8")

    # ---- REPORT.md ------------------------------------------------------------
    lines = [
        "# E16 — Seed-Level Statistical Validation vs Matched-Protocol SOTA",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Seeds paired with FDIL: {', '.join(str(s) for s in fdil_seeds)}"
        f" (n={len(fdil_seeds)}).",
        "- CLIP baseline / UTD-only / SLR-C-only / FDIL reconstructed from the saved E2",
        "  checkpoints for each seed; HLEG / LabCR / IntCLIP retrained per seed as",
        "  controlled feature-level reimplementations (head classes reused from",
        "  `build_roc_method_comparison.py`).",
        "- Shared protocol: frozen CLIP ViT-L/14 cache, official Intentonomy split,",
        "  validation-only class-wise thresholds, test used for reporting only.",
        "- Paired two-sided t-test over seeds; Holm correction across the comparator",
        "  family within each metric.",
        "",
        "## Mean +- 95% CI (test, %)",
        "",
        "| Method | " + " | ".join(METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in METRICS) + " |",
    ]
    order = [m for m in ("CLIP baseline", "HLEG", "LabCR", "IntCLIP",
                         "SLR-C only", "UTD only", "FDIL") if m in per_method]
    stat_by = {(r["method"], r["metric"]): r for r in summary_rows}
    for method in order:
        cells = []
        for metric in METRICS:
            r = stat_by[(method, metric)]
            if np.isnan(r["ci_low"]):
                cells.append(f"{r['mean']:.2f}")
            else:
                cells.append(f"{r['mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired t-test — FDIL vs each comparator (percentage points)",
        "",
        "| Comparison | Metric | n | mean delta | CI95 low | CI95 high | t | p | p (Holm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in test_rows:
        lines.append(
            f"| {r['comparison']} | {r['metric']} | {r['n']} | {r['mean_delta']:+.3f} | "
            f"{r['ci_low']:.3f} | {r['ci_high']:.3f} | {r['t_stat']:.3f} | "
            f"{_p_str(r['p_value'])} | {_p_str(r['p_holm'])} |"
        )

    lines += [
        "",
        "## Checkpoint integrity",
        "",
        "| Seed | SLR-C-only mAP | healthy |",
        "| --- | ---: | --- |",
    ]
    for r in integrity:
        lines.append(f"| {r['seed']} | {r['slrc_only_mAP']:.2f} | {r['healthy']} |")
    lines += [
        "",
        f"A healthy SLR-C-only run reaches mAP ~53; the floor is {SLRC_MAP_FLOOR}. Values near",
        "18.7 indicate the baseline checkpoint failed to load into the student.",
        "",
        "## Caveat on the reproduced comparators",
        "",
        "The generator of `logs/analysis/e1b_clip_feature_sota_20260615` is no longer in the",
        "repository, so these retrained HLEG / LabCR / IntCLIP numbers need not match that",
        "CSV exactly. Where they differ, the manuscript should quote the values above so the",
        "reported table and the significance test come from a single run.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {out_dir}")
    print("\n".join(lines[-40:]))


if __name__ == "__main__":
    main()
