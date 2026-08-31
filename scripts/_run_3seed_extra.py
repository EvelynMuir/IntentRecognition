#!/usr/bin/env python3
"""Driver: generate the extra 3-seed runs needed for the FDIL-full consistency pass.

Runs (all on cached CLIP features, lightweight MLP heads):
  - SLR-C residual FDIL at K in {10, 28} for seeds {0317h, 0615, 0616}
  - Unified joint-target KD (build_e3) for the two missing seeds {0317h, 0615}
    (seed 0616 already exists in e3_unified_vs_decoupled_seed20260616)

Run with the s2d env. Installs the lightning/rich/hydra meta-path stub so the
score-reconstruction + residual-training helpers import without the full stack.
"""
from __future__ import annotations

import sys
import types
import importlib.abc
import importlib.machinery

_STUB = set("lightning pytorch_lightning rich hydra omegaconf rootutils "
            "lightning_utilities torchmetrics wandb tensorboard".split())


class _D:
    def __init__(s, *a, **k): pass
    def __call__(s, *a, **k): return a[0] if (len(a) == 1 and callable(a[0])) else _D()
    def __getattr__(s, n): return _D()


class _M(types.ModuleType):
    __path__ = []
    def __getattr__(s, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return type(n, (_D,), {})


class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(s, fn, p=None, t=None):
        return importlib.machinery.ModuleSpec(fn, s) if fn.split(".")[0] in _STUB else None
    def create_module(s, spec): return _M(spec.name)
    def exec_module(s, m): pass


sys.meta_path.insert(0, _F())

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CACHE = "logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache"
SEEDS = {
    20260317: "logs/analysis/e2_privileged_distillation_seed20260317",
    20260615: "logs/analysis/e2_privileged_distillation_seed20260615",
    20260616: "logs/analysis/e2_privileged_distillation_seed20260616",
}
FDIL_SUMMARY = {
    20260317: "logs/analysis/e2_distillation_slrc_lcs_topk5_seed20260317/summary.json",
    20260615: "logs/analysis/e2_distillation_slrc_lcs_topk5_seed20260615/summary.json",
    20260616: "logs/analysis/e2_distillation_slrc_lcs_topk5_seed20260616/summary.json",
}

import scripts.analyze_distillation_slrc as slrc
import scripts.build_e3_unified_vs_decoupled as e3


def run_k(seed: int, k: int) -> None:
    out = f"logs/analysis/e2_distillation_slrc_lcs_topk{k}_seed{seed}"
    sys.argv = ["analyze_distillation_slrc.py", "--seed", str(seed), "--topk", str(k),
                "--prior-mode", "lexical_canonical_scenario",
                "--reuse-cache-dir", CACHE,
                "--teacher-run-dir", SEEDS[seed], "--output-dir", out]
    print(f"\n===== K={k} seed={seed} -> {out} =====", flush=True)
    slrc.main()


def run_joint(seed: int) -> None:
    out = f"logs/analysis/e3_unified_vs_decoupled_seed{seed}"
    sys.argv = ["build_e3_unified_vs_decoupled.py", "--seed", str(seed),
                "--fdil-summary", FDIL_SUMMARY[seed],
                "--utd-summary", f"{SEEDS[seed]}/summary.json",
                "--teacher-run-dir", SEEDS[seed], "--output-dir", out]
    print(f"\n===== joint-target KD seed={seed} -> {out} =====", flush=True)
    e3.main()


if __name__ == "__main__":
    for s in (20260317, 20260615, 20260616):
        for k in (10, 28):
            run_k(s, k)
    for s in (20260317, 20260615):
        run_joint(s)
    print("\n[ALL EXTRA 3-SEED RUNS DONE]", flush=True)
