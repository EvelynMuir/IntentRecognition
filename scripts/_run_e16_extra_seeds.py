#!/usr/bin/env python3
"""Train the two additional E2 seeds needed to take E16 from n=3 to n=5.

Reviewer 6 asked for a paired t-test; with only the three saved seeds the test
has 2 degrees of freedom and almost nothing survives the Holm correction. This
driver produces, for each new seed, the same two artefacts the existing seeds
have, so `build_e16_seed_significance.py --seeds ...` can pair on five runs:

  logs/analysis/e2_privileged_distillation_seed{seed}   (CLIP baseline + UTD-only)
  logs/analysis/e2_distillation_slrc_lcs_topk10_seed{seed}   (SLR-C + FDIL)

Mirrors scripts/_run_3seed_extra.py (argv injection under the training-dep stub).
Run from the project root with the `s2d` env.
"""

from __future__ import annotations

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
        return None


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = "logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache"
NEW_SEEDS = (20260801, 20260802)
TOPK = 10

import scripts.analyze_privileged_distillation as priv  # noqa: E402
import scripts.analyze_distillation_slrc as slrc  # noqa: E402


def run_privileged(seed: int) -> str:
    out = f"logs/analysis/e2_privileged_distillation_seed{seed}"
    sys.argv = ["analyze_privileged_distillation.py", "--seed", str(seed),
                "--reuse-cache-dir", CACHE, "--output-dir", out]
    print(f"\n===== privileged (baseline + UTD) seed={seed} -> {out} =====", flush=True)
    priv.main()
    return out


def run_slrc(seed: int, teacher_run: str) -> str:
    out = f"logs/analysis/e2_distillation_slrc_lcs_topk{TOPK}_seed{seed}"
    sys.argv = ["analyze_distillation_slrc.py", "--seed", str(seed), "--topk", str(TOPK),
                "--prior-mode", "lexical_canonical_scenario",
                "--reuse-cache-dir", CACHE,
                "--teacher-run-dir", teacher_run, "--output-dir", out]
    print(f"\n===== SLR-C + FDIL K={TOPK} seed={seed} -> {out} =====", flush=True)
    slrc.main()
    return out


if __name__ == "__main__":
    for s in NEW_SEEDS:
        teacher = run_privileged(s)
        run_slrc(s, teacher)
    print("\n[E16 EXTRA SEEDS DONE]", flush=True)
