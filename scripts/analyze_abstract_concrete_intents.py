#!/usr/bin/env python3
"""Analyze abstract vs concrete intent groups across key methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


ABSTRACT_IDS = [
    3,   # CreativeUnique
    4,   # CuriousAdventurousExcitingLife
    5,   # EasyLife
    6,   # EnjoyLife
    9,   # FineDesignLearnArt-Culture
    10,  # GoodParentEmoCloseChild
    11,  # Happy
    12,  # HardWorking
    13,  # Harmony
    14,  # Health
    15,  # InLove
    17,  # InspirOthrs
    18,  # ManagableMakePlan
    20,  # PassionAbSmthing
    22,  # ShareFeelings
    23,  # SocialLifeFriendship
    24,  # SuccInOccupHavGdJob
    27,  # WorkILike
]

CONCRETE_IDS = [
    0,   # Attractive
    1,   # BeatCompete
    2,   # Communicate
    7,   # FineDesignLearnArt-Arch
    8,   # FineDesignLearnArt-Art
    16,  # InLoveAnimal
    19,  # NatBeauty
    21,  # Playful
    25,  # TchOthrs
    26,  # ThngsInOrdr
]

CLASS_NAMES = [
    "Attractive",
    "BeatCompete",
    "Communicate",
    "CreativeUnique",
    "CuriousAdventurousExcitingLife",
    "EasyLife",
    "EnjoyLife",
    "FineDesignLearnArt-Arch",
    "FineDesignLearnArt-Art",
    "FineDesignLearnArt-Culture",
    "GoodParentEmoCloseChild",
    "Happy",
    "HardWorking",
    "Harmony",
    "Health",
    "InLove",
    "InLoveAnimal",
    "InspirOthrs",
    "ManagableMakePlan",
    "NatBeauty",
    "PassionAbSmthing",
    "Playful",
    "ShareFeelings",
    "SocialLifeFriendship",
    "SuccInOccupHavGdJob",
    "TchOthrs",
    "ThngsInOrdr",
    "WorkILike",
]


def _mean_group(per_class_f1: List[float], ids: List[int]) -> float:
    values = np.asarray([float(per_class_f1[i]) for i in ids], dtype=np.float32)
    return float(values.mean())


def _gain_rows(
    baseline_per_class_f1: List[float],
    method_per_class_f1: List[float],
    ids: List[int],
    top_n: int = 5,
) -> List[Dict[str, float | str]]:
    rows = []
    for idx in ids:
        gain = float(method_per_class_f1[idx] - baseline_per_class_f1[idx])
        rows.append(
            {
                "class_id": idx,
                "class_name": CLASS_NAMES[idx],
                "baseline_f1": float(baseline_per_class_f1[idx]),
                "method_f1": float(method_per_class_f1[idx]),
                "gain": gain,
            }
        )
    rows.sort(key=lambda item: item["gain"], reverse=True)
    return rows[:top_n]


def main() -> None:
    cal_path = Path("logs/analysis/full_calibrated_decision_rule_20260310_renamed/summary.json")
    text_path = Path("logs/analysis/full_text_prior_boundary_20260310_renamed/summary.json")
    fusion_path = Path("logs/analysis/full_learnable_local_fusion_20260310_renamed/summary.json")

    calibrated = json.loads(cal_path.read_text(encoding="utf-8"))
    text = json.loads(text_path.read_text(encoding="utf-8"))
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))

    method_rows = {
        "baseline": calibrated["baseline"]["global"]["test"],
        "baseline_classwise": calibrated["baseline"]["classwise"]["test"],
        "scenario_slr": calibrated["slr_v0"]["retuned_global"]["test"],
        "scenario_slr_classwise": calibrated["slr_v0"]["classwise"]["test"],
        "scenario_slr_group_frequency": calibrated["slr_v0"]["group_frequency"]["test"],
        "lexical_plus_canonical_slr": calibrated["source_ensemble"]["lexical_plus_canonical"]["global"]["test"],
        "lexical_plus_canonical_slr_classwise": calibrated["source_ensemble"]["lexical_plus_canonical"]["classwise"]["test"],
        "best_learnable_fusion": fusion["best_overall_by_val_macro"]["test"],
    }

    abstract_concrete = {}
    baseline_per_class = method_rows["baseline"]["per_class_f1"]
    for method_name, metrics in method_rows.items():
        per_class = metrics["per_class_f1"]
        abstract_concrete[method_name] = {
            "macro": float(metrics["macro"]),
            "hard": float(metrics["hard"]),
            "abstract_mean_f1": _mean_group(per_class, ABSTRACT_IDS),
            "concrete_mean_f1": _mean_group(per_class, CONCRETE_IDS),
            "abstract_minus_concrete": _mean_group(per_class, ABSTRACT_IDS) - _mean_group(per_class, CONCRETE_IDS),
            "top_abstract_gain_classes": _gain_rows(baseline_per_class, per_class, ABSTRACT_IDS),
            "top_concrete_gain_classes": _gain_rows(baseline_per_class, per_class, CONCRETE_IDS),
        }

    output = {
        "group_definition": {
            "abstract_ids": ABSTRACT_IDS,
            "concrete_ids": CONCRETE_IDS,
            "abstract_names": [CLASS_NAMES[i] for i in ABSTRACT_IDS],
            "concrete_names": [CLASS_NAMES[i] for i in CONCRETE_IDS],
        },
        "results": abstract_concrete,
    }

    out_dir = Path("logs/analysis/full_abstract_concrete_analysis_20260310")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
