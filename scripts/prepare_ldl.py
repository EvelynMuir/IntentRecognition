#!/usr/bin/env python3
"""Convert the Flickr-LDL/Twitter-LDL Caffe LMDB release to plain files.

The released MATLAB files are v7.3 (HDF5), and each LMDB value is a Caffe
``Datum`` containing a CHW BGR image plus eight unpacked ``float_data`` label
probabilities.  This script intentionally parses the tiny Datum wire format
directly, so installing the obsolete Caffe package is not required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import h5py
import lmdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.components.caffe_datum import datum_to_rgb, parse_caffe_datum  # noqa: E402


CLASS_NAMES_BY_DATASET = {
    "flickrldl": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
    "twitterldl": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
    "emotion6": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
}

CLASS_DEFINITIONS = {
    "amusement": "Enjoyment, humor, or playful entertainment.",
    "anger": "Strong displeasure, irritation, or hostility.",
    "awe": "Wonder or reverence caused by something impressive or vast.",
    "contentment": "Calm satisfaction, comfort, or peaceful happiness.",
    "disgust": "Revulsion, aversion, or strong dislike.",
    "excitement": "Energetic enthusiasm, anticipation, or stimulation.",
    "fear": "Alarm, anxiety, or response to danger and threat.",
    "sadness": "Sorrow, unhappiness, loss, or disappointment.",
    "joy": "Strong happiness, delight, pleasure, or celebration.",
    "surprise": "A sudden reaction to something unexpected, novel, or startling.",
    "neutral": "An emotionally balanced or low-arousal scene without a dominant positive or negative affect.",
}


def _parse_args() -> argparse.Namespace:
    default_root = PROJECT_ROOT.parent / "LDL"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ldl-root", type=Path, default=default_root)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["flickrldl", "twitterldl", "emotion6"],
        default=["flickrldl", "twitterldl"],
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=2,
        choices=[2, 3, 4, 5],
        help="One non-test official fold used for validation; the other three train folds remain training data.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Emotion6 validation ratio within official train.")
    parser.add_argument("--split-seed", type=int, default=20260821, help="Emotion6 stratified validation seed.")
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Also write individual JPEGs. The default keeps the original LMDB as an efficient image backend.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-all-labels",
        action="store_true",
        help="Read every 256x256 image payload to compare embedded labels with MAT. By default a deterministic sample is checked.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Debug-only cap per LMDB split.")
    return parser.parse_args()


def _decode_matlab_strings(handle: h5py.File, dataset: h5py.Dataset) -> list[str]:
    output = []
    for reference in np.asarray(dataset).ravel():
        codepoints = np.asarray(handle[reference]).ravel().astype(np.uint32)
        output.append("".join(chr(int(value)) for value in codepoints))
    return output


def load_config(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        result = {
            "image_names": np.asarray(_decode_matlab_strings(handle, handle["imgname"])),
            "distributions": np.asarray(handle["vote"], dtype=np.float32).T,
            "train_ind": np.asarray(handle["train_ind"], dtype=np.uint8).ravel(),
            "test_ind": np.asarray(handle["test_ind"], dtype=np.uint8).ravel(),
        }
        result["fold_index"] = (
            np.asarray(handle["split_index"], dtype=np.int16).ravel()
            if "split_index" in handle else np.zeros(result["train_ind"].shape, dtype=np.int16)
        )
        return result


def _iter_lmdb(
    path: Path, *, read_all_values: bool = False
) -> Iterator[Tuple[str, str, Dict[str, Any] | None]]:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(buffers=False) as transaction:
            cursor = transaction.cursor()
            for index, key in enumerate(cursor.iternext(keys=True, values=False)):
                key_text = key.decode("utf-8")
                original_name = key_text.split("_", 1)[1] if "_" in key_text else key_text
                # Values contain 196 KB raw images. Reading every value over NFS is
                # expensive, so the normal path checks a deterministic sample while
                # still enumerating every key. --verify-all-labels retains a full audit.
                should_read = read_all_values or index < 8 or index % 500 == 0
                value = transaction.get(key) if should_read else None
                yield key_text, original_name, parse_caffe_datum(value) if value is not None else None
    finally:
        env.close()


def _description_payload(dataset: str) -> Dict[str, Any]:
    class_names = CLASS_NAMES_BY_DATASET[dataset]
    return {
        "class_order_source": f"{dataset} release class order",
        "emotions": [
            {
                "emotion_name": name,
                "definition": CLASS_DEFINITIONS[name],
                "archetypes": [
                    {"text_query": f"a photo that evokes {name}"},
                    {"text_query": f"a visual scene expressing {CLASS_DEFINITIONS[name].lower()}"},
                ],
            }
            for name in class_names
        ],
    }


def prepare_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    root = args.ldl_root.resolve()
    class_names = CLASS_NAMES_BY_DATASET[dataset]
    config = load_config(root / "config" / "tmp" / f"{dataset}_config.mat")
    output_dir = root / "processed" / dataset
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    name_to_index = {str(name): idx for idx, name in enumerate(config["image_names"].tolist())}
    if len(name_to_index) != len(config["image_names"]):
        raise ValueError(f"{dataset}: duplicate image names in MATLAB config")

    validation_indices: set[int] = set()
    if dataset == "emotion6":
        if not 0.0 < float(args.val_ratio) < 1.0:
            raise ValueError("--val-ratio must be in (0, 1)")
        rng = np.random.RandomState(int(args.split_seed))
        hard = np.argmax(config["distributions"], axis=1)
        official_train = np.flatnonzero(config["train_ind"] == 1)
        for class_id in range(len(class_names)):
            candidates = official_train[hard[official_train] == class_id]
            shuffled = candidates[rng.permutation(len(candidates))]
            count = max(1, int(round(len(candidates) * float(args.val_ratio))))
            validation_indices.update(int(index) for index in shuffled[:count])

    seen: set[int] = set()
    records: list[Dict[str, Any]] = []
    max_label_error = 0.0
    for official_split in ("train", "test"):
        lmdb_name = (
            f"{official_split}_{dataset}_lmdb"
            if dataset == "emotion6" else f"{official_split}_{dataset}_split1_lmdb"
        )
        lmdb_path = root / "data" / lmdb_name
        if not (lmdb_path / "data.mdb").exists():
            raise FileNotFoundError(f"Extracted LMDB not found: {lmdb_path}")
        for local_idx, (lmdb_key, image_name, datum) in enumerate(
            _iter_lmdb(
                lmdb_path,
                read_all_values=bool(args.extract_images or args.verify_all_labels),
            )
        ):
            if args.max_samples is not None and local_idx >= int(args.max_samples):
                break
            if image_name not in name_to_index:
                raise KeyError(f"{dataset}: LMDB image {image_name!r} is absent from config")
            index = name_to_index[image_name]
            if index in seen:
                raise ValueError(f"{dataset}: duplicate LMDB image {image_name}")
            seen.add(index)

            expected = config["distributions"][index]
            if datum is not None:
                actual = np.asarray(datum["float_data"], dtype=np.float32)
                if actual.shape != expected.shape:
                    raise ValueError(f"{dataset}/{image_name}: label shape {actual.shape}, expected {expected.shape}")
                max_label_error = max(max_label_error, float(np.max(np.abs(actual - expected))))

            image_path = image_dir / image_name
            if args.extract_images and datum is not None and (args.overwrite or not image_path.exists()):
                image = datum_to_rgb(datum)
                image.save(image_path, format="JPEG", quality=int(args.jpeg_quality), subsampling=0)

            fold = int(config["fold_index"][index])
            if official_split == "test":
                fdil_split = "test"
            elif dataset == "emotion6":
                fdil_split = "val" if index in validation_indices else "train"
            else:
                fdil_split = "val" if fold == args.val_fold else "train"
            records.append(
                {
                    "original_index": index,
                    "image_id": f"{dataset}:{image_name}",
                    "image_name": image_name,
                    "image_path": f"images/{image_name}" if args.extract_images else "",
                    "lmdb_key": lmdb_key,
                    "official_split": official_split,
                    "fdil_split": fdil_split,
                    "fold_index": fold,
                    "distribution": expected.tolist(),
                    "hard_label": int(np.argmax(expected)),
                    "agreement": float(np.max(expected)),
                    "entropy": float(-(expected * np.log(np.clip(expected, 1e-12, 1.0))).sum()),
                }
            )

    records.sort(key=lambda item: int(item["original_index"]))
    if args.max_samples is None and len(records) != len(config["image_names"]):
        missing = len(config["image_names"]) - len(records)
        raise ValueError(f"{dataset}: expected {len(config['image_names'])} records, found {len(records)} ({missing} missing)")
    if max_label_error > 5e-6:
        raise ValueError(f"{dataset}: LMDB/MAT label mismatch, max error={max_label_error}")

    arrays = {
        "image_ids": np.asarray([r["image_id"] for r in records]),
        "image_paths": np.asarray([r["image_path"] for r in records]),
        "lmdb_keys": np.asarray([r["lmdb_key"] for r in records]),
        "distributions": np.asarray([r["distribution"] for r in records], dtype=np.float32),
        "hard_labels": np.asarray([r["hard_label"] for r in records], dtype=np.int64),
        "agreement": np.asarray([r["agreement"] for r in records], dtype=np.float32),
        "entropy": np.asarray([r["entropy"] for r in records], dtype=np.float32),
        "official_split": np.asarray([r["official_split"] for r in records]),
        "fdil_split": np.asarray([r["fdil_split"] for r in records]),
        "fold_index": np.asarray([r["fold_index"] for r in records], dtype=np.int16),
        "original_indices": np.asarray([r["original_index"] for r in records], dtype=np.int64),
        "class_names": np.asarray(class_names),
    }
    np.savez_compressed(output_dir / "metadata.npz", **arrays)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "emotion_descriptions.json").write_text(
        json.dumps(_description_payload(dataset), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    split_counts = {name: int(np.sum(arrays["fdil_split"] == name)) for name in ("train", "val", "test")}
    summary = {
        "dataset": dataset,
        "num_samples": len(records),
        "num_classes": len(class_names),
        "class_names": class_names,
        "validation_policy": (
            {"type": "stratified_holdout", "ratio": float(args.val_ratio), "seed": int(args.split_seed)}
            if dataset == "emotion6" else {"type": "official_fold", "fold": int(args.val_fold)}
        ),
        "split_counts": split_counts,
        "max_lmdb_mat_label_error": max_label_error,
        "label_verification": "all" if args.verify_all_labels else "first_8_and_every_500th",
        "image_backend": "jpeg" if args.extract_images else "lmdb",
        "distribution_row_sum_range": [
            float(arrays["distributions"].sum(1).min()),
            float(arrays["distributions"].sum(1).max()),
        ],
        "note": "labels is argmax one-hot only for legacy compatibility; train LDL/FDIL with distributions (soft_labels).",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summaries = []
    for dataset in args.datasets:
        summary = prepare_dataset(args, dataset)
        summaries.append(summary)
        print(f"[LDL] {dataset}: {summary['split_counts']} -> {args.ldl_root / 'processed' / dataset}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
