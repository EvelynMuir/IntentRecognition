#!/usr/bin/env python3
"""Build matched ViT-L/14 features for CocoER and EmotionCLIP adapters.

The existing FDIL full/crop global cache remains the source of labels and IDs.
This script only adds features that cache does not contain: mask-aware
EmotionCLIP embeddings and context/body/head spatial tokens for CocoER.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clip
from src.data.emotic_datamodule import EmoticPersonDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-file", default="../Emotic/Annotations/Annotations.mat")
    p.add_argument("--image-root", default="../Emotic/emotic")
    p.add_argument("--description-file", default="../Emotic/emotion_description_gemini.json")
    p.add_argument("--fdil-cache", default="logs/analysis/emotic_clip_dual_cache_full_20260323/_cache")
    p.add_argument("--output-dir", default="logs/analysis/emotic_matched_vitl14_features")
    p.add_argument("--clip-model", default="ViT-L/14", choices=["ViT-L/14"])
    p.add_argument("--method", default="all", choices=["all", "cocoer", "emotionclip"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--face-cache", default=None)
    p.add_argument("--disable-face-detector", action="store_true")
    p.add_argument("--require-face-detector", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def load_class_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [str(x["emotion_name"]) for x in payload.get("emotions", [])]
    if len(names) != 26:
        raise ValueError(f"Expected 26 EMOTIC classes, got {len(names)}")
    return names


def load_pool_samples(args: argparse.Namespace, class_names: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in ("val", "test"):
        ds = EmoticPersonDataset(
            annotation_file=args.annotation_file,
            image_root=args.image_root,
            split=split,
            class_names=class_names,
            transform=None,
            cache_index=True,
        )
        samples.extend(ds.samples)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    return samples


def validate_fdil_alignment(samples: list[dict[str, Any]], cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels, soft, ids = [], [], []
    for split in ("val", "test"):
        with np.load(cache_dir / f"{split}_clip.npz", allow_pickle=False) as z:
            labels.append(np.asarray(z["labels"], dtype=np.float32))
            soft.append(np.asarray(z["soft_labels"], dtype=np.float32))
            ids.extend(np.asarray(z["image_ids"]).astype(str).tolist())
    expected = [str(s["image_id"]) for s in samples]
    if ids[: len(expected)] != expected:
        raise RuntimeError("EMOTIC sample order does not match the existing FDIL cache")
    return np.concatenate(labels)[: len(expected)], np.concatenate(soft)[: len(expected)], expected


def fallback_head(body: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = body
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    cx = (x1 + x2) / 2.0
    side = max(int(min(w, h) * 0.45), 4)
    hx1 = int(round(cx - side / 2))
    return hx1, y1, hx1 + side, min(y2, y1 + side)


def build_face_cache(samples: list[dict[str, Any]], path: Path, disabled: bool, required: bool) -> dict[str, list[int]]:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if all(str(s["image_id"]) in cached for s in samples):
            return cached

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_image[str(sample["image_path"])].append(sample)

    app = None
    if not disabled:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:
            if required:
                raise RuntimeError("InsightFace buffalo_l is required but could not be initialized") from exc
            print(f"[faces] detector unavailable; using deterministic upper-body fallback: {exc}", flush=True)

    result: dict[str, list[int]] = {}
    for image_path, group in tqdm(by_image.items(), desc="Detecting/matching faces"):
        detections: list[tuple[float, float, float, float, float]] = []
        if app is not None:
            import cv2

            bgr = cv2.imread(image_path)
            if bgr is not None:
                for face in app.get(bgr):
                    x1, y1, x2, y2 = np.asarray(face.bbox).tolist()
                    detections.append((x1, y1, x2, y2, float(getattr(face, "det_score", 0.0))))
        for sample in group:
            bx1, by1, bx2, by2 = [int(v) for v in sample["bbox"]]
            candidates = []
            for x1, y1, x2, y2, score in detections:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    candidates.append((score, (int(x1), int(y1), int(x2), int(y2))))
            head = max(candidates, key=lambda x: x[0])[1] if candidates else fallback_head((bx1, by1, bx2, by2))
            result[str(sample["image_id"])] = list(head)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


class PoolImageDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], heads: dict[str, list[int]], clip_preprocess: Any, method: str):
        self.samples = samples
        self.heads = heads
        self.clip_preprocess = clip_preprocess
        self.method = method
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _safe_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return image
        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        body = tuple(int(v) for v in sample["bbox"])
        mask = Image.new("L", image.size, 0)
        mask.paste(255, body)
        result = {
            "context": self.clip_preprocess(image),
            "mask": self.mask_transform(mask)[0],
            "index": idx,
        }
        if self.method in {"all", "cocoer"}:
            head = tuple(int(v) for v in self.heads[str(sample["image_id"])])
            result["body"] = self.clip_preprocess(self._safe_crop(image, body))
            result["head"] = self.clip_preprocess(self._safe_crop(image, head))
        return result


def visual_tokens(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visual = model.visual
    x = visual.conv1(image.to(dtype=visual.conv1.weight.dtype))
    grid = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    cls_out = F.normalize(x[:, 0].float(), dim=-1)
    patches = x[:, 1:].reshape(x.shape[0], grid[0], grid[1], -1).permute(0, 3, 1, 2)
    patches = F.adaptive_avg_pool2d(patches.float(), (7, 7)).flatten(2).transpose(1, 2)
    return cls_out, patches


def mask_aware_image(model: torch.nn.Module, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """EmotionCLIP's extra bbox-mask token, adapted to OpenAI ViT-L/14 weights."""
    visual = model.visual
    x = visual.conv1(image.to(dtype=visual.conv1.weight.dtype))
    gh, gw = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    weights = F.adaptive_avg_pool2d(mask[:, None].float(), (gh, gw)).flatten(2).transpose(1, 2)
    mask_token = (weights.to(x.dtype) * visual.positional_embedding[1:].to(x.dtype)).sum(1, keepdim=True)
    x = torch.cat([x, mask_token], dim=1)
    x = visual.ln_pre(x).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)
    x = visual.ln_post(x[:, 0])
    if visual.proj is not None:
        x = x @ visual.proj
    return F.normalize(x.float(), dim=-1)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(Path(args.description_file))
    samples = load_pool_samples(args, class_names)
    labels, soft, ids = validate_fdil_alignment(samples, Path(args.fdil_cache))
    if len(samples) != 10613 and args.max_samples is None:
        raise RuntimeError(f"Expected the 10,613-person pool, got {len(samples)}")

    heads: dict[str, list[int]] = {}
    if args.method in {"all", "cocoer"}:
        face_path = Path(args.face_cache) if args.face_cache else out / "face_boxes.json"
        heads = build_face_cache(
            samples,
            face_path,
            bool(args.disable_face_detector),
            bool(args.require_face_detector),
        )
    device = torch.device(args.device)
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    text = None
    if args.method in {"all", "cocoer"}:
        prompts = [f"This photo shows {name.lower()} emotion." for name in class_names]
        with torch.inference_mode():
            text = F.normalize(model.encode_text(clip.tokenize(prompts).to(device)).float(), dim=-1)

    ds = PoolImageDataset(samples, heads, preprocess, args.method)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    n = len(ds)
    token_shape = (n, 49, 768)
    arrays = {}
    if args.method in {"all", "cocoer"}:
        arrays.update({
            "context_tokens": np.lib.format.open_memmap(out / "context_tokens.npy", mode="w+", dtype=np.float16, shape=token_shape),
            "body_tokens": np.lib.format.open_memmap(out / "body_tokens.npy", mode="w+", dtype=np.float16, shape=token_shape),
            "head_tokens": np.lib.format.open_memmap(out / "head_tokens.npy", mode="w+", dtype=np.float16, shape=token_shape),
            "vi_logits": np.lib.format.open_memmap(out / "vi_logits.npy", mode="w+", dtype=np.float16, shape=(n, 26)),
        })
    if args.method in {"all", "emotionclip"}:
        arrays["emotionclip_features"] = np.lib.format.open_memmap(
            out / "emotionclip_features.npy", mode="w+", dtype=np.float16, shape=(n, 768)
        )

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch in tqdm(loader, desc="Extracting ViT-L/14 matched features"):
            idx = np.asarray(batch["index"], dtype=np.int64)
            context = batch["context"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            values = {}
            if args.method in {"all", "cocoer"}:
                body = batch["body"].to(device, non_blocking=True)
                head = batch["head"].to(device, non_blocking=True)
                ctx_cls, ctx_tok = visual_tokens(model, context)
                _, body_tok = visual_tokens(model, body)
                _, head_tok = visual_tokens(model, head)
                values.update({
                    "context_tokens": ctx_tok,
                    "body_tokens": body_tok,
                    "head_tokens": head_tok,
                    "vi_logits": model.logit_scale.exp().float() * ctx_cls @ text.T,
                })
            if args.method in {"all", "emotionclip"}:
                values["emotionclip_features"] = mask_aware_image(model, context, mask)
            for key, value in values.items():
                arrays[key][idx] = value.detach().cpu().numpy().astype(np.float16)

    for arr in arrays.values():
        arr.flush()
    np.save(out / "labels.npy", labels)
    np.save(out / "soft_labels.npy", soft)
    (out / "image_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps({"clip_model": args.clip_model, "method": args.method, "persons": n, "classes": class_names, "feature_dtype": "float16"}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved matched features to {out}")


if __name__ == "__main__":
    main()
