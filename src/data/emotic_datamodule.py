from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.components.cutout import CutoutPIL

Image.MAX_IMAGE_PIXELS = 200000000
CACHE_VERSION = 2


def _to_person_list(person_field: Any) -> List[Any]:
    if isinstance(person_field, np.ndarray):
        return list(np.ravel(person_field))
    return [person_field]


class EmoticPersonDataset(Dataset):
    """Person-level Emotic dataset with bbox crops and split-aware label aggregation."""

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        split: str,
        class_names: Sequence[str],
        transform: Optional[transforms.Compose] = None,
        min_crop_size: int = 4,
        cache_index: bool = True,
    ) -> None:
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.split = str(split)
        self.class_names = [str(name) for name in class_names]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.transform = transform
        self.min_crop_size = int(min_crop_size)
        self.cache_index = bool(cache_index)

        cache_path = self.annotation_file.parent / f".emotic_{self.split}_person_cache.pt"
        cache_is_fresh = (
            cache_path.exists()
            and cache_path.stat().st_mtime >= self.annotation_file.stat().st_mtime
        )
        if self.cache_index and cache_is_fresh:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                payload.get("class_names") == self.class_names
                and int(payload.get("cache_version", -1)) == CACHE_VERSION
            ):
                self.samples = payload["samples"]
                return

        mat = loadmat(self.annotation_file, squeeze_me=True, struct_as_record=False)
        if self.split not in mat:
            raise KeyError(f"Split '{self.split}' not found in {self.annotation_file}")
        entries = mat[self.split]
        self.samples = self._build_samples(entries)
        if self.cache_index:
            torch.save(
                {
                    "cache_version": CACHE_VERSION,
                    "class_names": self.class_names,
                    "samples": self.samples,
                },
                cache_path,
            )

    def _extract_final_labels(self, person: Any) -> List[str]:
        if self.split in {"val", "test"} and hasattr(person, "combined_categories"):
            raw = getattr(person, "combined_categories")
        else:
            ann = getattr(person, "annotations_categories", None)
            raw = getattr(ann, "categories", [])

        values = np.ravel(raw) if isinstance(raw, np.ndarray) else [raw]
        labels: List[str] = []
        for value in values:
            label = str(value).strip()
            if not label or label == "[]":
                continue
            labels.append(label)
        return labels

    def _extract_soft_labels(self, person: Any, hard_labels: Sequence[str]) -> torch.Tensor:
        soft = np.zeros(self.num_classes, dtype=np.float32)
        ann = getattr(person, "annotations_categories", None)

        if isinstance(ann, np.ndarray):
            annotators = list(np.ravel(ann))
            for item in annotators:
                raw = getattr(item, "categories", [])
                values = np.ravel(raw) if isinstance(raw, np.ndarray) else [raw]
                for value in values:
                    label = str(value).strip()
                    if label in self.class_to_idx:
                        soft[self.class_to_idx[label]] += 1.0
            if annotators:
                soft /= float(len(annotators))
        else:
            for label in hard_labels:
                if label in self.class_to_idx:
                    soft[self.class_to_idx[label]] = 1.0

        return soft

    def _resolve_image_path(self, entry: Any) -> Path:
        folder = str(getattr(entry, "folder"))
        filename = str(getattr(entry, "filename"))
        return self.image_root / folder / filename

    def _sanitize_bbox(self, bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        coords = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if coords.size != 4:
            return None

        x1, y1, x2, y2 = coords.tolist()
        x1 = int(np.floor(max(0.0, min(x1, width - 1))))
        y1 = int(np.floor(max(0.0, min(y1, height - 1))))
        x2 = int(np.ceil(max(0.0, min(x2, width))))
        y2 = int(np.ceil(max(0.0, min(y2, height))))

        if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
            return None
        return x1, y1, x2, y2

    def _build_samples(self, entries: Sequence[Any]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for image_index, entry in enumerate(entries):
            image_path = self._resolve_image_path(entry)

            width = getattr(getattr(entry, "image_size", None), "n_col", None)
            height = getattr(getattr(entry, "image_size", None), "n_row", None)
            if width is None or height is None:
                with Image.open(image_path) as img:
                    width, height = img.size

            people = _to_person_list(getattr(entry, "person"))
            for person_index, person in enumerate(people):
                labels = self._extract_final_labels(person)
                if not labels:
                    continue

                bbox = self._sanitize_bbox(getattr(person, "body_bbox", None), int(width), int(height))
                if bbox is None:
                    continue

                targets = np.zeros(self.num_classes, dtype=np.float32)
                for label in labels:
                    idx = self.class_to_idx.get(label)
                    if idx is not None:
                        targets[idx] = 1.0

                soft_labels = self._extract_soft_labels(person, labels)
                positive_mask = targets > 0
                agreement = (
                    float(soft_labels[positive_mask].mean())
                    if bool(positive_mask.any())
                    else 0.0
                )

                samples.append(
                    {
                        "image_path": image_path,
                        "bbox": bbox,
                        "labels": targets,
                        "soft_labels": soft_labels,
                        "agreement": agreement,
                        "image_id": f"{self.split}:{image_index}:{person_index}",
                        "label_names": labels,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        full_image = Image.open(sample["image_path"]).convert("RGB")
        crop_image = full_image.crop(sample["bbox"])
        if self.transform:
            image = self.transform(crop_image)
            image_full = self.transform(full_image)
        else:
            image = crop_image
            image_full = full_image

        return {
            "image": image,
            "image_full": image_full,
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
            "soft_labels": torch.tensor(sample["soft_labels"], dtype=torch.float32),
            "agreement": torch.tensor(sample["agreement"], dtype=torch.float32),
            "bbox": torch.tensor(sample["bbox"], dtype=torch.int32),
            "image_id": sample["image_id"],
            "image_path": str(sample["image_path"]),
        }


class EmoticDataModule(LightningDataModule):
    """Lightning datamodule for person-level Emotic emotion recognition."""

    def __init__(
        self,
        data_dir: str = "data/",
        annotation_file: str = "Annotations/Annotations.mat",
        image_root: str = "emotic",
        description_file: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
        min_crop_size: int = 4,
        cache_index: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                CutoutPIL(cutout_factor=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.val_test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size
        self.class_names = self._load_class_names()
        self._num_classes = len(self.class_names)

    def _load_class_names(self) -> List[str]:
        description_file = self.hparams.description_file
        if description_file:
            path = Path(description_file)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                emotions = payload.get("emotions", [])
                names = [str(item["emotion_name"]).strip() for item in emotions if item.get("emotion_name")]
                if names:
                    return names

        mat = loadmat(Path(self.hparams.annotation_file), squeeze_me=True, struct_as_record=False)
        all_names = set()
        for split in ("train", "val", "test"):
            for entry in mat[split]:
                for person in _to_person_list(getattr(entry, "person")):
                    if split in {"val", "test"} and hasattr(person, "combined_categories"):
                        raw = getattr(person, "combined_categories")
                    else:
                        ann = getattr(person, "annotations_categories", None)
                        raw = getattr(ann, "categories", [])
                    values = np.ravel(raw) if isinstance(raw, np.ndarray) else [raw]
                    for value in values:
                        label = str(value).strip()
                        if label and label != "[]":
                            all_names.add(label)
        return sorted(all_names)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def prepare_data(self) -> None:
        annotation_file = Path(self.hparams.annotation_file)
        image_root = Path(self.hparams.image_root)
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root not found: {image_root}")

    def _build_dataset(self, split: str, transform: transforms.Compose) -> EmoticPersonDataset:
        return EmoticPersonDataset(
            annotation_file=self.hparams.annotation_file,
            image_root=self.hparams.image_root,
            split=split,
            class_names=self.class_names,
            transform=transform,
            min_crop_size=self.hparams.min_crop_size,
            cache_index=self.hparams.cache_index,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if stage == "fit" or stage is None:
            if self.data_train is None:
                self.data_train = self._build_dataset("train", self.train_transform)
            if self.data_val is None:
                self.data_val = self._build_dataset("val", self.val_test_transform)

        if stage == "validate" and self.data_val is None:
            self.data_val = self._build_dataset("val", self.val_test_transform)

        if stage == "test" or stage is None:
            if self.data_test is None:
                self.data_test = self._build_dataset("test", self.val_test_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
