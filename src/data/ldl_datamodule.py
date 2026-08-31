from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import lmdb
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.components.caffe_datum import datum_to_rgb, parse_caffe_datum


class LDLImageDataset(Dataset):
    """Plain-image view of a dataset produced by ``scripts/prepare_ldl.py``."""

    def __init__(self, data_dir: str, split: str, transform: Optional[Any] = None) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.transform = transform
        metadata = np.load(self.data_dir / "metadata.npz", allow_pickle=False)
        split_values = np.asarray(metadata["fdil_split"]).astype(str)
        self.indices = np.flatnonzero(split_values == self.split)
        self.image_paths = np.asarray(metadata["image_paths"]).astype(str)[self.indices]
        self.lmdb_keys = np.asarray(metadata["lmdb_keys"]).astype(str)[self.indices]
        self.official_splits = np.asarray(metadata["official_split"]).astype(str)[self.indices]
        self.image_ids = np.asarray(metadata["image_ids"]).astype(str)[self.indices]
        self.soft_labels = np.asarray(metadata["distributions"], dtype=np.float32)[self.indices]
        self.hard_labels = np.asarray(metadata["hard_labels"], dtype=np.int64)[self.indices]
        self.agreement = np.asarray(metadata["agreement"], dtype=np.float32)[self.indices]
        self.entropy = np.asarray(metadata["entropy"], dtype=np.float32)[self.indices]
        self.class_names = np.asarray(metadata["class_names"]).astype(str).tolist()
        self._lmdb_envs: dict[str, Any] = {}

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        relative_path = self.image_paths[idx]
        image_path = self.data_dir / relative_path if relative_path else None
        if image_path is not None and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image_source = str(image_path)
        else:
            official_split = self.official_splits[idx]
            env = self._get_lmdb(official_split)
            key = self.lmdb_keys[idx].encode("utf-8")
            with env.begin(buffers=False) as transaction:
                payload = transaction.get(key)
            if payload is None:
                raise KeyError(f"LMDB key not found: {self.lmdb_keys[idx]}")
            image = datum_to_rgb(parse_caffe_datum(payload))
            image_source = f"lmdb://{official_split}/{self.lmdb_keys[idx]}"
        transformed = self.transform(image) if self.transform else image
        one_hot = np.zeros(len(self.class_names), dtype=np.float32)
        one_hot[self.hard_labels[idx]] = 1.0
        return {
            "image": transformed,
            "image_full": transformed,
            "labels": torch.from_numpy(one_hot),
            "soft_labels": torch.from_numpy(self.soft_labels[idx]),
            "agreement": torch.tensor(float(self.agreement[idx]), dtype=torch.float32),
            "entropy": torch.tensor(float(self.entropy[idx]), dtype=torch.float32),
            "hard_label": torch.tensor(int(self.hard_labels[idx]), dtype=torch.long),
            "image_id": self.image_ids[idx],
            "image_path": image_source,
        }

    def _get_lmdb(self, official_split: str) -> Any:
        env = self._lmdb_envs.get(official_split)
        if env is None:
            dataset_name = self.data_dir.name
            lmdb_name = (
                f"{official_split}_{dataset_name}_lmdb"
                if dataset_name == "emotion6" else f"{official_split}_{dataset_name}_split1_lmdb"
            )
            path = self.data_dir.parents[1] / "data" / lmdb_name
            env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=64)
            self._lmdb_envs[official_split] = env
        return env

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_lmdb_envs"] = {}
        return state


class LDLDataModule(LightningDataModule):
    """Flickr-LDL/Twitter-LDL data module compatible with FDIL CLIP caching."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.data_train: Optional[LDLImageDataset] = None
        self.data_val: Optional[LDLImageDataset] = None
        self.data_test: Optional[LDLImageDataset] = None
        self.batch_size_per_device = int(batch_size)
        metadata_path = Path(data_dir) / "metadata.npz"
        self.class_names = (
            np.asarray(np.load(metadata_path, allow_pickle=False)["class_names"]).astype(str).tolist()
            if metadata_path.exists()
            else []
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def prepare_data(self) -> None:
        metadata = Path(self.hparams.data_dir) / "metadata.npz"
        if not metadata.exists():
            raise FileNotFoundError(f"LDL metadata not found: {metadata}; run scripts/prepare_ldl.py first")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError("Batch size must be divisible by the number of devices")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if stage in (None, "fit"):
            self.data_train = self.data_train or LDLImageDataset(self.hparams.data_dir, "train", self.train_transform)
            self.data_val = self.data_val or LDLImageDataset(self.hparams.data_dir, "val", self.eval_transform)
        if stage == "validate":
            self.data_val = self.data_val or LDLImageDataset(self.hparams.data_dir, "val", self.eval_transform)
        if stage in (None, "test"):
            self.data_test = self.data_test or LDLImageDataset(self.hparams.data_dir, "test", self.eval_transform)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_train, True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_val, False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_test, False)
