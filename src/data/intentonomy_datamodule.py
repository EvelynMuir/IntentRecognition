from typing import Any, Dict, Optional

import json
import os
from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.components.cutout import CutoutPIL

Image.MAX_IMAGE_PIXELS = 200000000


class IntentonomyDataset(Dataset):
    """Dataset for Intentonomy multi-label classification."""

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize Intentonomy dataset.

        :param annotation_file: Path to JSON annotation file.
        :param image_dir: Directory containing images.
        :param transform: Optional transform to be applied on images.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load annotations
        with open(annotation_file, "r") as f:
            data = json.load(f)

        # Build category mapping
        if "id" in data["categories"][0]:
            self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        else:
            self.categories = {cat["category_id"]: cat["name"] for cat in data["categories"]}

        self.num_classes = len(self.categories)

        # Check if annotations use softprob (for training) or category_ids (for val/test)
        # Use the first annotation to determine the format
        use_softprob = False
        if data["annotations"] and "category_ids_softprob" in data["annotations"][0]:
            use_softprob = True

        # Build image_id to annotation mapping
        self.annotations = {}
        self.use_softprob = use_softprob
        for ann in data["annotations"]:
            if "image_id" in ann:
                image_id = ann["image_id"]
            else:
                image_id = ann["image_category_id"]

            if use_softprob:
                # Store softprob directly as a list
                self.annotations[image_id] = ann["category_ids_softprob"]
            elif "category_ids" in ann:
                # Store category_ids list
                self.annotations[image_id] = ann["category_ids"]
            else:
                self.annotations[image_id] = ann["category_category_ids"]

        # Build image list
        self.images = []
        for img_info in data["images"]:
            image_id = img_info["id"]
            filename = img_info["filename"]
            # Remove 'low/' prefix if present
            if filename.startswith("low/"):
                filename = filename[4:]
            
            image_path = self.image_dir / filename
            if image_path.exists() and image_id in self.annotations:
                self.images.append((image_id, image_path))

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset.

        :param idx: Index of the item.
        :return: Dictionary containing image and labels.
        """
        image_id, image_path = self.images[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels (multi-label)
        annotation_data = self.annotations[image_id]
        if self.use_softprob:
            # Use softprob directly (already a probability vector)
            labels = torch.tensor(annotation_data, dtype=torch.float32)
        else:
            # Convert category_ids to one-hot encoding
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[annotation_data] = 1.0

        return {"image": image, "labels": labels, "image_id": image_id}


class IntentonomyDataModule(LightningDataModule):
    """`LightningDataModule` for the Intentonomy dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).

        def setup(self, stage):
        # Things to do on every process in DDP.

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader
    ```
    """

    def __init__(
        self,
        data_dir: str = "data/",
        annotation_dir: str = "data/annotation/",
        image_dir: str = "data/images/low",
        train_annotation: str = "intentonomy_train2020.json",
        val_annotation: str = "intentonomy_val2020.json",
        test_annotation: str = "intentonomy_test2020.json",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
    ) -> None:
        """Initialize a `IntentonomyDataModule`.

        :param data_dir: Base data directory.
        :param annotation_dir: Directory containing annotation JSON files.
        :param image_dir: Directory containing images.
        :param train_annotation: Training annotation filename.
        :param val_annotation: Validation annotation filename.
        :param test_annotation: Test annotation filename.
        :param batch_size: Batch size.
        :param num_workers: Number of workers for data loading.
        :param pin_memory: Whether to pin memory.
        :param image_size: Image size for resizing.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Data transformations
        # Training: Use RandAugment and Cutout
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
        self._num_classes: Optional[int] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of Intentonomy classes.
        """
        if self._num_classes is None:
            # Load one annotation file to get num_classes
            annotation_file = os.path.join(
                self.hparams.annotation_dir, self.hparams.train_annotation
            )
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    data = json.load(f)
                self._num_classes = len(data["categories"])
            else:
                # Default value if file doesn't exist yet
                self._num_classes = 28
        return self._num_classes

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within.
        """
        # Check if annotation files exist
        annotation_dir = Path(self.hparams.annotation_dir)
        for ann_file in [
            self.hparams.train_annotation,
            self.hparams.val_annotation,
            self.hparams.test_annotation,
        ]:
            if not (annotation_dir / ann_file).exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_dir / ann_file}")

        # Check if image directory exists
        image_dir = Path(self.hparams.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets only if not loaded already
        if stage == "fit" or stage is None:
            if self.data_train is None:
                train_ann_file = os.path.join(
                    self.hparams.annotation_dir, self.hparams.train_annotation
                )
                self.data_train = IntentonomyDataset(
                    annotation_file=train_ann_file,
                    image_dir=self.hparams.image_dir,
                    transform=self.train_transform,
                )
                # Get num_classes from dataset
                self._num_classes = self.data_train.num_classes

            if self.data_val is None:
                val_ann_file = os.path.join(
                    self.hparams.annotation_dir, self.hparams.val_annotation
                )
                self.data_val = IntentonomyDataset(
                    annotation_file=val_ann_file,
                    image_dir=self.hparams.image_dir,
                    transform=self.val_test_transform,
                )

        if stage == "test" or stage is None:
            if self.data_test is None:
                test_ann_file = os.path.join(
                    self.hparams.annotation_dir, self.hparams.test_annotation
                )
                self.data_test = IntentonomyDataset(
                    annotation_file=test_ann_file,
                    image_dir=self.hparams.image_dir,
                    transform=self.val_test_transform,
                )
                # Get num_classes from dataset if not set
                if self._num_classes is None:
                    self._num_classes = self.data_test.num_classes

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

