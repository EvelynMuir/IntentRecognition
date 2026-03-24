from pathlib import Path

import pytest
import torch

from src.data.emotic_datamodule import EmoticDataModule


@pytest.mark.skipif(
    not Path("/home/evelynmuir/lambda/projects/IntentRecognition/Emotic/Annotations/Annotations.mat").exists(),
    reason="Local Emotic dataset not available.",
)
def test_emotic_datamodule_person_level() -> None:
    dm = EmoticDataModule(
        annotation_file="/home/evelynmuir/lambda/projects/IntentRecognition/Emotic/Annotations/Annotations.mat",
        image_root="/home/evelynmuir/lambda/projects/IntentRecognition/Emotic/emotic",
        description_file="/home/evelynmuir/lambda/projects/IntentRecognition/Emotic/emotion_description_gemini.json",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        image_size=224,
    )

    dm.prepare_data()
    dm.setup()

    assert dm.num_classes == 26
    assert dm.data_train and dm.data_val and dm.data_test
    assert len(dm.data_train) > 8_000
    assert len(dm.data_val) > 1_000
    assert len(dm.data_test) > 1_000

    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["image_full"].shape == (4, 3, 224, 224)
    assert batch["labels"].shape == (4, 26)
    assert batch["soft_labels"].shape == (4, 26)
    assert batch["agreement"].shape == (4,)
    assert batch["image"].dtype == torch.float32
    assert torch.all(batch["labels"] >= 0)
    assert torch.all(batch["soft_labels"] >= 0)
