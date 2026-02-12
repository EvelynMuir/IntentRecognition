import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rootutils
import torch
from lightning import seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.intentonomy_datamodule import IntentonomyDataModule
from src.models.intentonomy_resnet101_mcc_module import IntentonomyResNet101Module


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor using ImageNet mean/std.

    Args:
        tensor: (3, H, W), normalized.
    Returns:
        (3, H, W) in [0, 1] range.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    img = tensor * std + mean
    return img.clamp(0.0, 1.0)


def get_patch_grid(num_patches: int) -> Tuple[int, int]:
    """Infer (H, W) grid from number of patches."""
    h = int(num_patches**0.5)
    if h * h == num_patches:
        w = h
    else:
        # Fallback: assume square-ish grid
        w = num_patches // h
    return h, w


def visualize_topk_patches_on_image(
    img: torch.Tensor,
    topk_indices: torch.Tensor,
    patch_scores: torch.Tensor,
    title: str = "",
) -> None:
    """Visualize top-k patches on a single image.

    Args:
        img: (3, H, W) tensor in [0, 1].
        topk_indices: (K,) patch indices.
        patch_scores: (N,) all patch scores.
        title: plot title.
    """
    img_np = img.permute(1, 2, 0).cpu().numpy()
    h_img, w_img, _ = img_np.shape

    num_patches = patch_scores.numel()
    grid_h, grid_w = get_patch_grid(num_patches)
    patch_h = h_img / grid_h
    patch_w = w_img / grid_w

    # Prepare heatmap from scores
    heatmap = patch_scores.view(grid_h, grid_w).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: original image with top-k rectangles
    ax = axes[0]
    ax.imshow(img_np)
    ax.axis("off")
    ax.set_title(title)

    for idx in topk_indices.cpu().tolist():
        row = idx // grid_w
        col = idx % grid_w
        x1 = col * patch_w
        y1 = row * patch_h
        rect = patches.Rectangle(
            (x1, y1),
            patch_w,
            patch_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Right: heatmap of all patch scores
    ax_hm = axes[1]
    ax_hm.imshow(heatmap, cmap="jet")
    ax_hm.axis("off")
    ax_hm.set_title("Patch scores")

    plt.tight_layout()
    plt.show()


def run_visualization(
    ckpt_path: str,
    num_samples: int = 20,
    image_size: int = 224,
    annotation_dir: str = None,
    image_dir: str = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data module (only test set is needed)
    # Use default paths if not provided
    if annotation_dir is None:
        # Try to infer from common locations
        import os
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        default_annotation_dir = project_root / ".." / "Intentonomy" / "data" / "annotation"
        default_image_dir = project_root / ".." / "Intentonomy" / "data" / "images" / "low"
        annotation_dir = str(default_annotation_dir.resolve())
        image_dir = str(default_image_dir.resolve()) if image_dir is None else image_dir
    else:
        if image_dir is None:
            # If annotation_dir is provided but image_dir is not, infer from annotation_dir
            from pathlib import Path
            annotation_path = Path(annotation_dir)
            image_dir = str(annotation_path.parent.parent / "images" / "low")

    dm = IntentonomyDataModule(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        image_size=image_size,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # Load model from checkpoint
    model = IntentonomyResNet101Module.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()

    seed_everything(42)

    collected = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)

            logits, selection_info = model(images, return_selection_info=True)
            patch_scores = selection_info["patch_scores"]  # (B, N, 1)
            topk_indices = selection_info["topk_indices"]  # (B, K, 1)

            bsz = images.shape[0]
            for i in range(bsz):
                if collected >= num_samples:
                    return

                img = denormalize_image(images[i].detach().cpu())

                scores_i = patch_scores[i].squeeze(-1)  # (N,)
                topk_i = topk_indices[i].squeeze(-1)  # (K,)

                visualize_topk_patches_on_image(
                    img,
                    topk_i,
                    scores_i,
                    title=f"Sample {collected + 1}",
                )

                collected += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize scorer-selected top-k patches on test images."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of test images to visualize.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size used for resizing (must match training).",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Directory containing annotation JSON files. If not provided, will try to infer from ../Intentonomy/data/annotation",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images. If not provided, will try to infer from ../Intentonomy/data/images/low",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(
        ckpt_path=args.ckpt_path,
        num_samples=args.num_samples,
        image_size=args.image_size,
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
    )


