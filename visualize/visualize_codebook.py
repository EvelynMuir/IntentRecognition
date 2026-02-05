import argparse
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import rootutils
import torch
from lightning import seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.intentonomy_datamodule import IntentonomyDataModule
from src.models.intentonomy_clip_vit_codebook_module import IntentonomyClipViTCodebookModule


def clean_state_dict_for_loading(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理state_dict，移除torch.compile产生的_orig_mod前缀和EMA相关前缀。
    
    :param state_dict: 原始state_dict
    :return: 清理后的state_dict
    """
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        new_key = k
        
        # 移除 ema_model.module. 前缀（如果存在）
        if new_key.startswith("ema_model.module."):
            continue
        
        # 移除 net._orig_mod. 前缀（torch.compile产生）
        if new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod."):]
        
        new_state_dict[new_key] = v
    
    return new_state_dict


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


def show_codebook_neighbors(images, code_indices, factor_id, code_id, topk=20):
    """
    Visualize images that have a specific code for a specific factor.
    
    Args:
        images: (N, C, H, W) tensor of images
        code_indices: (N, K) tensor of code indices for each image
        factor_id: int, the factor index to filter by
        code_id: int, the code index to filter by
        topk: int, maximum number of images to show (default 20)
    """
    idx = (code_indices[:, factor_id] == code_id).nonzero().squeeze()

    if len(idx) == 0:
        print("No samples for this code.")
        return

    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    
    idx = idx[:topk]

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(idx):
            img_idx = idx[i].item()
            img = denormalize_image(images[img_idx])
            img_np = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
        ax.axis("off")

    plt.suptitle(f"Factor {factor_id} - Code {code_id} ({len(idx)} images)")
    plt.tight_layout()
    plt.show()


def run_codebook_visualization(
    ckpt_path: str,
    factor_id: int,
    code_id: int,
    topk: int = 20,
    image_size: int = 224,
    annotation_dir: str = None,
    image_dir: str = None,
) -> None:
    """Run codebook visualization for a specific factor and code.
    
    Args:
        ckpt_path: Path to Lightning checkpoint (.ckpt).
        factor_id: Factor index to visualize.
        code_id: Code index to visualize.
        topk: Maximum number of images to show (default 20).
        image_size: Image size used for resizing (must match training).
        annotation_dir: Directory containing annotation JSON files.
        image_dir: Directory containing images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data module (only test set is needed)
    # Use default paths if not provided
    if annotation_dir is None:
        # Try to infer from common locations
        project_root = Path(__file__).parent.parent.parent
        default_annotation_dir = project_root  / "Intentonomy" / "data" / "annotation"
        default_image_dir = project_root / "Intentonomy" / "data" / "images" / "low"
        annotation_dir = str(default_annotation_dir.resolve())
        image_dir = str(default_image_dir.resolve()) if image_dir is None else image_dir
    else:
        if image_dir is None:
            # If annotation_dir is provided but image_dir is not, infer from annotation_dir
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
    # 手动加载checkpoint并清理state_dict，以处理torch.compile产生的_orig_mod前缀
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 清理state_dict，移除_orig_mod前缀
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
        cleaned_state_dict = clean_state_dict_for_loading(original_state_dict)
        checkpoint["state_dict"] = cleaned_state_dict
    
    # 将清理后的checkpoint保存到临时文件，然后使用load_from_checkpoint加载
    # 这样可以避免load_from_checkpoint在加载时因为key不匹配而报错
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp_file:
        tmp_ckpt_path = tmp_file.name
        torch.save(checkpoint, tmp_ckpt_path)
    
    try:
        # 使用清理后的临时checkpoint文件加载模型
        model = IntentonomyClipViTCodebookModule.load_from_checkpoint(
            tmp_ckpt_path,
            map_location=device,
            weights_only=False
        )
    finally:
        # 清理临时文件
        Path(tmp_ckpt_path).unlink(missing_ok=True)
    
    model.to(device)
    model.eval()

    seed_everything(42)

    # Collect all images and code indices
    all_images = []
    all_code_indices = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            
            # Get code indices for this batch
            code_indices = model.get_code_indices(images)  # [B, K]
            
            # Store images and code indices
            all_images.append(images.cpu())
            all_code_indices.append(code_indices.cpu())

    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)  # [N, C, H, W]
    all_code_indices = torch.cat(all_code_indices, dim=0)  # [N, K]

    print(f"Total images collected: {len(all_images)}")
    print(f"Code indices shape: {all_code_indices.shape}")
    print(f"Factor {factor_id} code range: [{all_code_indices[:, factor_id].min().item()}, {all_code_indices[:, factor_id].max().item()}]")

    # Visualize
    show_codebook_neighbors(all_images, all_code_indices, factor_id, code_id, topk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize images for a specific factor and code in CLIP ViT Codebook model."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--factor_id",
        type=int,
        required=True,
        help="Factor index to visualize (0 to k_semantic_blocks-1).",
    )
    parser.add_argument(
        "--code_id",
        type=int,
        required=True,
        help="Code index to visualize (0 to codebook_size-1).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Maximum number of images to show (default 20).",
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
    run_codebook_visualization(
        ckpt_path=args.ckpt_path,
        factor_id=args.factor_id,
        code_id=args.code_id,
        topk=args.topk,
        image_size=args.image_size,
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
    )

