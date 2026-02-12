"""
Script to compute factor-to-anchor mapping from pretrained checkpoint.

This script:
1. Loads a pretrained checkpoint
2. Iterates through the full training dataset
3. Collects z_q_split_all (quantized factor vectors)
4. Computes mapping using compute_mapping function
5. Saves mapping result to file

Usage:
    python src/scripts/compute_mapping.py --ckpt_path <path> --output_path <path> [--device cuda]
    
Or with Hydra:
    python src/scripts/compute_mapping.py ckpt_path=<path> output_path=<path>
"""
import argparse
from pathlib import Path
from typing import Optional

import hydra
import rootutils
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Get project root directory and config path
# Note: Hydra's config_path must be relative to current working directory
# Assuming script is run from project root, use "configs"
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH_RELATIVE = "configs"  # Relative path for Hydra (from project root)

from src.models.intentonomy_clip_vit_codebook_module import (
    IntentonomyClipViTCodebookModule,
    clean_state_dict_for_loading,
)


def custom_collate_fn(batch):
    """Custom collate function to properly handle dictionary batches.
    
    This ensures that dictionary batches are properly collated,
    stacking tensors and keeping non-tensor values as lists.
    """
    if len(batch) == 0:
        return {}
    
    # Check if batch items are dictionaries
    if isinstance(batch[0], dict):
        # Collate dictionary items
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            # Stack tensors, keep other types as lists
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result
    else:
        # Fall back to default collate behavior
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)


def collect_z_q_split_all(
    model: LightningModule,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Collect z_q_split_all from all batches in the dataloader.
    
    :param model: The model to use for forward pass.
    :param dataloader: DataLoader for training dataset.
    :param device: Device to run inference on.
    :return: Tensor of shape [N, 5, block_dim] containing all quantized factor vectors.
    """
    model.eval()
    model.to(device)
    
    z_q_split_all_list = []
    
    print("Collecting z_q_split_all from training dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")
            
            # Debug: print batch structure on first batch
            if batch_idx == 0:
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"Batch keys: {batch.keys()}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: type={type(value)}")
                elif isinstance(batch, (list, tuple)):
                    print(f"Batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        if isinstance(item, torch.Tensor):
                            print(f"  batch[{i}]: shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"  batch[{i}]: type={type(item)}")
            
            # Handle both dict and list/tuple batch formats
            if isinstance(batch, dict):
                x = batch["image"].to(device)
            elif isinstance(batch, (list, tuple)):
                # If batch is a list/tuple, check if it's a list of dicts or a list of tensors
                if len(batch) > 0 and isinstance(batch[0], dict):
                    # List of dicts: extract images from each dict and stack them
                    x = torch.stack([item["image"] for item in batch]).to(device)
                else:
                    # List of tensors: find the image tensor (should be 4D: [B, C, H, W] with C=3)
                    x = None
                    for item in batch:
                        if isinstance(item, torch.Tensor):
                            # Image tensor should be 4D with shape [B, 3, H, W] or 3D with [3, H, W]
                            if item.dim() == 4 and item.shape[1] == 3:
                                x = item
                                break
                            elif item.dim() == 3 and item.shape[0] == 3:
                                # Single image, need to add batch dimension
                                x = item.unsqueeze(0)
                                break
                    if x is None:
                        raise ValueError(
                            f"Could not find image tensor in batch. "
                            f"Batch type: {type(batch)}, length: {len(batch)}. "
                            f"Expected a 4D tensor with shape [B, 3, H, W] or 3D tensor with [3, H, W]."
                        )
                    x = x.to(device)
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}. Expected dict, list, or tuple.")
            
            # Validate image shape
            if x.dim() != 4 or x.shape[1] != 3:
                raise ValueError(
                    f"Invalid image shape: {x.shape}. Expected [B, 3, H, W] with 3 RGB channels. "
                    f"Got {x.dim()} dimensions and {x.shape[1] if x.dim() >= 2 else 'N/A'} channels."
                )
            
            # Forward pass with return_vq_info=True to get z_quantized
            _, _, _, z_quantized = model.forward(x, return_vq_info=True)
            # z_quantized shape: [B, 5, block_dim]
            
            z_q_split_all_list.append(z_quantized.cpu())
    
    # Concatenate all batches
    z_q_split_all = torch.cat(z_q_split_all_list, dim=0)  # [N, 5, block_dim]
    print(f"Collected z_q_split_all with shape: {z_q_split_all.shape}")
    
    return z_q_split_all


def compute_and_save_mapping(
    ckpt_path: str,
    output_path: str,
    data_config: Optional[DictConfig] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Compute mapping from pretrained checkpoint and save to file.
    
    :param ckpt_path: Path to pretrained checkpoint.
    :param output_path: Path to save mapping result.
    :param data_config: Optional data module config (if None, will use default).
    :param device: Device to run inference on (if None, will use cuda if available).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load checkpoint manually to handle state_dict cleaning
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Clean state_dict (remove torch.compile and EMA prefixes)
    cleaned_state_dict = clean_state_dict_for_loading(state_dict)
    # cleaned_state_dict = state_dict
    
    # Remove text_anchors and proj_text_anchors from checkpoint
    # These will be regenerated based on current model configuration
    keys_to_remove = []
    for key in cleaned_state_dict.keys():
        if "text_anchors" in key or "proj_text_anchors" in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        print(f"Removing {key} from checkpoint (will be regenerated)")
        del cleaned_state_dict[key]
    
    # Load model from checkpoint config
    # First, try to get model config from checkpoint
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        # Create model with same config as checkpoint
        from hydra import compose, initialize
        with initialize(config_path=CONFIG_PATH_RELATIVE, version_base="1.3"):
            cfg = compose(config_name="train.yaml")
            # Override with checkpoint config if available
            if "net" in hparams:
                # Model config is stored, but we'll use current config
                pass
        
        # Load model using Lightning's load_from_checkpoint but with custom state_dict
        model = IntentonomyClipViTCodebookModule.load_from_checkpoint(
            ckpt_path,
            map_location=device,
            weights_only=False,
            strict=False,  # Allow partial loading
        )
    else:
        # Fallback: load model from current config
        from hydra import compose, initialize
        with initialize(config_path=CONFIG_PATH_RELATIVE, version_base="1.3"):
            cfg = compose(config_name="train.yaml")
            model = instantiate(cfg.model)
    
    # Load cleaned state_dict manually with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
    # Filter out text_anchors related keys from missing_keys (expected)
    missing_keys_filtered = [k for k in missing_keys if "text_anchors" not in k and "proj_text_anchors" not in k]
    unexpected_keys_filtered = [k for k in unexpected_keys if "text_anchors" not in k and "proj_text_anchors" not in k]
    
    if missing_keys_filtered:
        print(f"Warning: Missing keys (excluding text_anchors): {missing_keys_filtered[:10]}")
        if len(missing_keys_filtered) > 10:
            print(f"  (and {len(missing_keys_filtered) - 10} more)")
    
    if unexpected_keys_filtered:
        print(f"Warning: Unexpected keys (excluding text_anchors): {unexpected_keys_filtered[:10]}")
        if len(unexpected_keys_filtered) > 10:
            print(f"  (and {len(unexpected_keys_filtered) - 10} more)")
    
    # Ensure text_anchors are regenerated (they should be generated in __init__, but double-check)
    if not hasattr(model, 'proj_text_anchors') or model.proj_text_anchors is None:
        print("Regenerating text_anchors...")
        model._generate_text_anchors()
    
    print("Model loaded successfully!")
    
    # Instantiate data module
    if data_config is None:
        # Use default data config
        from hydra import compose, initialize
        with initialize(config_path=CONFIG_PATH_RELATIVE, version_base="1.3"):
            cfg = compose(config_name="train.yaml")
            data_config = cfg.data
    
    datamodule: LightningDataModule = instantiate(data_config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    # Get training dataloader
    original_dataloader = datamodule.train_dataloader()
    # Recreate dataloader with custom collate_fn to ensure proper dictionary batching
    from torch.utils.data import DataLoader
    # Preserve sampler if it exists, otherwise use shuffle
    dataloader_kwargs = {
        "dataset": original_dataloader.dataset,
        "batch_size": original_dataloader.batch_size,
        "num_workers": original_dataloader.num_workers,
        "pin_memory": original_dataloader.pin_memory,
        "collate_fn": custom_collate_fn,
    }
    # Handle sampler/shuffle: if sampler exists, use it; otherwise use shuffle
    if original_dataloader.sampler is not None:
        dataloader_kwargs["sampler"] = original_dataloader.sampler
        dataloader_kwargs["shuffle"] = False
    else:
        dataloader_kwargs["shuffle"] = True  # Training dataloader typically shuffles
    
    train_dataloader = DataLoader(**dataloader_kwargs)
    print(f"Training dataset size: {len(train_dataloader.dataset)}")
    print(f"Number of batches: {len(train_dataloader)}")
    
    # Collect z_q_split_all
    z_q_split_all = collect_z_q_split_all(model, train_dataloader, device)
    
    # Get proj_text_anchors
    proj_text_anchors = model.proj_text_anchors  # [5, block_dim]
    print(f"proj_text_anchors shape: {proj_text_anchors.shape}")
    
    # Compute mapping
    print("Computing mapping...")
    mapping, similarity_matrix = model.compute_mapping(z_q_split_all, proj_text_anchors)
    
    print(f"Mapping result: {mapping.tolist()}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix:\n{similarity_matrix}")
    
    # Save mapping result
    mapping_result = {
        "mapping": mapping,
        "similarity_matrix": similarity_matrix,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(mapping_result, output_path)
    print(f"Mapping saved to: {output_path}")


def main_with_args() -> None:
    """Main entry point using command line arguments."""
    parser = argparse.ArgumentParser(description="Compute factor-to-anchor mapping from pretrained checkpoint")
    parser.add_argument("--ckpt_path", "-c", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to save mapping result")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu, default: auto)")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH_RELATIVE, help="Path to Hydra config directory (relative to current working directory)")
    parser.add_argument("--config_name", type=str, default="train.yaml", help="Hydra config name")
    
    args = parser.parse_args()
    
    # Load data config from Hydra
    from hydra import compose, initialize
    with initialize(config_path=args.config_path, version_base="1.3"):
        cfg = compose(config_name=args.config_name)
        data_config = cfg.data
    
    device = torch.device(args.device) if args.device else None
    
    compute_and_save_mapping(
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        data_config=data_config,
        device=device,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main_with_hydra(cfg: DictConfig) -> None:
    """Main entry point using Hydra config."""
    ckpt_path = cfg.get("ckpt_path")
    output_path = cfg.get("output_path", "mapping_result.pth")
    device_str = cfg.get("device", None)
    
    if not ckpt_path:
        raise ValueError("ckpt_path must be provided in config")
    
    device = torch.device(device_str) if device_str else None
    
    compute_and_save_mapping(
        ckpt_path=ckpt_path,
        output_path=output_path,
        data_config=cfg.data,
        device=device,
    )


if __name__ == "__main__":
    # Try to use command line arguments first, fall back to Hydra if no args provided
    import sys
    if len(sys.argv) > 1 and any(arg.startswith("--ckpt_path") or arg.startswith("-c") for arg in sys.argv):
        main_with_args()
    else:
        main_with_hydra()

