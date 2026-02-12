"""
收集 z_q_split_all 并计算 factor 到 anchor 的映射关系。

使用训练好的 Stage1 模型在整个训练集上运行，收集所有样本的 z_q_split_all，
然后计算 mapping 并保存。
"""
import argparse
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.intentonomy_clip_vit_codebook_module import (
    IntentonomyClipViTCodebookModule,
)


def collect_z_q_split_all(
    ckpt_path: str,
    output_dir: str = ".",
    batch_size: int = 256,
    num_workers: int = 4,
    annotation_dir: str = None,
    image_dir: str = None,
    image_size: int = 224,
) -> None:
    """收集 z_q_split_all 并计算 mapping。
    
    Args:
        ckpt_path: Stage1 checkpoint 路径
        output_dir: 输出目录
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        annotation_dir: 标注文件目录
        image_dir: 图像目录
        image_size: 图像大小
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载配置（使用默认配置）
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    
    # 如果已经初始化，先清理
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=[
            "data.batch_size=" + str(batch_size),
            "data.num_workers=" + str(num_workers),
            "model=intentonomy_clip_vit_codebook",
        ])
    
    # 如果提供了 annotation_dir 和 image_dir，覆盖配置
    if annotation_dir is not None:
        cfg.data.annotation_dir = annotation_dir
    if image_dir is not None:
        cfg.data.image_dir = image_dir
    if image_size is not None:
        cfg.data.image_size = image_size
    
    # 实例化数据模块
    from hydra.utils import instantiate
    datamodule: LightningDataModule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    
    print(f"Training set size: {len(datamodule.data_train)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # 加载模型
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 清理 state_dict（使用模块中的函数）
    def clean_state_dict_for_loading(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """清理state_dict，移除torch.compile产生的_orig_mod前缀和EMA相关前缀。"""
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
    
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
        cleaned_state_dict = clean_state_dict_for_loading(original_state_dict)
        checkpoint["state_dict"] = cleaned_state_dict
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp_file:
        tmp_ckpt_path = tmp_file.name
        torch.save(checkpoint, tmp_ckpt_path)
    
    try:
        model = IntentonomyClipViTCodebookModule.load_from_checkpoint(
            tmp_ckpt_path,
            map_location=device,
            weights_only=False
        )
    finally:
        Path(tmp_ckpt_path).unlink(missing_ok=True)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # 收集 z_q_split_all
    print("Collecting z_q_split_all from training set...")
    z_q_split_list = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Processing batches"):
            images = batch["image"].to(device)
            
            # Forward pass with return_vq_info=True
            _, _, _, z_quantized = model.forward(images, return_vq_info=True)
            # z_quantized shape: [B, 5, block_dim]
            
            z_q_split_list.append(z_quantized.cpu())
    
    # 拼接所有 batch
    z_q_split_all = torch.cat(z_q_split_list, dim=0)  # [N, 5, block_dim]
    print(f"Collected z_q_split_all shape: {z_q_split_all.shape}")
    
    # 获取 proj_text_anchors
    proj_text_anchors = model.proj_text_anchors.cpu()  # [5, block_dim]
    print(f"proj_text_anchors shape: {proj_text_anchors.shape}")
    
    # 计算 mapping
    print("Computing mapping...")
    mapping, similarity_matrix = IntentonomyClipViTCodebookModule.compute_mapping(
        z_q_split_all, proj_text_anchors
    )
    print(f"Mapping: {mapping}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix:\n{similarity_matrix}")
    
    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 z_q_split_all 和 proj_text_anchors
    z_q_split_path = output_dir / "z_q_split_all.pth"
    torch.save({
        "z_q_split_all": z_q_split_all,
        "proj_text_anchors": proj_text_anchors,
    }, z_q_split_path)
    print(f"Saved z_q_split_all to: {z_q_split_path}")
    
    # 保存 mapping 结果
    mapping_path = output_dir / "mapping_result.pth"
    torch.save({
        "mapping": mapping,
        "similarity_matrix": similarity_matrix,
    }, mapping_path)
    print(f"Saved mapping result to: {mapping_path}")
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="收集 z_q_split_all 并计算 mapping")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train/runs/2026-02-05_13-47-02/checkpoints/epoch_046.ckpt",
        help="Stage1 checkpoint 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="输出目录",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="批次大小",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器工作进程数",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="标注文件目录（可选，使用配置默认值）",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="图像目录（可选，使用配置默认值）",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="图像大小",
    )
    
    args = parser.parse_args()
    
    collect_z_q_split_all(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()

