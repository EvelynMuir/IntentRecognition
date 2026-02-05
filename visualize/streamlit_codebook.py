import json
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import rootutils
import streamlit as st
import torch
from lightning import seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.intentonomy_datamodule import IntentonomyDataModule
from src.models.intentonomy_clip_vit_codebook_module import IntentonomyClipViTCodebookModule

# 导入 factor_analysis 模块（同目录下的文件）
import sys
from pathlib import Path
visualize_dir = Path(__file__).parent
if str(visualize_dir) not in sys.path:
    sys.path.insert(0, str(visualize_dir))
from factor_analysis import factor_drop_test, get_quantized_features

# 尝试导入MultiStream模块（如果存在）
try:
    from src.models.intentonomy_clip_vit_multistream_module import IntentonomyClipViTMultiStreamModule
    HAS_MULTISTREAM = True
except ImportError:
    HAS_MULTISTREAM = False
    IntentonomyClipViTMultiStreamModule = None


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


@st.cache_resource
def load_model_and_data(ckpt_path: str, annotation_dir: str, image_dir: str, image_size: int = 224):
    """加载模型和数据，使用缓存避免重复加载。
    
    Args:
        ckpt_path: 模型checkpoint路径
        annotation_dir: 标注文件目录
        image_dir: 图像目录
        image_size: 图像尺寸
    
    Returns:
        model: 加载的模型
        test_loader: 测试数据加载器
        device: 设备
        model_type: 模型类型（"codebook" 或 "multistream"）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据模块
    dm = IntentonomyDataModule(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        image_size=image_size,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    # 加载模型
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
        cleaned_state_dict = clean_state_dict_for_loading(original_state_dict)
        checkpoint["state_dict"] = cleaned_state_dict
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp_file:
        tmp_ckpt_path = tmp_file.name
        torch.save(checkpoint, tmp_ckpt_path)
    
    model = None
    model_type = "codebook"
    
    try:
        # 尝试加载MultiStream模型
        if HAS_MULTISTREAM:
            try:
                model = IntentonomyClipViTMultiStreamModule.load_from_checkpoint(
                    tmp_ckpt_path,
                    map_location=device,
                    weights_only=False
                )
                model_type = "multistream"
            except Exception:
                # 如果失败，尝试加载Codebook模型
                pass
        
        # 如果MultiStream加载失败，加载Codebook模型
        if model is None:
            model = IntentonomyClipViTCodebookModule.load_from_checkpoint(
                tmp_ckpt_path,
                map_location=device,
                weights_only=False
            )
            model_type = "codebook"
    finally:
        Path(tmp_ckpt_path).unlink(missing_ok=True)
    
    model.to(device)
    model.eval()
    
    return model, test_loader, device, model_type


def get_project_root() -> Path:
    """获取项目根目录（包含 .project-root 文件的目录）。"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / ".project-root").exists():
            return current
        current = current.parent
    # 如果找不到，返回当前文件所在目录的父目录（lightning-hydra）
    return Path(__file__).resolve().parent.parent


def scan_checkpoint_files(logs_dir: str = "logs/train/runs") -> Dict[str, str]:
    """扫描所有可用的 epoch checkpoint 文件，并返回对应的 tensorboard 目录名称。
    
    Args:
        logs_dir: 日志目录路径（相对于项目根目录）
    
    Returns:
        checkpoint_dict: 字典，key 为显示名称（tensorboard 目录路径），value 为 checkpoint 文件路径
                       格式：{ "logs/train/runs/xxxx/tensorboard/xxx": "logs/train/runs/xxxx/checkpoints/epoch_xxx.ckpt" }
                        如果有多个 epoch checkpoint，每个都会有一个条目
    """
    import re
    
    project_root = get_project_root()
    logs_path = project_root / logs_dir
    if not logs_path.exists():
        return {}
    
    checkpoint_dict = {}
    # 匹配 epoch_xxx.ckpt 格式的文件名
    epoch_pattern = re.compile(r'^epoch_\d+\.ckpt$')
    
    for run_dir in sorted(logs_path.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue
        
        # 获取 tensorboard 目录名称
        tensorboard_dir = run_dir / "tensorboard"
        if not tensorboard_dir.exists():
            continue
        
        # 获取 tensorboard 下的子目录名称
        tb_subdirs = [d for d in tensorboard_dir.iterdir() if d.is_dir()]
        if not tb_subdirs:
            continue
        
        # 使用第一个 tensorboard 子目录作为名称（只使用子目录名称，不包含完整路径）
        tb_subdir = sorted(tb_subdirs)[0]
        tb_display_name = tb_subdir.name  # 只使用子目录名称，例如 "Intentonomy-Clip-ViT-Codebook-512-Factor-Separation-Loss"
        
        # 扫描所有 epoch_xxx.ckpt 格式的文件
        epoch_checkpoints = []
        for ckpt_file in sorted(checkpoints_dir.glob("*.ckpt")):
            if epoch_pattern.match(ckpt_file.name):
                epoch_checkpoints.append(ckpt_file)
        
        # 为每个 epoch checkpoint 创建条目
        for ckpt_file in epoch_checkpoints:
            ckpt_rel_path = ckpt_file.relative_to(project_root)
            # 显示名称只使用 tensorboard 子目录名称，如果有多个 epoch 则加上 epoch 信息
            display_key = f"{tb_display_name} - {ckpt_file.stem}"
            checkpoint_dict[display_key] = str(ckpt_rel_path)
    
    return checkpoint_dict


def scan_tensorboard_dirs(logs_dir: str = "logs/train/runs") -> List[str]:
    """扫描所有可用的 tensorboard 目录。
    
    Args:
        logs_dir: 日志目录路径（相对于项目根目录）
    
    Returns:
        tensorboard_dirs: tensorboard 目录路径列表，格式为 logs/train/runs/xxxx/tensorboard/xxx
    """
    project_root = get_project_root()
    logs_path = project_root / logs_dir
    if not logs_path.exists():
        return []
    
    tensorboard_dirs = []
    for run_dir in sorted(logs_path.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        tensorboard_dir = run_dir / "tensorboard"
        if not tensorboard_dir.exists():
            continue
        
        # 扫描 tensorboard 目录下的所有子目录
        for subdir in sorted(tensorboard_dir.iterdir()):
            if subdir.is_dir():
                # 使用相对于项目根目录的路径，格式为 logs/train/runs/xxxx/tensorboard/xxx
                rel_path = subdir.relative_to(project_root)
                tensorboard_dirs.append(str(rel_path))
    
    return tensorboard_dirs


def load_anchor_texts(anchors_texts_path: str) -> Dict[str, List[str]]:
    """加载语义锚点文本映射。
    
    Args:
        anchors_texts_path: semantic_anchors_texts.pth文件路径
    
    Returns:
        anchor_texts: 字典，键为维度名称（coco, places, emotion, ava, actions），值为文本列表
    """
    if not Path(anchors_texts_path).exists():
        return {}
    
    try:
        anchor_texts = torch.load(anchors_texts_path, map_location='cpu')
        return anchor_texts
    except Exception as e:
        st.warning(f"无法加载anchor文本文件: {str(e)}")
        return {}


def extract_base_class_name(text: str) -> str:
    """从prompt文本中提取基础类别名。
    
    例如：
    - "a photo of person" -> "person"
    - "a photo of person_noise_1" -> "person"
    - "a photo of a person playing instrument" -> "playing instrument"
    - "a Black and White photo" -> "Black and White"
    
    Args:
        text: prompt文本
    
    Returns:
        基础类别名
    """
    # 移除常见的prompt前缀
    text = text.strip()
    
    # 处理 "a photo of {name}" 格式
    if text.startswith("a photo of "):
        text = text[len("a photo of "):]
    elif text.startswith("a photo of a person "):
        text = text[len("a photo of a person "):]
    elif text.startswith("a "):
        # 处理 "a {style} photo" 格式
        if " photo" in text:
            text = text[2:].replace(" photo", "")
    
    # 移除噪声后缀（如 "_noise_1", "_aug_0"）
    if "_noise_" in text:
        text = text.split("_noise_")[0]
    elif "_aug_" in text:
        text = text.split("_aug_")[0]
    
    return text.strip()


def get_image_labels(image_id: int, annotation_data: Dict, intent_names: List[str]) -> List[str]:
    """获取图像的intent标签。
    
    Args:
        image_id: 图像ID
        annotation_data: 标注数据字典（从JSON文件加载）
        intent_names: intent类别名称列表
    
    Returns:
        labels: intent标签名称列表
    """
    if not annotation_data or not intent_names:
        return []
    
    # 查找对应的annotation
    for ann in annotation_data.get("annotations", []):
        ann_image_id = ann.get("image_id")
        if ann_image_id is None:
            # 尝试其他可能的键名
            ann_image_id = ann.get("image_category_id")
        
        if ann_image_id == image_id:
            # 检查是否有category_ids
            category_ids = []
            if "category_ids" in ann:
                category_ids = ann["category_ids"]
            elif "category_ids_softprob" in ann:
                # 对于softprob，取概率>0的类别
                softprob = ann["category_ids_softprob"]
                if isinstance(softprob, list):
                    category_ids = [i for i, prob in enumerate(softprob) if prob > 0]
                else:
                    # 如果是numpy数组或tensor
                    category_ids = [i for i in range(len(softprob)) if softprob[i] > 0]
            else:
                return []
            
            # 转换为标签名称
            labels = []
            for cat_id in category_ids:
                if isinstance(cat_id, (int, np.integer)) and 0 <= cat_id < len(intent_names):
                    labels.append(intent_names[cat_id])
            
            return labels
    
    return []


def get_intent_names(annotation_dir: str) -> List[str]:
    """从标注文件中获取意图类别名称列表。
    
    Args:
        annotation_dir: 标注文件目录
    
    Returns:
        intent_names: 意图类别名称列表，按 ID 排序
    """
    # 尝试从测试标注文件中获取
    test_annotation_file = Path(annotation_dir) / "intentonomy_test2020.json"
    if not test_annotation_file.exists():
        # 尝试验证集标注文件
        test_annotation_file = Path(annotation_dir) / "intentonomy_val2020.json"
    
    if test_annotation_file.exists():
        with open(test_annotation_file, "r") as f:
            data = json.load(f)
        
        # 获取类别列表并按 ID 排序
        if "id" in data["categories"][0]:
            categories = sorted(data["categories"], key=lambda x: x["id"])
            intent_names = [cat["name"] for cat in categories]
        else:
            categories = sorted(data["categories"], key=lambda x: x["category_id"])
            intent_names = [cat["name"] for cat in categories]
        
        return intent_names
    
    # 如果文件不存在，返回默认的类别名称
    return [
        'Attractive', 'BeatCompete', 'Communicate', 'CreativeUnique',
        'CuriousAdventurousExcitingLife', 'EasyLife', 'EnjoyLife',
        'FineDesignLearnArt-Arch', 'FineDesignLearnArt-Art', 'FineDesignLearnArt-Culture',
        'GoodParentEmoCloseChild', 'Happy', 'HardWorking',
        'Harmony', 'Health', 'InLove', 'InLoveAnimal',
        'InspirOthrs', 'ManagableMakePlan', 'NatBeauty', 'PassionAbSmthing',
        'Playful', 'ShareFeelings', 'SocialLifeFriendship', 'SuccInOccupHavGdJob',
        'TchOthrs', 'ThngsInOrdr', 'WorkILike'
    ]


def collect_code_statistics(model, test_loader, device, model_type: str):
    """收集所有图像的code统计信息。
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        model_type: 模型类型（"codebook" 或 "multistream"）
    
    Returns:
        all_images: 所有图像张量 [N, C, H, W]
        all_code_indices: 所有code索引 [N, K] 或 Dict[str, torch.Tensor]
        code_counts: 每个factor的code使用频率统计 [K个Counter] 或 Dict[str, Counter]
        all_image_ids: 所有图像的ID列表
        dimension_mapping: 维度映射（对于multistream模型，factor_id到维度名的映射）
    """
    seed_everything(42)
    
    all_images = []
    all_code_indices_list = []
    all_image_ids = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = len(test_loader)
    
            with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            image_ids = batch.get("image_id", None)
            
            # 获取code索引
            code_indices = model.get_code_indices(images)
            
            # 存储图像和code索引
            all_images.append(images.cpu())
            all_code_indices_list.append(code_indices)
            
            # 存储image_id
            if image_ids is not None:
                if isinstance(image_ids, torch.Tensor):
                    all_image_ids.extend(image_ids.cpu().tolist())
                elif isinstance(image_ids, list):
                    all_image_ids.extend(image_ids)
                else:
                    # 单个值
                    all_image_ids.append(image_ids)
            else:
                # 如果没有image_id，使用索引作为占位符
                batch_size = images.shape[0]
                start_idx = len(all_image_ids)
                all_image_ids.extend(range(start_idx, start_idx + batch_size))
            
            # 更新进度
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"处理批次 {batch_idx + 1}/{total_batches}")
    
    # 合并所有批次
    all_images = torch.cat(all_images, dim=0)  # [N, C, H, W]
    
    # 处理code indices
    dimension_mapping = {}
    if model_type == "multistream":
        # MultiStream模型返回字典格式
        all_code_indices = {}
        code_counts = {}
        
        # 维度名称列表
        dimension_names = ["coco", "places", "emotion", "ava", "actions"]
        dimension_display_names = {
            "coco": "object (coco)",
            "places": "scene (places)",
            "emotion": "emotion",
            "ava": "style (ava)",
            "actions": "action (actions)"
        }
        
        # 合并每个维度的code indices
        for dim_name in dimension_names:
            if dim_name in all_code_indices_list[0]:
                # 合并所有批次的该维度code indices
                dim_indices = []
                for batch_indices in all_code_indices_list:
                    if isinstance(batch_indices, dict) and dim_name in batch_indices:
                        dim_indices.append(batch_indices[dim_name].cpu())
                
                if dim_indices:
                    all_code_indices[dim_name] = torch.cat(dim_indices, dim=0)  # [N, N_patches]
                    # 对于每个图像，取最常用的code（或第一个patch的code）
                    # 这里我们取第一个patch的code作为代表
                    image_codes = all_code_indices[dim_name][:, 0]  # [N]
                    code_counts[dim_name] = Counter(image_codes.cpu().numpy().tolist())
        
        # 创建factor_id到维度名的映射（按顺序）
        for factor_id, dim_name in enumerate(dimension_names):
            if dim_name in all_code_indices:
                dimension_mapping[factor_id] = dim_name
    else:
        # Codebook模型返回tensor格式 [B, K]
        all_code_indices = torch.cat([x.cpu() if isinstance(x, torch.Tensor) else x for x in all_code_indices_list], dim=0)  # [N, K]
        
        # 计算每个factor的code使用频率
        k_semantic_blocks = all_code_indices.shape[1]
        code_counts = []
        
        for k in range(k_semantic_blocks):
            codes = all_code_indices[:, k].cpu().numpy().tolist()
            code_counts.append(Counter(codes))
    
    progress_bar.empty()
    status_text.empty()
    
    return all_images, all_code_indices, code_counts, all_image_ids, dimension_mapping


def main():
    st.set_page_config(page_title="Codebook 可视化", layout="wide")
    st.title("Codebook Factor Code 可视化")
    st.markdown("选择 factor 查看其 top-20 最常用的 code 及其对应的图像")
    
    # 创建标签页
    tab1, tab2 = st.tabs(["Factor Code 可视化", "Factor Drop 分析"])
    
    # 侧边栏：配置参数
    with st.sidebar:
        st.header("配置")
        
        # 获取项目根目录
        project_root = get_project_root()
        
        # 默认路径设置（基于项目根目录）
        default_annotation_dir = str(project_root.parent / "Intentonomy" / "data" / "annotation")
        default_image_dir = str(project_root.parent / "Intentonomy" / "data" / "images" / "low")
        
        # 扫描可用的 checkpoint 文件（返回字典：显示名称 -> checkpoint 路径）
        checkpoint_dict = scan_checkpoint_files()
        
        # Checkpoint路径选择器
        if checkpoint_dict:
            # 获取显示选项（tensorboard 目录名称）和对应的 checkpoint 路径
            display_options = list(checkpoint_dict.keys())
            default_ckpt_idx = 0
            selected_display_name = st.selectbox(
                "模型 Checkpoint",
                options=display_options,
                index=default_ckpt_idx,
                help="选择 checkpoint，显示名称为 tensorboard 目录路径"
            )
            # 获取对应的 checkpoint 路径
            selected_ckpt = checkpoint_dict[selected_display_name]
            # 将相对路径转换为绝对路径
            ckpt_path = str(project_root / selected_ckpt) if not Path(selected_ckpt).is_absolute() else selected_ckpt
        else:
            ckpt_path_input = st.text_input(
                "模型 Checkpoint 路径",
                value="",
                help="未找到自动检测的 checkpoint，请手动输入路径"
            )
            ckpt_path = ckpt_path_input
        
        # 数据路径（使用默认值）
        annotation_dir = st.text_input(
            "标注文件目录",
            value=default_annotation_dir,
            help="包含标注 JSON 文件的目录"
        )
        
        image_dir = st.text_input(
            "图像目录",
            value=default_image_dir,
            help="包含图像的目录"
        )
        
        image_size = st.number_input(
            "图像尺寸",
            min_value=224,
            max_value=512,
            value=224,
            step=32
        )
        
        load_button = st.button("加载模型和数据", type="primary")
    
    # 主界面
    if load_button:
        if not ckpt_path or not annotation_dir or not image_dir:
            st.error("请填写所有必需的路径！")
            return
        
        if not Path(ckpt_path).exists():
            st.error(f"Checkpoint 文件不存在: {ckpt_path}")
            return
        
        if not Path(annotation_dir).exists():
            st.error(f"标注目录不存在: {annotation_dir}")
            return
        
        if not Path(image_dir).exists():
            st.error(f"图像目录不存在: {image_dir}")
            return
        
        # 加载模型和数据
        with st.spinner("正在加载模型和数据..."):
            try:
                model, test_loader, device, model_type = load_model_and_data(
                    ckpt_path, annotation_dir, image_dir, image_size
                )
                st.success(f"模型加载成功！模型类型: {model_type}")
            except Exception as e:
                st.error(f"加载模型失败: {str(e)}")
                return
        
        # 加载anchor文本映射
        project_root = get_project_root()
        anchors_texts_path = project_root / "semantic_anchors_texts.pth"
        anchor_texts = load_anchor_texts(str(anchors_texts_path))
        
        # 加载annotation数据（用于获取图像标签）
        annotation_data = {}
        test_annotation_file = Path(annotation_dir) / "intentonomy_test2020.json"
        if not test_annotation_file.exists():
            test_annotation_file = Path(annotation_dir) / "intentonomy_val2020.json"
        if test_annotation_file.exists():
            with open(test_annotation_file, "r") as f:
                annotation_data = json.load(f)
        
        # 收集code统计信息
        with st.spinner("正在收集 code 统计信息..."):
            try:
                all_images, all_code_indices, code_counts, all_image_ids, dimension_mapping = collect_code_statistics(
                    model, test_loader, device, model_type
                )
                st.success(f"收集完成！共 {len(all_images)} 张图像")
            except Exception as e:
                st.error(f"收集统计信息失败: {str(e)}")
                return
        
        # 获取意图类别名称
        intent_names = get_intent_names(annotation_dir)
        
        # 计算k_semantic_blocks
        if model_type == "multistream":
            k_semantic_blocks = len(dimension_mapping)
        else:
            k_semantic_blocks = all_code_indices.shape[1] if isinstance(all_code_indices, torch.Tensor) else 0
        
        # 保存到session state
        st.session_state['model'] = model
        st.session_state['model_type'] = model_type
        st.session_state['all_images'] = all_images
        st.session_state['all_code_indices'] = all_code_indices
        st.session_state['code_counts'] = code_counts
        st.session_state['device'] = device
        st.session_state['k_semantic_blocks'] = k_semantic_blocks
        st.session_state['intent_names'] = intent_names
        st.session_state['annotation_dir'] = annotation_dir
        st.session_state['all_image_ids'] = all_image_ids
        st.session_state['dimension_mapping'] = dimension_mapping
        st.session_state['anchor_texts'] = anchor_texts
        st.session_state['annotation_data'] = annotation_data
    
    # 如果已经加载了数据，显示factor选择界面
    if 'all_images' in st.session_state:
        with tab1:
            k_semantic_blocks = st.session_state['k_semantic_blocks']
            all_images = st.session_state['all_images']
            all_code_indices = st.session_state['all_code_indices']
            code_counts = st.session_state['code_counts']
            model_type = st.session_state.get('model_type', 'codebook')
            dimension_mapping = st.session_state.get('dimension_mapping', {})
            anchor_texts = st.session_state.get('anchor_texts', {})
            all_image_ids = st.session_state.get('all_image_ids', [])
            annotation_data = st.session_state.get('annotation_data', {})
            intent_names = st.session_state.get('intent_names', [])
            
            # 维度选择（仅对MultiStream模型显示）
            selected_dimension = None
            if model_type == "multistream" and dimension_mapping:
                st.header("选择维度")
                dimension_display_names = {
                    "coco": "object (coco)",
                    "places": "scene (places)",
                    "emotion": "emotion",
                    "ava": "style (ava)",
                    "actions": "action (actions)"
                }
                
                # 创建维度选项列表
                dimension_options = []
                for factor_id, dim_name in sorted(dimension_mapping.items()):
                    display_name = dimension_display_names.get(dim_name, dim_name)
                    dimension_options.append((factor_id, display_name, dim_name))
                
                selected_option = st.selectbox(
                    "选择要查看的维度",
                    options=range(len(dimension_options)),
                    format_func=lambda x: dimension_options[x][1]
                )
                
                factor_id, _, selected_dimension = dimension_options[selected_option]
            else:
                # Codebook模型：直接选择factor
                st.header("选择 Factor")
                factor_id = st.selectbox(
                    "选择要查看的 Factor",
                    options=list(range(k_semantic_blocks)),
                    format_func=lambda x: f"Factor {x}"
                )
            
            # 获取该factor的code统计
            if model_type == "multistream":
                if selected_dimension and selected_dimension in code_counts:
                    factor_code_counts = code_counts[selected_dimension]
                    # 获取该维度的code indices
                    factor_code_indices = all_code_indices[selected_dimension]
                else:
                    st.error("无法找到选定的维度数据")
                    return
            else:
                factor_code_counts = code_counts[factor_id]
                factor_code_indices = all_code_indices[:, factor_id]
            
            # 获取top-20 code
            top_codes = factor_code_counts.most_common(20)
            
            # 显示标题
            if model_type == "multistream" and selected_dimension:
                dimension_display = {
                    "coco": "Object (COCO)",
                    "places": "Scene (Places365)",
                    "emotion": "Emotion",
                    "ava": "Style (AVA)",
                    "actions": "Action (Stanford 40)"
                }.get(selected_dimension, selected_dimension)
                st.header(f"{dimension_display} - Top-20 Code")
            else:
                st.header(f"Factor {factor_id} - Top-20 Code")
            
            # 显示统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总图像数", len(all_images))
            with col2:
                st.metric("唯一 Code 数", len(factor_code_counts))
            with col3:
                max_code = factor_code_indices.max().item() if isinstance(factor_code_indices, torch.Tensor) else max(factor_code_counts.keys())
                st.metric("Code 范围", f"0 - {max_code}")
            
            # 显示top-20 code的统计表格
            st.subheader("Top-20 Code 使用频率")
            
            # 准备表格数据
            code_ids = []
            counts = []
            percentages = []
            semantic_labels = []
            
            for code_id, count in top_codes:
                code_ids.append(code_id)
                counts.append(count)
                percentages.append(f"{count / len(all_images) * 100:.2f}")
                
                # 获取对应的语义标签
                semantic_label = ""
                if model_type == "multistream" and selected_dimension and selected_dimension in anchor_texts:
                    anchor_text_list = anchor_texts[selected_dimension]
                    if code_id < len(anchor_text_list):
                        anchor_text = anchor_text_list[code_id]
                        semantic_label = extract_base_class_name(anchor_text)
                elif model_type == "codebook" and anchor_texts:
                    # 对于codebook模型，尝试从所有维度查找（如果存在）
                    for dim_name, anchor_text_list in anchor_texts.items():
                        if code_id < len(anchor_text_list):
                            anchor_text = anchor_text_list[code_id]
                            semantic_label = extract_base_class_name(anchor_text)
                            break
                
                semantic_labels.append(semantic_label)
            
            top_codes_data = {
                "Code ID": code_ids,
                "对应语义": semantic_labels,
                "使用次数": counts,
                "使用比例 (%)": percentages
            }
            df = pd.DataFrame(top_codes_data)
            st.dataframe(df, width='stretch')
            
            # 为每个top code显示图像
            st.subheader("Top-20 Code 对应的图像")
            
            # 获取语义标签用于显示
            semantic_label_for_code = {}
            for code_id, _ in top_codes:
                semantic_label = ""
                if model_type == "multistream" and selected_dimension and selected_dimension in anchor_texts:
                    anchor_text_list = anchor_texts[selected_dimension]
                    if code_id < len(anchor_text_list):
                        anchor_text = anchor_text_list[code_id]
                        semantic_label = extract_base_class_name(anchor_text)
                elif model_type == "codebook":
                    # 对于codebook模型，尝试从所有维度查找（如果anchor文本存在）
                    for dim_name, anchor_text_list in anchor_texts.items():
                        if code_id < len(anchor_text_list):
                            anchor_text = anchor_text_list[code_id]
                            semantic_label = extract_base_class_name(anchor_text)
                            break
                
                semantic_label_for_code[code_id] = semantic_label
            
            # 使用tabs来组织显示
            tab_labels = []
            for code_id, _ in top_codes:
                semantic = semantic_label_for_code.get(code_id, "")
                if semantic:
                    tab_labels.append(f"Code {code_id} ({semantic})")
                else:
                    tab_labels.append(f"Code {code_id}")
            
            tabs = st.tabs(tab_labels)
            
            for tab_idx, (code_id, count) in enumerate(top_codes):
                with tabs[tab_idx]:
                    semantic = semantic_label_for_code.get(code_id, "")
                    if semantic:
                        st.markdown(f"**Code {code_id} ({semantic})**: 使用 {count} 次 ({count / len(all_images) * 100:.2f}%)")
                    else:
                        st.markdown(f"**Code {code_id}**: 使用 {count} 次 ({count / len(all_images) * 100:.2f}%)")
                    
                    # 找到使用该code的图像索引
                    if model_type == "multistream":
                        # 对于MultiStream模型，需要检查每个patch的code
                        # 这里我们检查第一个patch的code
                        if selected_dimension and selected_dimension in all_code_indices:
                            dim_code_indices = all_code_indices[selected_dimension]  # [N, N_patches]
                            # 取第一个patch的code
                            image_codes = dim_code_indices[:, 0]  # [N]
                            idx = (image_codes == code_id).nonzero().squeeze()
                        else:
                            st.warning("无法找到选定的维度数据")
                            continue
                    else:
                        idx = (all_code_indices[:, factor_id] == code_id).nonzero().squeeze()
                    
                    # 检查是否有匹配的图像
                    if idx.numel() == 0:
                        st.warning("没有找到使用此 code 的图像")
                        continue
                    
                    # 如果是0维张量（只有一个匹配），转换为1维
                    if idx.dim() == 0:
                        idx = idx.unsqueeze(0)
                    
                    # 限制显示数量（最多20张）
                    display_count = min(len(idx), 20)
                    idx = idx[:display_count]
                    
                    # 显示图像网格（每行4张）
                    num_cols = 4
                    num_rows = (display_count + num_cols - 1) // num_cols
                    
                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col_idx in range(num_cols):
                            img_idx_global = row * num_cols + col_idx
                            if img_idx_global < len(idx):
                                img_idx = idx[img_idx_global]
                                img = denormalize_image(all_images[img_idx])
                                img_np = img.permute(1, 2, 0).cpu().numpy()
                                
                                # 获取图像标签
                                caption = f"图像 {img_idx.item()}"
                                if img_idx.item() < len(all_image_ids) and annotation_data:
                                    image_id = all_image_ids[img_idx.item()]
                                    labels = get_image_labels(image_id, annotation_data, intent_names)
                                    if labels:
                                        labels_str = ", ".join(labels[:3])  # 最多显示3个标签
                                        if len(labels) > 3:
                                            labels_str += f" (+{len(labels)-3})"
                                        caption += f"\n{labels_str}"
                                
                                cols[col_idx].image(
                                    img_np, 
                                    width='stretch', 
                                    caption=caption
                                )
        
        # Factor Drop 分析标签页
        with tab2:
            st.header("Factor Drop 分析")
            st.markdown("分析每个 factor 被移除后对各个 intent 预测的影响")
            
            if 'all_images' in st.session_state:
                model = st.session_state['model']
                all_images = st.session_state['all_images']
                device = st.session_state['device']
                intent_names = st.session_state.get('intent_names', [])
                k_semantic_blocks = st.session_state['k_semantic_blocks']
                
                # 设置采样数量（可选，避免计算时间过长）
                st.subheader("分析设置")
                max_samples = st.number_input(
                    "最大样本数（用于分析，0 表示使用全部数据）",
                    min_value=0,
                    max_value=len(all_images),
                    value=min(1000, len(all_images)),
                    help="使用较少的样本可以加快计算速度"
                )
                
                if st.button("开始 Factor Drop 分析", type="primary"):
                    # 选择样本
                    if max_samples > 0 and max_samples < len(all_images):
                        # 随机采样
                        indices = torch.randperm(len(all_images))[:max_samples]
                        sample_images = all_images[indices].to(device)
                    else:
                        sample_images = all_images.to(device)
                    
                    # 显示进度
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 获取量化特征
                    status_text.text("正在获取量化特征...")
                    progress_bar.progress(0.1)
                    
                    with torch.no_grad():
                        z_quantized = get_quantized_features(model, sample_images)
                    
                    # 执行 factor drop 测试
                    status_text.text("正在执行 Factor Drop 测试...")
                    progress_bar.progress(0.3)
                    
                    drops = factor_drop_test(model, z_quantized, intent_names)
                    drops_np = drops.detach().cpu().numpy()  # [F, num_intents]
                    
                    progress_bar.progress(1.0)
                    status_text.text("分析完成！")
                    progress_bar.empty()
                    status_text.empty()
                    
                    # 保存结果到 session state
                    st.session_state['factor_drop_results'] = drops_np
                    st.success(f"分析完成！共分析了 {len(sample_images)} 个样本")
                
                # 显示热力图
                if 'factor_drop_results' in st.session_state:
                    drops_np = st.session_state['factor_drop_results']
                    intent_names = st.session_state.get('intent_names', [f'Intent {i}' for i in range(drops_np.shape[1])])
                    
                    st.subheader("Factor Drop 影响热力图")
                    st.markdown("颜色越深表示该 factor 被移除后对该 intent 的影响越大")
                    
                    # 创建 DataFrame 用于 plotly
                    factor_ids = [f'Factor {i}' for i in range(drops_np.shape[0])]
                    
                    # 转置数据以便热力图显示（intents 在 Y 轴，factors 在 X 轴）
                    heatmap_data = drops_np.T  # [num_intents, F]
                    
                    # 创建热力图
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Factor", y="Intent", color="影响程度"),
                        x=factor_ids,
                        y=intent_names,
                        aspect="auto",
                        color_continuous_scale="Reds",
                    )
                    
                    # 添加文本标注（如果数据不太大）
                    if heatmap_data.size <= 500:  # 只在数据不太大时显示文本
                        fig.update_traces(
                            text=heatmap_data.round(3),
                            texttemplate="%{text}",
                            textfont={"size": 10}
                        )
                    
                    fig.update_layout(
                        title="Factor Drop 对 Intent 预测的影响",
                        xaxis_title="Factor ID",
                        yaxis_title="Intent",
                        height=max(600, len(intent_names) * 20),
                        width=max(800, len(factor_ids) * 100),
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # 显示数值表格
                    st.subheader("详细数值")
                    with st.expander("查看详细数值表格"):
                        df_drops = pd.DataFrame(
                            heatmap_data,
                            index=intent_names,
                            columns=factor_ids
                        )
                        st.dataframe(df_drops, width='stretch')
                    
                    # 显示统计信息
                    st.subheader("统计信息")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最大影响", f"{drops_np.max():.4f}")
                    with col2:
                        st.metric("平均影响", f"{drops_np.mean():.4f}")
                    with col3:
                        st.metric("最小影响", f"{drops_np.min():.4f}")
            else:
                st.info("请先加载模型和数据")


if __name__ == "__main__":
    main()

