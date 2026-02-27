"""Visualization utilities for attention maps and labels."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom


def visualize_attention_map(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.6,
    cmap: str = 'jet',
    resize_method: str = 'bilinear'
) -> np.ndarray:
    """Overlay attention map on image.
    
    :param image: Original image array of shape [H, W, 3] in [0, 1] range.
    :param attention_map: Attention map of shape [H_attn, W_attn].
    :param alpha: Transparency factor for attention overlay (0-1).
    :param cmap: Colormap name for attention visualization.
    :param resize_method: Resize method ('bilinear' or 'nearest').
    :return: Overlaid image of shape [H, W, 3] in [0, 1] range.
    """
    # Validate inputs
    if attention_map.size == 0:
        raise ValueError(f"Attention map is empty. Shape: {attention_map.shape}")
    
    if image.size == 0:
        raise ValueError(f"Image is empty. Shape: {image.shape}")
    
    # Ensure attention_map is 2D
    if attention_map.ndim != 2:
        raise ValueError(f"Attention map must be 2D, got shape: {attention_map.shape}")
    
    # Normalize attention map to [0, 1]
    attn_min = attention_map.min()
    attn_max = attention_map.max()
    
    if attn_max - attn_min < 1e-8:
        # If all values are the same, create uniform attention map
        attn_norm = np.ones_like(attention_map)
    else:
        attn_norm = (attention_map - attn_min) / (attn_max - attn_min + 1e-8)
    
    # Resize attention map to match image size
    img_h, img_w = image.shape[:2]
    attn_h, attn_w = attention_map.shape
    
    if attn_h != img_h or attn_w != img_w:
        zoom_factors = (img_h / attn_h, img_w / attn_w)
        if resize_method == 'bilinear':
            attn_norm = zoom(attn_norm, zoom_factors, order=1)
        else:
            attn_norm = zoom(attn_norm, zoom_factors, order=0)
    
    # Apply colormap to attention map
    cmap_func = cm.get_cmap(cmap)
    attn_colored = cmap_func(attn_norm)[:, :, :3]  # Remove alpha channel
    
    # Overlay attention on image
    overlaid = (1 - alpha) * image + alpha * attn_colored
    
    # Ensure values are in [0, 1]
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def get_class_names(annotation_file: str) -> List[str]:
    """Get class names from annotation JSON file.
    
    :param annotation_file: Path to annotation JSON file.
    :return: List of class names in order.
    """
    import json
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    
    # Handle different annotation formats
    if categories and 'name' in categories[0]:
        class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x.get('id', x.get('category_id', 0)))]
    else:
        # Fallback: use default class names
        class_names = [
            'Attractive', 'BeatCompete', 'Communicate', 'CreativeUnique',
            'CuriousAdventurousExcitingLife', 'EasyLife', 'EnjoyLife',
            'FineDesignLearnArt-Arch', 'FineDesignLearnArt-Art', 'FineDesignLearnArt-Culture',
            'GoodParentEmoCloseChild', 'Happy', 'HardWorking',
            'Harmony', 'Health', 'InLove', 'InLoveAnimal',
            'InspirOthrs', 'ManagableMakePlan', 'NatBeauty', 'PassionAbSmthing',
            'Playful', 'ShareFeelings', 'SocialLifeFriendship', 'SuccInOccupHavGdJob',
            'TchOthrs', 'ThngsInOrdr', 'WorkILike'
        ]
    
    return class_names


def format_labels(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_probs: torch.Tensor,
    class_names: List[str],
    threshold: float = 0.5
) -> Dict[str, List[Tuple[str, float, bool, bool]]]:
    """Format labels for display.
    
    :param true_labels: Ground truth labels tensor of shape [num_classes].
    :param pred_labels: Predicted labels tensor (after thresholding) of shape [num_classes].
    :param pred_probs: Prediction probabilities tensor of shape [num_classes].
    :param class_names: List of class names.
    :param threshold: Threshold for binary classification.
    :return: Dictionary with 'true', 'predicted', 'correct', 'incorrect' lists.
              Each list contains tuples of (class_name, probability, is_true, is_predicted).
    """
    true_labels_np = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    pred_probs_np = pred_probs.cpu().numpy() if isinstance(pred_probs, torch.Tensor) else pred_probs
    pred_labels_np = (pred_probs_np >= threshold).astype(float)
    
    result = {
        'true': [],
        'predicted': [],
        'correct': [],
        'incorrect': []
    }
    
    for i, class_name in enumerate(class_names):
        is_true = true_labels_np[i] > 0.5
        is_pred = pred_labels_np[i] > 0.5
        prob = float(pred_probs_np[i])
        
        label_info = (class_name, prob, is_true, is_pred)
        
        if is_true:
            result['true'].append(label_info)
        
        if is_pred:
            result['predicted'].append(label_info)
        
        if is_true and is_pred:
            result['correct'].append(label_info)
        elif (is_true and not is_pred) or (not is_true and is_pred):
            result['incorrect'].append(label_info)
    
    # Sort by probability (descending)
    for key in result:
        result[key].sort(key=lambda x: x[1], reverse=True)
    
    return result


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor using ImageNet mean/std.
    
    :param tensor: Image tensor of shape (3, H, W) or (H, W, 3), normalized.
    :return: Image array of shape (H, W, 3) in [0, 1] range.
    """
    # Get device from input tensor
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
    
    # Handle different tensor shapes
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch item
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    
    # Denormalize
    img = tensor * std.squeeze() + mean.squeeze()
    img = img.clamp(0.0, 1.0)
    
    # Convert to numpy
    img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
    
    return img_np


def create_attention_figure(
    image: np.ndarray,
    attention_map: np.ndarray,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Create a matplotlib figure showing original image and attention overlay.
    
    :param image: Original image array of shape [H, W, 3] in [0, 1] range.
    :param attention_map: Attention map of shape [H_attn, W_attn].
    :param figsize: Figure size (width, height).
    :return: Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Attention overlay
    overlaid = visualize_attention_map(image, attention_map)
    axes[1].imshow(overlaid)
    axes[1].set_title('Attention Map Overlay', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    return fig

