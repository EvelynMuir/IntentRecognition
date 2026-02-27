"""Streamlit application for visualizing CLIP ViT + MLP attention maps on test set."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rootutils
import streamlit as st
import torch
from lightning import seed_everything
from PIL import Image
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.intentonomy_datamodule import IntentonomyDataModule
from src.models.intentonomy_clip_vit_module import IntentonomyClipViTModule
from src.utils.attention_extractor import AttentionExtractor
from src.utils.visualization import (
    visualize_attention_map,
    get_class_names,
    format_labels,
    denormalize_image,
    create_attention_figure
)

# Set page config
st.set_page_config(
    page_title="CLIP ViT Attention Visualization",
    page_icon="👁️",
    layout="wide"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'extractor' not in st.session_state:
    st.session_state.extractor = None


def clean_state_dict_for_loading(state_dict: Dict) -> Dict:
    """Clean state_dict for loading."""
    from collections import OrderedDict
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        # Remove ema_model.module. prefix if exists
        if new_key.startswith("ema_model.module."):
            continue
        # Remove net._orig_mod. prefix (torch.compile)
        if new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod."):]
        new_state_dict[new_key] = v
    return new_state_dict


@st.cache_resource
def load_model_and_data(
    ckpt_path: str,
    annotation_dir: str,
    image_dir: str,
    image_size: int = 224
) -> Tuple:
    """Load model and test data.
    
    :return: Tuple of (model, test_loader, device, class_names)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data module
    dm = IntentonomyDataModule(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        image_size=image_size,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    # Get class names from annotation file
    test_annotation_file = Path(annotation_dir) / "intentonomy_test2020.json"
    class_names = get_class_names(str(test_annotation_file))
    
    # Load model
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
        cleaned_state_dict = clean_state_dict_for_loading(original_state_dict)
        checkpoint["state_dict"] = cleaned_state_dict
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp_file:
        tmp_ckpt_path = tmp_file.name
        torch.save(checkpoint, tmp_ckpt_path)
    
    try:
        model = IntentonomyClipViTModule.load_from_checkpoint(
            tmp_ckpt_path,
            map_location=device,
            weights_only=False
        )
    finally:
        Path(tmp_ckpt_path).unlink(missing_ok=True)
    
    model.to(device)
    model.eval()
    
    return model, test_loader, device, class_names


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / ".project-root").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def main():
    """Main Streamlit application."""
    st.title("👁️ CLIP ViT + MLP Attention Map Visualization")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Checkpoint path input
        ckpt_path = st.text_input(
            "Checkpoint Path",
            value="",
            help="Path to the trained CLIP ViT + MLP model checkpoint"
        )
        
        # Data paths
        project_root = get_project_root()
        default_annotation_dir = str(project_root / "Intentonomy" / "data" / "annotation")
        default_image_dir = str(project_root / "Intentonomy" / "data" / "images" / "low")
        
        annotation_dir = st.text_input(
            "Annotation Directory",
            value=default_annotation_dir,
            help="Directory containing annotation JSON files"
        )
        
        image_dir = st.text_input(
            "Image Directory",
            value=default_image_dir,
            help="Directory containing images"
        )
        
        image_size = st.number_input(
            "Image Size",
            min_value=224,
            max_value=512,
            value=224,
            step=32,
            help="Input image size"
        )
        
        # Load model button
        load_model = st.button("🔄 Load Model", type="primary")
        
        if load_model:
            if not ckpt_path:
                st.error("Please provide a checkpoint path!")
                return
            
            if not Path(ckpt_path).exists():
                st.error(f"Checkpoint file not found: {ckpt_path}")
                return
            
            with st.spinner("Loading model and data..."):
                try:
                    model, test_loader, device, class_names = load_model_and_data(
                        ckpt_path, annotation_dir, image_dir, image_size
                    )
                    st.session_state.model = model
                    st.session_state.test_loader = test_loader
                    st.session_state.device = device
                    st.session_state.class_names = class_names
                    st.session_state.model_loaded = True
                    
                    # Create attention extractor
                    st.session_state.extractor = AttentionExtractor(model)
                    
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Please load a model using the sidebar configuration.")
        return
    
    model = st.session_state.model
    test_loader = st.session_state.test_loader
    device = st.session_state.device
    class_names = st.session_state.class_names
    extractor = st.session_state.extractor
    
    # Image selection
    st.header("📸 Image Selection")
    
    # Get total number of images
    total_images = len(test_loader.dataset)
    
    col1, col2 = st.columns(2)
    with col1:
        image_idx = st.number_input(
            "Image Index",
            min_value=0,
            max_value=total_images - 1,
            value=0,
            step=1,
            help=f"Select image index (0-{total_images-1})"
        )
    
    with col2:
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold for binary classification"
        )
    
    # Load selected image
    try:
        sample = test_loader.dataset[image_idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)  # Add batch dimension
        true_labels = sample["labels"]
        image_id = sample["image_id"]
        
        # Get prediction
        with torch.no_grad():
            logits = model(image_tensor)
            pred_probs = torch.sigmoid(logits)[0]
            pred_labels = (pred_probs >= threshold).float()
        
        # Extract attention maps
        st.header("🔍 Attention Map Extraction")
        
        # Layer selection
        num_layers = len(model.net.backbone.transformer.resblocks)
        selected_layers = st.multiselect(
            "Select Transformer Layers",
            options=list(range(num_layers)),
            default=[num_layers - 1],  # Default to last layer
            help="Select which transformer layers to visualize"
        )
        
        if not selected_layers:
            st.warning("Please select at least one layer!")
            return
        
        with st.spinner("Extracting attention maps..."):
            attention_maps_dict = extractor.extract_attention_maps(
                image_tensor,
                layer_indices=selected_layers
            )
        
        # Determine patch size based on model
        clip_model_name = getattr(model.net, 'clip_model_name', 'ViT-B/32')
        if 'ViT-B/32' in clip_model_name:
            patch_size = 32
        elif 'ViT-B/16' in clip_model_name:
            patch_size = 16
        elif 'ViT-L/14' in clip_model_name:
            patch_size = 14
        else:
            patch_size = 32  # Default
        
        # Convert attention weights to maps
        attention_maps = {}
        for layer_idx, attn_weights in attention_maps_dict.items():
            try:
                attention_map = extractor.get_cls_attention_map(
                    attn_weights, image_size, patch_size
                )
                # Validate attention map
                if attention_map.size == 0:
                    st.warning(f"Attention map for layer {layer_idx} is empty. Shape: {attention_map.shape}")
                    continue
                if attention_map.ndim != 2:
                    st.warning(f"Attention map for layer {layer_idx} is not 2D. Shape: {attention_map.shape}")
                    continue
                attention_maps[layer_idx] = attention_map
            except Exception as e:
                st.warning(f"Failed to extract attention map for layer {layer_idx}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                continue
        
        # Display results
        st.header("📊 Results")
        
        # Image and attention visualization
        denormalized_image = denormalize_image(image_tensor[0])
        
        # Check if we have any valid attention maps
        if not attention_maps:
            st.error("No valid attention maps were extracted. Please check the model and layer selection.")
        else:
            # Create tabs for different layers
            if len(selected_layers) > 1:
                tabs = st.tabs([f"Layer {i}" for i in selected_layers if i in attention_maps])
                for tab, layer_idx in zip(tabs, [i for i in selected_layers if i in attention_maps]):
                    with tab:
                        attention_map = attention_maps[layer_idx]
                        
                        # Create figure
                        try:
                            fig = create_attention_figure(denormalized_image, attention_map)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Error visualizing attention map for layer {layer_idx}: {str(e)}")
            else:
                # Single layer - show side by side
                layer_idx = selected_layers[0]
                if layer_idx in attention_maps:
                    attention_map = attention_maps[layer_idx]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(denormalized_image, use_container_width=True)
                    
                    with col2:
                        st.subheader(f"Attention Map (Layer {layer_idx})")
                        try:
                            overlaid = visualize_attention_map(denormalized_image, attention_map)
                            st.image(overlaid, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error visualizing attention map: {str(e)}")
                            st.text(f"Attention map shape: {attention_map.shape}")
                            st.text(f"Image shape: {denormalized_image.shape}")
                else:
                    st.warning(f"Attention map for layer {layer_idx} is not available.")
        
        # Labels display
        st.header("🏷️ Labels")
        
        formatted_labels = format_labels(
            true_labels, pred_labels, pred_probs, class_names, threshold
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("✅ True Labels")
            if formatted_labels['true']:
                for name, prob, is_true, is_pred in formatted_labels['true']:
                    st.text(f"• {name}")
            else:
                st.text("None")
        
        with col2:
            st.subheader("🔮 Predicted Labels")
            if formatted_labels['predicted']:
                for name, prob, is_true, is_pred in formatted_labels['predicted']:
                    color = "🟢" if is_true else "🔴"
                    st.text(f"{color} {name} ({prob:.3f})")
            else:
                st.text("None")
        
        with col3:
            st.subheader("✓ Correct")
            if formatted_labels['correct']:
                for name, prob, is_true, is_pred in formatted_labels['correct']:
                    st.text(f"• {name} ({prob:.3f})")
            else:
                st.text("None")
        
        with col4:
            st.subheader("✗ Incorrect")
            if formatted_labels['incorrect']:
                for name, prob, is_true, is_pred in formatted_labels['incorrect']:
                    if is_true:
                        st.text(f"🔴 FN: {name}")
                    else:
                        st.text(f"🔴 FP: {name} ({prob:.3f})")
            else:
                st.text("None")
        
        # Statistics
        st.header("📈 Statistics")
        
        num_true = len(formatted_labels['true'])
        num_pred = len(formatted_labels['predicted'])
        num_correct = len(formatted_labels['correct'])
        num_incorrect = len(formatted_labels['incorrect'])
        
        if num_true > 0:
            precision = num_correct / num_pred if num_pred > 0 else 0.0
            recall = num_correct / num_true if num_true > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{precision:.3f}")
            with col2:
                st.metric("Recall", f"{recall:.3f}")
            with col3:
                st.metric("F1 Score", f"{f1:.3f}")
            with col4:
                st.metric("Image ID", str(image_id))
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

