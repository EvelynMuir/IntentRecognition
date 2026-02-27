"""Generate confusion matrix visualization for ViT CLIP MLP model on test set."""

import argparse
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rootutils
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.intentonomy_datamodule import IntentonomyDataModule
from src.models.intentonomy_clip_vit_module import IntentonomyClipViTModule
from src.utils.metrics import get_best_f1_scores
from src.utils.visualization import get_class_names


def clean_state_dict_for_loading(state_dict: Dict) -> Dict:
    """Clean state_dict for loading."""
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


def load_model(
    ckpt_path: str,
    device: torch.device
) -> IntentonomyClipViTModule:
    """Load model from checkpoint.
    
    :param ckpt_path: Path to checkpoint file.
    :param device: Device to load model on.
    :return: Loaded model.
    """
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
    
    return model


def collect_predictions(
    model: IntentonomyClipViTModule,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets from model on dataloader.
    
    :param model: The model to evaluate.
    :param dataloader: DataLoader for test set.
    :param device: Device to run inference on.
    :return: Tuple of (predictions, targets) as numpy arrays.
    """
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            images = batch["image"].to(device)
            targets = batch["labels"].to(device)
            
            # Get predictions
            logits = model(images)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_preds, all_targets


def compute_multilabel_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """Compute confusion matrix for multi-label classification.
    
    For multi-label classification, we compute a matrix where:
    - Element (i, j) represents: when true label contains class i, 
      how often predicted label contains class j
    
    :param y_true: True labels, shape (n_samples, n_classes), binary.
    :param y_pred: Predicted labels, shape (n_samples, n_classes), binary.
    :return: Confusion matrix of shape (n_classes, n_classes).
    """
    n_classes = y_true.shape[1]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    
    # For each class i (true class)
    for i in range(n_classes):
        # Find samples where true label contains class i
        mask_i = y_true[:, i] == 1
        if mask_i.sum() == 0:
            # No samples with this class, skip
            continue
        
        # For each class j (predicted class)
        for j in range(n_classes):
            # Count how often predicted label contains class j when true label contains class i
            confusion_matrix[i, j] = (y_pred[mask_i, j] == 1).sum()
        
        # Normalize by number of samples with class i
        confusion_matrix[i, :] /= mask_i.sum()
    
    return confusion_matrix


def visualize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    figsize: Tuple[int, int] = (16, 14),
    dpi: int = 300,
    show_values: bool = True,
    fmt: str = ".2f"
) -> None:
    """Visualize confusion matrix as heatmap.
    
    :param confusion_matrix: Confusion matrix of shape (n_classes, n_classes).
    :param class_names: List of class names.
    :param output_path: Path to save the figure.
    :param figsize: Figure size (width, height).
    :param dpi: Resolution for saved figure.
    :param show_values: Whether to show values in cells.
    :param fmt: Format string for values.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=show_values,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Frequency"},
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray"
    )
    
    plt.title("Confusion Matrix for Multi-Label Classification\n(ViT CLIP MLP)", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Confusion matrix saved to: {output_path}")
    
    # Also save as PDF if output is PNG
    if output_path.endswith(".png"):
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        print(f"Confusion matrix also saved to: {pdf_path}")
    
    plt.close()


def print_statistics(
    confusion_matrix: np.ndarray,
    class_names: List[str]
) -> None:
    """Print statistics from confusion matrix.
    
    :param confusion_matrix: Confusion matrix of shape (n_classes, n_classes).
    :param class_names: List of class names.
    """
    print("\n" + "="*80)
    print("Confusion Matrix Statistics")
    print("="*80)
    
    # Per-class accuracy (diagonal elements)
    print("\nPer-Class Accuracy (Diagonal Elements):")
    print("-" * 80)
    accuracies = []
    for i, class_name in enumerate(class_names):
        acc = confusion_matrix[i, i]
        accuracies.append(acc)
        print(f"{class_name:35s}: {acc:.4f}")
    
    print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Accuracy: {np.std(accuracies):.4f}")
    
    # Find most confused pairs (off-diagonal)
    print("\nTop 10 Most Confused Class Pairs (Off-Diagonal):")
    print("-" * 80)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                confused_pairs.append((i, j, confusion_matrix[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, value in confused_pairs[:10]:
        print(f"True: {class_names[i]:35s} -> Pred: {class_names[j]:35s}: {value:.4f}")
    
    print("="*80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix for ViT CLIP MLP model"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Directory containing annotation JSON files (default: from project root)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images (default: from project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/confusion_matrix.png",
        help="Output path for confusion matrix image (default: output/confusion_matrix.png)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (default: 224)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary classification (default: optimal threshold from validation set)"
    )
    parser.add_argument(
        "--use_val_for_threshold",
        action="store_true",
        help="Use validation set to find optimal threshold (default: False, use test set)"
    )
    parser.add_argument(
        "--no_values",
        action="store_true",
        help="Don't show values in confusion matrix cells"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Set default paths if not provided
    if args.annotation_dir is None:
        args.annotation_dir = str(project_root.parent / "Intentonomy" / "data" / "annotation")
    if args.image_dir is None:
        args.image_dir = str(project_root.parent / "Intentonomy" / "data" / "images" / "low")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data module
    print("Loading data...")
    dm = IntentonomyDataModule(
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    # Get class names
    test_annotation_file = Path(args.annotation_dir) / "intentonomy_test2020.json"
    class_names = get_class_names(str(test_annotation_file))
    print(f"Number of classes: {len(class_names)}")
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = load_model(args.ckpt_path, device)
    print("Model loaded successfully!")
    
    # Collect predictions on test set
    print("Collecting predictions on test set...")
    test_preds, test_targets = collect_predictions(model, test_loader, device)
    print(f"Collected {len(test_preds)} predictions")
    
    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold:.4f}")
    elif args.use_val_for_threshold:
        # Load validation set to find optimal threshold
        print("Loading validation set to find optimal threshold...")
        dm.setup(stage="validate")
        val_loader = dm.val_dataloader()
        val_preds, val_targets = collect_predictions(model, val_loader, device)
        
        f1_dict = get_best_f1_scores(val_targets, val_preds)
        threshold = f1_dict["threshold"]
        print(f"Optimal threshold from validation set: {threshold:.4f}")
    else:
        # Use test set to find optimal threshold
        print("Finding optimal threshold on test set...")
        f1_dict = get_best_f1_scores(test_targets, test_preds)
        threshold = f1_dict["threshold"]
        print(f"Optimal threshold from test set: {threshold:.4f}")
    
    # Convert predictions to binary using threshold
    print(f"Converting predictions to binary using threshold {threshold:.4f}...")
    binary_preds = (test_preds >= threshold).astype(np.int32)
    
    # Compute confusion matrix
    print("Computing confusion matrix...")
    confusion_matrix = compute_multilabel_confusion_matrix(test_targets, binary_preds)
    
    # Print statistics
    print_statistics(confusion_matrix, class_names)
    
    # Visualize
    print("Generating visualization...")
    visualize_confusion_matrix(
        confusion_matrix,
        class_names,
        str(output_path),
        figsize=(16, 14),
        dpi=args.dpi,
        show_values=not args.no_values,
        fmt=".2f"
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

