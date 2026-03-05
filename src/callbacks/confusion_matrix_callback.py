"""
Callback to generate confusion matrix visualization after testing.
"""
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import Callback, Trainer
from lightning.pytorch import LightningModule

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from visualize.confusion_matrix import (
    compute_multilabel_confusion_matrix,
)
from src.utils.metrics import get_best_f1_scores
from src.utils.visualization import get_class_names


class ConfusionMatrixCallback(Callback):
    """Callback to generate confusion matrix visualization after testing.
    
    This callback collects predictions and targets from the test set,
    computes a confusion matrix, and saves a visualization.
    """
    
    def __init__(
        self,
        annotation_file: Optional[str] = None,
        output_filename: str = "confusion_matrix.png",
        show_values: bool = True,
        dpi: int = 300,
    ) -> None:
        """Initialize the callback.
        
        :param annotation_file: Path to annotation JSON file for class names.
                                If None, will try to infer from default location.
        :param output_filename: Filename for the confusion matrix image.
        :param show_values: Whether to show values in confusion matrix cells.
        :param dpi: Resolution for saved figure.
        """
        super().__init__()
        self.annotation_file = annotation_file
        self.output_filename = output_filename
        self.show_values = show_values
        self.dpi = dpi
    
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends.
        
        :param trainer: The trainer instance.
        :param pl_module: The lightning module.
        """
        # Check if module has test predictions and targets
        if not hasattr(pl_module, 'test_preds_list') or not hasattr(pl_module, 'test_targets_list'):
            print("Warning: Module does not have test_preds_list or test_targets_list. Skipping confusion matrix generation.")
            return
        
        # Collect all predictions and targets. Some modules clear test_*_list in on_test_epoch_end,
        # so we also support cached numpy snapshots.
        if len(pl_module.test_preds_list) > 0:
            test_preds_all = torch.cat(pl_module.test_preds_list, dim=0).numpy()  # [N, num_classes]
            test_targets_all = torch.cat(pl_module.test_targets_list, dim=0).numpy()  # [N, num_classes]
        elif hasattr(pl_module, "test_preds_all") and hasattr(pl_module, "test_targets_all"):
            if pl_module.test_preds_all is None or pl_module.test_targets_all is None:
                print("Warning: test predictions cache is empty. Skipping confusion matrix generation.")
                return
            test_preds_all = pl_module.test_preds_all
            test_targets_all = pl_module.test_targets_all
        else:
            print("Warning: test_preds_list is empty and no cached test predictions found. Skipping confusion matrix generation.")
            return
        
        print(f"Collected {len(test_preds_all)} test predictions for confusion matrix")
        
        # Get optimal threshold
        f1_dict = get_best_f1_scores(test_targets_all, test_preds_all)
        threshold = f1_dict["threshold"]
        print(f"Using optimal threshold: {threshold:.4f}")
        
        # Convert predictions to binary using threshold
        binary_preds = (test_preds_all >= threshold).astype(np.int32)
        
        # Get class names
        class_names = [f"Class_{i}" for i in range(test_targets_all.shape[1])]
        if self.annotation_file is None:
            # Try to infer from default location
            project_root = Path(__file__).resolve().parent.parent.parent
            annotation_dir = project_root.parent / "Intentonomy" / "data" / "annotation"
            annotation_file = annotation_dir / "intentonomy_test2020.json"
            if annotation_file.exists():
                self.annotation_file = str(annotation_file)
                class_names = get_class_names(str(annotation_file))
            else:
                print(f"Warning: Could not find annotation file at {annotation_file}. Using default class names.")
        else:
            annotation_file = Path(self.annotation_file)
            if not annotation_file.exists():
                print(f"Warning: Annotation file not found at {self.annotation_file}. Using default class names.")
            else:
                class_names = get_class_names(str(annotation_file))
        
        if len(class_names) != test_targets_all.shape[1]:
            print(f"Warning: Number of class names ({len(class_names)}) does not match number of classes ({test_targets_all.shape[1]}). Using default names.")
            class_names = [f"Class_{i}" for i in range(test_targets_all.shape[1])]
        
        # Compute confusion matrix
        print("Computing confusion matrix...")
        confusion_matrix = compute_multilabel_confusion_matrix(test_targets_all, binary_preds)
        
        # Get output directory
        if trainer.logger is not None:
            if hasattr(trainer.logger, 'log_dir') and trainer.logger.log_dir:
                output_dir = Path(trainer.logger.log_dir)
            elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                output_dir = Path(trainer.logger.save_dir)
            else:
                # Fallback to trainer's log_dir
                output_dir = Path(trainer.log_dir) if hasattr(trainer, 'log_dir') else Path(".")
        else:
            # Fallback to trainer's log_dir
            output_dir = Path(trainer.log_dir) if hasattr(trainer, 'log_dir') else Path(".")
        
        # Create output path
        output_path = output_dir / self.output_filename
        
        # Visualize confusion matrix
        print(f"Generating confusion matrix visualization at {output_path}...")
        
        # Get layer index from module if available (for title customization)
        layer_idx = getattr(pl_module, 'layer_idx', None)
        
        plt.figure(figsize=(16, 14), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=self.show_values,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Normalized Frequency"},
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="gray"
        )
        
        # Set title with layer index if available
        title = "Confusion Matrix for Multi-Label Classification\n(CLIP ViT Layer CLS Token MLP)"
        if layer_idx is not None:
            title = f"Confusion Matrix for Multi-Label Classification\n(CLIP ViT Layer {layer_idx} CLS Token MLP)"
        
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
        plt.ylabel("True Class", fontsize=12, fontweight="bold")
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(str(output_path), dpi=self.dpi, bbox_inches="tight")
        print(f"Confusion matrix saved to: {output_path}")
        
        # Also save as PDF if output is PNG
        if str(output_path).endswith(".png"):
            pdf_path = str(output_path).replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Confusion matrix also saved to: {pdf_path}")
        
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
