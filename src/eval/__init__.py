"""Evaluation utilities for heterogeneous graph neural networks."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate various metrics for heterogeneous GNN evaluation."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
            class_names: Optional class names.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def calculate_classification_metrics(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        y_prob: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary of metrics.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        metrics["precision_macro"] = precision
        metrics["recall_macro"] = recall
        metrics["f1_macro"] = f1
        
        # Micro averages
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average="micro", zero_division=0
        )
        metrics["precision_micro"] = precision_micro
        metrics["recall_micro"] = recall_micro
        metrics["f1_micro"] = f1_micro
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average="weighted", zero_division=0
        )
        metrics["precision_weighted"] = precision_weighted
        metrics["recall_weighted"] = recall_weighted
        metrics["f1_weighted"] = f1_weighted
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy()
            
            if self.num_classes == 2:
                # Binary classification
                metrics["auc_roc"] = roc_auc_score(y_true_np, y_prob_np[:, 1])
                metrics["auc_pr"] = average_precision_score(y_true_np, y_prob_np[:, 1])
            else:
                # Multi-class classification
                try:
                    metrics["auc_roc_ovr"] = roc_auc_score(
                        y_true_np, y_prob_np, multi_class="ovr", average="macro"
                    )
                    metrics["auc_roc_ovo"] = roc_auc_score(
                        y_true_np, y_prob_np, multi_class="ovo", average="macro"
                    )
                except ValueError:
                    # Handle case where some classes are missing
                    metrics["auc_roc_ovr"] = 0.0
                    metrics["auc_roc_ovo"] = 0.0
        
        return metrics
    
    def calculate_per_class_metrics(
        self,
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Dict[str, List[float]]:
        """Calculate per-class metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary of per-class metrics.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=None, zero_division=0
        )
        
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }
    
    def plot_confusion_matrix(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Optional path to save plot.
            figsize: Figure size.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        cm = confusion_matrix(y_true_np, y_pred_np)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_classification_report(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Plot classification report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Optional path to save plot.
            figsize: Figure size.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        report = classification_report(
            y_true_np, y_pred_np, target_names=self.class_names, output_dict=True
        )
        
        # Extract metrics for plotting
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precision = [report[cls]["precision"] for cls in classes]
        recall = [report[cls]["recall"] for cls in classes]
        f1 = [report[cls]["f1-score"] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
        
        ax.set_xlabel("Classes")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class ModelEvaluator:
    """Comprehensive model evaluator for heterogeneous GNNs."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to use.
            num_classes: Number of classes.
            class_names: Optional class names.
        """
        self.model = model
        self.device = device
        self.metrics_calc = MetricsCalculator(num_classes, class_names)
    
    def evaluate_model(
        self,
        data: Any,
        split: str = "test",
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, List[float], Tensor]]:
        """Evaluate model on specified split.
        
        Args:
            data: Graph data.
            split: Data split ('train', 'val', 'test').
            return_predictions: Whether to return predictions.
            
        Returns:
            Evaluation results.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Get mask
            if split == "train":
                mask = data.train_mask
            elif split == "val":
                mask = data.val_mask
            elif split == "test":
                mask = data.test_mask
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Forward pass
            if hasattr(data, 'node_type'):
                logits = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
            else:
                logits = self.model(data.x, data.edge_index, data.edge_type)
            
            # Get predictions
            y_true = data.y[mask]
            y_pred = logits[mask].argmax(dim=1)
            y_prob = torch.softmax(logits[mask], dim=1)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_classification_metrics(
                y_true, y_pred, y_prob
            )
            
            # Add per-class metrics
            per_class_metrics = self.metrics_calc.calculate_per_class_metrics(
                y_true, y_pred
            )
            metrics.update(per_class_metrics)
            
            # Add split information
            metrics["split"] = split
            metrics["num_samples"] = mask.sum().item()
            
            results = {"metrics": metrics}
            
            if return_predictions:
                results["predictions"] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            
            return results
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data: Any,
        split: str = "test",
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names to models.
            data: Graph data.
            split: Data split.
            
        Returns:
            Comparison results.
        """
        results = {}
        
        for name, model in models.items():
            # Temporarily replace model
            original_model = self.model
            self.model = model
            self.model.to(self.device)
            
            # Evaluate
            eval_results = self.evaluate_model(data, split)
            results[name] = eval_results["metrics"]
            
            # Restore original model
            self.model = original_model
        
        return results
    
    def plot_evaluation_results(
        self,
        data: Any,
        split: str = "test",
        save_dir: Optional[str] = None,
    ) -> None:
        """Plot evaluation results.
        
        Args:
            data: Graph data.
            split: Data split.
            save_dir: Optional directory to save plots.
        """
        results = self.evaluate_model(data, split, return_predictions=True)
        
        y_true = results["predictions"]["y_true"]
        y_pred = results["predictions"]["y_pred"]
        
        # Plot confusion matrix
        cm_path = f"{save_dir}/confusion_matrix_{split}.png" if save_dir else None
        self.metrics_calc.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # Plot classification report
        cr_path = f"{save_dir}/classification_report_{split}.png" if save_dir else None
        self.metrics_calc.plot_classification_report(y_true, y_pred, cr_path)
    
    def generate_report(
        self,
        data: Any,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            data: Graph data.
            save_path: Optional path to save report.
            
        Returns:
            Report string.
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("HETEROGENEOUS GNN EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Evaluate on all splits
        for split in ["train", "val", "test"]:
            results = self.evaluate_model(data, split)
            metrics = results["metrics"]
            
            report_lines.append(f"\n{split.upper()} SET RESULTS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Number of samples: {metrics['num_samples']}")
            report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report_lines.append(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            report_lines.append(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            report_lines.append(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
            report_lines.append(f"Precision (Micro): {metrics['precision_micro']:.4f}")
            report_lines.append(f"Recall (Micro): {metrics['recall_micro']:.4f}")
            report_lines.append(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
            
            if "auc_roc_ovr" in metrics:
                report_lines.append(f"AUC-ROC (OVR): {metrics['auc_roc_ovr']:.4f}")
                report_lines.append(f"AUC-ROC (OVO): {metrics['auc_roc_ovo']:.4f}")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text
