"""
Utility functions for the Fairwashing Detection Framework.
"""

import numpy as np
import torch
import random
import os
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_demographic_parity(y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
    """
    Compute demographic parity difference.

    Args:
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values

    Returns:
        Demographic parity difference
    """
    groups = np.unique(sensitive_attr)
    rates = []
    for group in groups:
        mask = sensitive_attr == group
        rate = np.mean(y_pred[mask])
        rates.append(rate)
    return np.max(rates) - np.min(rates)


def compute_equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
    """
    Compute equalized odds difference.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values

    Returns:
        Equalized odds difference
    """
    groups = np.unique(sensitive_attr)
    tpr_diffs = []
    fpr_diffs = []

    for label in np.unique(y_true):
        tprs = []
        fprs = []
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == label)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask] == y_true[mask])
                tprs.append(tpr)

            mask_neg = (sensitive_attr == group) & (y_true != label)
            if np.sum(mask_neg) > 0:
                fpr = np.mean(y_pred[mask_neg] == label)
                fprs.append(fpr)

        if len(tprs) > 1:
            tpr_diffs.append(np.max(tprs) - np.min(tprs))
        if len(fprs) > 1:
            fpr_diffs.append(np.max(fprs) - np.min(fprs))

    return np.max(tpr_diffs + fpr_diffs) if (tpr_diffs + fpr_diffs) else 0.0


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             sensitive_attr: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive fairness metrics.

    Returns:
        Dictionary of fairness metrics
    """
    return {
        "demographic_parity": compute_demographic_parity(y_pred, sensitive_attr),
        "equalized_odds": compute_equalized_odds(y_true, y_pred, sensitive_attr),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_explanation_similarity(exp1: np.ndarray, exp2: np.ndarray, 
                                   metric: str = "ssim") -> float:
    """
    Compute similarity between two explanations.

    Args:
        exp1, exp2: Explanation arrays
        metric: Similarity metric ("ssim", "pcc", "mse", "spearman")

    Returns:
        Similarity score
    """
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr, spearmanr

    if metric == "ssim":
        if exp1.ndim == 1:
            # For 1D, use correlation-based proxy
            return np.abs(pearsonr(exp1, exp2)[0])
        return ssim(exp1, exp2, data_range=exp1.max() - exp1.min())
    elif metric == "pcc":
        return pearsonr(exp1.flatten(), exp2.flatten())[0]
    elif metric == "spearman":
        return spearmanr(exp1.flatten(), exp2.flatten())[0]
    elif metric == "mse":
        return np.mean((exp1 - exp2) ** 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_explanations(original_img: np.ndarray, 
                     original_exp: np.ndarray,
                     attacked_exp: np.ndarray,
                     title: str = "Explanation Comparison",
                     save_path: Optional[str] = None):
    """Plot original and attacked explanations side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img, cmap="gray" if original_img.ndim == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_exp, cmap="hot")
    axes[1].set_title("Original Explanation")
    axes[1].axis("off")

    axes[2].imshow(attacked_exp, cmap="hot")
    axes[2].set_title("Attacked/Manipulated Explanation")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compute_top_k_intersection(exp1: np.ndarray, exp2: np.ndarray, k: int = 1000) -> float:
    """
    Compute top-k intersection between two explanations.

    Args:
        exp1, exp2: Explanation arrays
        k: Number of top features to consider

    Returns:
        Intersection ratio
    """
    flat1 = exp1.flatten()
    flat2 = exp2.flatten()

    top_k_1 = set(np.argsort(flat1)[-k:])
    top_k_2 = set(np.argsort(flat2)[-k:])

    intersection = len(top_k_1 & top_k_2)
    return intersection / k


def compute_center_of_mass(exp: np.ndarray) -> Tuple[float, float]:
    """
    Compute center of mass of an explanation map.

    Args:
        exp: 2D explanation array

    Returns:
        (y_center, x_center)
    """
    exp = np.abs(exp)
    total = np.sum(exp)
    if total == 0:
        return (exp.shape[0] / 2, exp.shape[1] / 2)

    y_indices, x_indices = np.indices(exp.shape)
    y_center = np.sum(y_indices * exp) / total
    x_center = np.sum(x_indices * exp) / total

    return (y_center, x_center)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   path: str) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
