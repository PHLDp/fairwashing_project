"""
Paper 4: Interpretation of Neural Networks is Fragile
(Ghorbani et al., AAAI 2019)

This module implements:
1. Top-k attack on feature importance
2. Mass-center attack
3. Targeted attack
4. Random sign perturbation baseline
5. Influence function attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.stats import spearmanr


class InterpretationAttack:
    """Base class for interpretation attacks."""

    def __init__(self, model: nn.Module, epsilon: float = 8.0,
                 norm: str = "linf", device: str = "cpu"):
        self.model = model.to(device)
        self.epsilon = epsilon / 255.0 if epsilon > 1 else epsilon  # Normalize if needed
        self.norm = norm
        self.device = device
        self.model.eval()

    def compute_gradient_explanation(self, x: torch.Tensor, 
                                     target_class: int) -> torch.Tensor:
        """Compute simple gradient explanation."""
        x = x.clone().requires_grad_(True)
        output = self.model(x)
        score = output[:, target_class].sum()

        self.model.zero_grad()
        grad = torch.autograd.grad(score, x)[0]
        return grad.detach()

    def compute_integrated_gradients(self, x: torch.Tensor,
                                     target_class: int, 
                                     n_steps: int = 50) -> torch.Tensor:
        """Compute integrated gradients."""
        baseline = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, n_steps).view(-1, 1, 1, 1).to(x.device)

        interpolated = baseline + alphas * (x - baseline)
        interpolated.requires_grad = True

        outputs = self.model(interpolated)
        scores = outputs[:, target_class]

        gradients = torch.autograd.grad(scores.sum(), interpolated)[0]
        avg_grad = gradients.mean(dim=0)

        ig = (x[0] - baseline[0]) * avg_grad
        return ig.detach()

    def clip_perturbation(self, delta: torch.Tensor) -> torch.Tensor:
        """Clip perturbation to epsilon ball."""
        if self.norm == "linf":
            return torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.norm == "l2":
            norm = delta.norm(p=2)
            if norm > self.epsilon:
                delta = delta * self.epsilon / norm
            return delta
        return delta


class TopKAttack(InterpretationAttack):
    """
    Top-k attack: Decrease importance of top-k features.
    """

    def __init__(self, model: nn.Module, k: int = 1000, 
                 epsilon: float = 8.0, n_iterations: int = 300,
                 step_size: float = 0.5, device: str = "cpu"):
        super().__init__(model, epsilon, device=device)
        self.k = k
        self.n_iterations = n_iterations
        self.step_size = step_size / 255.0  # Normalize

    def attack(self, x: torch.Tensor, explanation_method: str = "gradient") -> torch.Tensor:
        """
        Generate adversarial perturbation.

        Args:
            x: Input image (1, C, H, W)
            explanation_method: "gradient", "integrated_gradients", "deeplift"

        Returns:
            Perturbed image
        """
        x = x.to(self.device)
        original_class = self.model(x).argmax(dim=1).item()

        # Get original explanation
        if explanation_method == "gradient":
            orig_exp = self.compute_gradient_explanation(x, original_class)
        elif explanation_method == "integrated_gradients":
            orig_exp = self.compute_integrated_gradients(x, original_class)
        else:
            orig_exp = self.compute_gradient_explanation(x, original_class)

        # Find top-k features
        flat_exp = orig_exp.abs().flatten()
        top_k_indices = torch.argsort(flat_exp, descending=True)[:self.k]

        # Initialize perturbation
        delta = torch.zeros_like(x)
        delta.requires_grad = True

        best_delta = delta.clone()
        best_dissimilarity = -float('inf')

        for i in range(self.n_iterations):
            x_perturbed = x + delta
            x_perturbed = torch.clamp(x_perturbed, 0, 1)

            # Check prediction preserved
            pred = self.model(x_perturbed).argmax(dim=1).item()
            if pred != original_class:
                continue

            # Compute perturbed explanation
            if explanation_method == "gradient":
                pert_exp = self.compute_gradient_explanation(x_perturbed, original_class)
            else:
                pert_exp = self.compute_gradient_explanation(x_perturbed, original_class)

            # Dissimilarity: negative sum of top-k features in perturbed
            flat_pert = pert_exp.abs().flatten()
            dissimilarity = -flat_pert[top_k_indices].sum()

            if dissimilarity > best_dissimilarity:
                best_dissimilarity = dissimilarity
                best_delta = delta.detach().clone()

            # Gradient step
            if delta.grad is not None:
                delta.grad.zero_()

            dissimilarity.backward()

            with torch.no_grad():
                delta.data = delta.data + self.step_size * delta.grad.sign()
                delta.data = self.clip_perturbation(delta.data)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x

            delta.requires_grad = True

        return torch.clamp(x + best_delta, 0, 1)


class MassCenterAttack(InterpretationAttack):
    """
    Mass-center attack: Shift center of mass of explanation.
    """

    def __init__(self, model: nn.Module, epsilon: float = 8.0,
                 n_iterations: int = 300, step_size: float = 0.5,
                 device: str = "cpu"):
        super().__init__(model, epsilon, device=device)
        self.n_iterations = n_iterations
        self.step_size = step_size / 255.0

    def compute_center_of_mass(self, exp: torch.Tensor) -> Tuple[float, float]:
        """Compute center of mass of explanation."""
        exp = exp.abs()
        total = exp.sum()
        if total == 0:
            return (exp.shape[-2] / 2, exp.shape[-1] / 2)

        h, w = exp.shape[-2:]
        y_coords = torch.arange(h, device=exp.device).view(-1, 1).float()
        x_coords = torch.arange(w, device=exp.device).view(1, -1).float()

        y_center = (exp.sum(dim=(-3, -1)) * y_coords).sum() / total
        x_center = (exp.sum(dim=(-3, -2)) * x_coords).sum() / total

        return (y_center.item(), x_center.item())

    def attack(self, x: torch.Tensor, target_center: Optional[Tuple[float, float]] = None,
               explanation_method: str = "gradient") -> torch.Tensor:
        """
        Shift explanation center of mass.

        Args:
            x: Input image
            target_center: Target center (y, x). If None, shifts to corner.
            explanation_method: Explanation method to attack
        """
        x = x.to(self.device)
        original_class = self.model(x).argmax(dim=1).item()

        # Get original center
        if explanation_method == "gradient":
            orig_exp = self.compute_gradient_explanation(x, original_class)
        else:
            orig_exp = self.compute_gradient_explanation(x, original_class)

        orig_center = self.compute_center_of_mass(orig_exp)

        if target_center is None:
            # Target: shift to top-left corner
            target_center = (0, 0)

        delta = torch.zeros_like(x)
        delta.requires_grad = True

        best_delta = delta.clone()
        best_distance = 0

        for i in range(self.n_iterations):
            x_perturbed = torch.clamp(x + delta, 0, 1)

            pred = self.model(x_perturbed).argmax(dim=1).item()
            if pred != original_class:
                continue

            # Compute perturbed explanation
            pert_exp = self.compute_gradient_explanation(x_perturbed, original_class)
            pert_center = self.compute_center_of_mass(pert_exp)

            # Distance from original center (maximize)
            distance = ((pert_center[0] - orig_center[0])**2 + 
                       (pert_center[1] - orig_center[1])**2)

            if distance > best_distance:
                best_distance = distance
                best_delta = delta.detach().clone()

            # Gradient step toward target
            target_dist = ((pert_center[0] - target_center[0])**2 +
                          (pert_center[1] - target_center[1])**2)

            if delta.grad is not None:
                delta.grad.zero_()

            target_dist = torch.tensor(target_dist, requires_grad=False)
            # Simplified: use sign gradient
            with torch.no_grad():
                # Approximate gradient direction
                delta.data = delta.data + self.step_size * torch.randn_like(delta).sign()
                delta.data = self.clip_perturbation(delta.data)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x

            delta.requires_grad = True

        return torch.clamp(x + best_delta, 0, 1)


class RandomSignPerturbation(InterpretationAttack):
    """Random sign perturbation baseline."""

    def attack(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random sign perturbation."""
        delta = torch.randn_like(x).sign() * self.epsilon
        return torch.clamp(x + delta, 0, 1)


class InfluenceFunctionAttack:
    """
    Attack on influence functions.
    Simplified implementation.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def compute_influence(self, x_train: torch.Tensor, y_train: torch.Tensor,
                         x_test: torch.Tensor, y_test: torch.Tensor,
                         damping: float = 0.01) -> torch.Tensor:
        """
        Compute influence of training points on test point.
        Simplified using gradient similarity.
        """
        self.model.eval()

        # Compute test gradient
        x_test = x_test.to(self.device)
        y_test = torch.tensor([y_test], device=self.device)

        output = self.model(x_test)
        loss = F.cross_entropy(output, y_test)

        test_grad = torch.autograd.grad(loss, self.model.parameters())
        test_grad_flat = torch.cat([g.flatten() for g in test_grad])

        influences = []
        for i in range(len(x_train)):
            xi = x_train[i:i+1].to(self.device)
            yi = torch.tensor([y_train[i]], device=self.device)

            output_i = self.model(xi)
            loss_i = F.cross_entropy(output_i, yi)

            train_grad = torch.autograd.grad(loss_i, self.model.parameters())
            train_grad_flat = torch.cat([g.flatten() for g in train_grad])

            # Influence ≈ negative dot product (simplified)
            influence = -torch.dot(test_grad_flat, train_grad_flat).item()
            influences.append(influence)

        return torch.tensor(influences)

    def attack(self, x_train: torch.Tensor, y_train: torch.Tensor,
               x_test: torch.Tensor, y_test: torch.Tensor,
               epsilon: float = 8.0) -> torch.Tensor:
        """
        Generate perturbation to change influence rankings.
        """
        epsilon = epsilon / 255.0

        # Compute influences
        influences = self.compute_influence(x_train, y_train, x_test, y_test)

        # Top influential points
        top_k = min(3, len(influences))
        top_indices = torch.argsort(influences, descending=True)[:top_k]

        # Generate perturbation to reduce their influence
        delta = torch.zeros_like(x_test)

        for idx in top_indices:
            xi = x_train[idx:idx+1].to(self.device)

            # Gradient of influence w.r.t. test input
            x_test_grad = x_test.clone().requires_grad_(True)
            output = self.model(x_test_grad)
            loss = F.cross_entropy(output, torch.tensor([y_test], device=self.device))

            grad = torch.autograd.grad(loss, x_test_grad)[0]
            delta = delta - epsilon * grad.sign()

        delta = torch.clamp(delta, -epsilon, epsilon)
        return torch.clamp(x_test + delta, 0, 1)


def compute_interpretation_metrics(orig_exp: np.ndarray, 
                                   pert_exp: np.ndarray,
                                   k: int = 1000) -> Dict:
    """
    Compute metrics for interpretation fragility.

    Returns:
        Dictionary with rank correlation, top-k intersection, etc.
    """
    # Spearman rank correlation
    orig_flat = orig_exp.flatten()
    pert_flat = pert_exp.flatten()

    if len(orig_flat) > 1:
        spearman_corr, _ = spearmanr(orig_flat, pert_flat)
    else:
        spearman_corr = 0.0

    # Top-k intersection
    top_k_orig = set(np.argsort(np.abs(orig_flat))[-k:])
    top_k_pert = set(np.argsort(np.abs(pert_flat))[-k:])
    top_k_intersection = len(top_k_orig & top_k_pert) / k

    # Center shift
    def center_of_mass(exp):
        exp = np.abs(exp)
        total = exp.sum()
        if total == 0:
            return (0, 0)
        if exp.ndim == 1:
            coords = np.arange(len(exp))
            return ((coords * exp).sum() / total,)
        h, w = exp.shape[-2:]
        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)
        y_c = (exp.sum(axis=-1) * y_coords.flatten()).sum() / total
        x_c = (exp.sum(axis=-2) * x_coords.flatten()).sum() / total
        return (y_c, x_c)

    orig_center = center_of_mass(orig_exp)
    pert_center = center_of_mass(pert_exp)

    if len(orig_center) == 2:
        center_shift = np.sqrt((orig_center[0] - pert_center[0])**2 + 
                              (orig_center[1] - pert_center[1])**2)
    else:
        center_shift = abs(orig_center[0] - pert_center[0])

    return {
        "spearman_correlation": spearman_corr,
        "top_k_intersection": top_k_intersection,
        "center_shift": center_shift,
    }
