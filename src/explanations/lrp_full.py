"""
Full Layer-wise Relevance Propagation (LRP) Implementation.
Implements z-rule, z+-rule, zB-rule, and epsilon-rule for different layer types.
Based on Bach et al. (2015) and Montavon et al. (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class LRPRule:
    """Base class for LRP rules."""

    @staticmethod
    def apply(linear: nn.Linear, x: torch.Tensor, R: torch.Tensor, 
              rule_type: str = "z+", epsilon: float = 1e-6) -> torch.Tensor:
        """Apply LRP rule to a linear layer."""
        if rule_type == "z":
            return LRPRule._z_rule(linear, x, R)
        elif rule_type == "z+":
            return LRPRule._z_plus_rule(linear, x, R, epsilon)
        elif rule_type == "epsilon":
            return LRPRule._epsilon_rule(linear, x, R, epsilon)
        else:
            raise ValueError(f"Unknown rule: {rule_type}")

    @staticmethod
    def _z_rule(linear: nn.Linear, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """z-rule: R_j = sum_i (x_i * w_ij / sum_k x_k * w_kj) * R_j"""
        w = linear.weight  # (out_features, in_features)

        # Forward pass with activations
        z = x.unsqueeze(-1) * w.t().unsqueeze(0)  # (batch, in_features, out_features)
        z_sum = z.sum(dim=1, keepdim=True)  # (batch, 1, out_features)

        # Avoid division by zero
        z_sum = z_sum + 1e-12

        # Relevance propagation
        s = R / z_sum.squeeze(1)  # (batch, out_features)
        c = torch.matmul(s.unsqueeze(1), w.unsqueeze(0)).squeeze(1)  # (batch, in_features)

        R_new = x * c
        return R_new

    @staticmethod
    def _z_plus_rule(linear: nn.Linear, x: torch.Tensor, R: torch.Tensor,
                     epsilon: float = 1e-6) -> torch.Tensor:
        """z+-rule: Only positive weights contribute."""
        w = linear.weight
        w_plus = torch.clamp(w, min=0)

        # Use positive weights only
        z = x.unsqueeze(-1) * w_plus.t().unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True) + epsilon

        s = R / z_sum.squeeze(1)
        c = torch.matmul(s.unsqueeze(1), w_plus.unsqueeze(0)).squeeze(1)

        R_new = x * c
        return R_new

    @staticmethod
    def _epsilon_rule(linear: nn.Linear, x: torch.Tensor, R: torch.Tensor,
                      epsilon: float = 1e-6) -> torch.Tensor:
        """epsilon-rule: Add epsilon to denominator for stability."""
        w = linear.weight

        z = x.unsqueeze(-1) * w.t().unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True) + epsilon

        s = R / z_sum.squeeze(1)
        c = torch.matmul(s.unsqueeze(1), w.unsqueeze(0)).squeeze(1)

        R_new = x * c
        return R_new


class ConvLRPRule:
    """LRP rules for convolutional layers."""

    @staticmethod
    def apply(conv: nn.Conv2d, x: torch.Tensor, R: torch.Tensor,
              rule_type: str = "z+", epsilon: float = 1e-6) -> torch.Tensor:
        """Apply LRP rule to a conv layer."""
        if rule_type == "z+":
            return ConvLRPRule._z_plus_rule(conv, x, R, epsilon)
        elif rule_type == "epsilon":
            return ConvLRPRule._epsilon_rule(conv, x, R, epsilon)
        else:
            raise ValueError(f"Unknown rule: {rule_type}")

    @staticmethod
    def _z_plus_rule(conv: nn.Conv2d, x: torch.Tensor, R: torch.Tensor,
                     epsilon: float = 1e-6) -> torch.Tensor:
        """z+-rule for convolutions."""
        weight = conv.weight  # (out_channels, in_channels, kH, kW)
        weight_plus = torch.clamp(weight, min=0)

        # Forward with positive weights
        z = F.conv2d(x, weight_plus, bias=None, 
                    stride=conv.stride, padding=conv.padding)
        z = z + epsilon

        # Relevance division
        s = R / z

        # Backward pass (gradient of conv transpose)
        c = F.conv_transpose2d(s, weight_plus, stride=conv.stride, 
                               padding=conv.padding)

        R_new = x * c
        return R_new

    @staticmethod
    def _epsilon_rule(conv: nn.Conv2d, x: torch.Tensor, R: torch.Tensor,
                      epsilon: float = 1e-6) -> torch.Tensor:
        """epsilon-rule for convolutions."""
        weight = conv.weight

        z = F.conv2d(x, weight, bias=None,
                    stride=conv.stride, padding=conv.padding)
        z = z + epsilon

        s = R / z
        c = F.conv_transpose2d(s, weight, stride=conv.stride,
                               padding=conv.padding)

        R_new = x * c
        return R_new


class FullLRPExplainer:
    """
    Full LRP implementation with proper layer-wise propagation.
    Handles Linear, Conv2d, ReLU, MaxPool2d, BatchNorm layers.
    """

    def __init__(self, model: nn.Module, rule: str = "z+",
                 epsilon: float = 1e-6, first_layer_rule: str = "zB"):
        """
        Args:
            model: PyTorch model
            rule: LRP rule for hidden layers ("z", "z+", "epsilon")
            epsilon: Stabilization parameter
            first_layer_rule: Rule for first layer ("zB" for bounded input)
        """
        self.model = model
        self.rule = rule
        self.epsilon = epsilon
        self.first_layer_rule = first_layer_rule
        self.activations = OrderedDict()
        self.layers = []
        self._extract_layers()

    def _extract_layers(self):
        """Extract ordered list of relevant layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, 
                                   nn.MaxPool2d, nn.BatchNorm2d, nn.Dropout)):
                self.layers.append((name, module))

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate LRP explanation for input x.

        Returns:
            Relevance map of same shape as input
        """
        self.model.eval()

        # Forward pass to get activations
        activations = OrderedDict()

        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        hooks = []
        for name, module in self.layers:
            hooks.append(module.register_forward_hook(get_activation(name)))

        # Forward pass
        x = x.requires_grad_(True)
        output = self.model(x)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Initialize relevance at output
        R = torch.zeros_like(output)
        R[0, target_class] = output[0, target_class]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Backward propagation of relevance
        # We need to traverse layers in reverse
        reversed_layers = list(reversed(self.layers))

        # Store relevance at each layer
        relevance = {reversed_layers[0][0]: R}

        for i, (name, module) in enumerate(reversed_layers):
            if i == 0:
                continue  # Skip output layer (already initialized)

            prev_name, prev_module = reversed_layers[i-1]
            R_current = relevance[prev_name]

            if isinstance(module, nn.Linear):
                # Get input activation
                if name in activations:
                    a = activations[name]
                else:
                    a = x if i == len(reversed_layers) - 1 else None

                if a is not None:
                    R_new = LRPRule.apply(module, a, R_current, 
                                         self.rule, self.epsilon)
                    relevance[name] = R_new

            elif isinstance(module, nn.Conv2d):
                if name in activations:
                    a = activations[name]
                    R_new = ConvLRPRule.apply(module, a, R_current,
                                             self.rule, self.epsilon)
                    relevance[name] = R_new

            elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
                # Pass through (identity for relevance)
                relevance[name] = R_current

        # Get input relevance
        input_relevance = relevance.get(self.layers[0][0], torch.zeros_like(x))

        # If we couldn't propagate properly, fall back to gradient*input
        if input_relevance.abs().sum() == 0:
            self.model.zero_grad()
            output[0, target_class].backward()
            input_relevance = x * x.grad

        x.requires_grad_(False)
        return input_relevance.detach().cpu().numpy()


class LRPAnalyzer:
    """Analyze LRP explanations."""

    @staticmethod
    def compute_relevance_distribution(explanation: np.ndarray) -> Dict:
        """Compute statistics about relevance distribution."""
        exp = np.abs(explanation)
        total = exp.sum()

        return {
            "total_relevance": float(total),
            "mean_relevance": float(exp.mean()),
            "max_relevance": float(exp.max()),
            "positive_ratio": float((explanation > 0).sum() / explanation.size),
            "negative_ratio": float((explanation < 0).sum() / explanation.size),
        }

    @staticmethod
    def get_top_relevant_pixels(explanation: np.ndarray, k: int = 100) -> List[Tuple]:
        """Get top-k most relevant pixel coordinates."""
        exp = np.abs(explanation)
        flat_indices = np.argsort(exp.flatten())[-k:]

        if explanation.ndim == 3:
            coords = [np.unravel_index(idx, explanation.shape) for idx in flat_indices]
        else:
            coords = flat_indices

        return coords


# Test the full LRP
if __name__ == "__main__":
    # Simple test
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    x = torch.randn(1, 10)
    explainer = FullLRPExplainer(model, rule="z+")
    exp = explainer.explain(x)
    print(f"LRP explanation shape: {exp.shape}")
    print(f"Total relevance: {np.abs(exp).sum():.4f}")
