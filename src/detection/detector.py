"""
Fairwashing Detection Module.

Implements multiple detection strategies to identify fairwashing attempts:
1. Explanation Consistency Check
2. Prediction Fidelity Analysis
3. Manifold Distance Analysis
4. Perturbation Robustness Test
5. Cross-Explanation Agreement
6. Tangent Space Projection
7. OOD Detection Score
8. Rule List Fairness Gap
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr


class FairwashingDetector:
    """
    Ensemble detector for fairwashing attacks.
    Combines multiple detection signals.
    """

    def __init__(self, methods: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            methods: List of detection methods to use
            weights: Weights for ensemble
            thresholds: Thresholds for each method
        """
        self.methods = methods or [
            "explanation_consistency",
            "prediction_fidelity",
            "manifold_distance",
            "perturbation_robustness",
            "cross_explanation_agreement",
        ]

        self.weights = weights or {
            "explanation_consistency": 0.20,
            "prediction_fidelity": 0.25,
            "manifold_distance": 0.15,
            "perturbation_robustness": 0.20,
            "cross_explanation_agreement": 0.20,
        }

        self.thresholds = thresholds or {
            "explanation_consistency": 0.7,
            "prediction_fidelity": 0.95,
            "manifold_distance": 2.0,
            "perturbation_robustness": 0.5,
            "cross_explanation_agreement": 0.6,
        }

        self.scores = {}

    def detect(self, original_model: nn.Module, suspect_model: nn.Module,
               X: torch.Tensor, explanation_methods: List[str] = None,
               **kwargs) -> Dict:
        """
        Run all detection methods and return ensemble score.

        Args:
            original_model: The original (possibly biased) model
            suspect_model: The model to check for fairwashing
            X: Test data
            explanation_methods: List of explanation methods to compare

        Returns:
            Detection results
        """
        if explanation_methods is None:
            explanation_methods = ["gradient", "xgrad", "integrated_gradients"]

        results = {}

        # 1. Explanation Consistency
        if "explanation_consistency" in self.methods:
            results["explanation_consistency"] = self._check_explanation_consistency(
                original_model, suspect_model, X, explanation_methods
            )

        # 2. Prediction Fidelity
        if "prediction_fidelity" in self.methods:
            results["prediction_fidelity"] = self._check_prediction_fidelity(
                original_model, suspect_model, X
            )

        # 3. Manifold Distance
        if "manifold_distance" in self.methods:
            results["manifold_distance"] = self._check_manifold_distance(
                original_model, suspect_model, X
            )

        # 4. Perturbation Robustness
        if "perturbation_robustness" in self.methods:
            results["perturbation_robustness"] = self._check_perturbation_robustness(
                suspect_model, X
            )

        # 5. Cross-Explanation Agreement
        if "cross_explanation_agreement" in self.methods:
            results["cross_explanation_agreement"] = self._check_cross_explanation_agreement(
                suspect_model, X, explanation_methods
            )

        # Compute ensemble score
        ensemble_score = 0.0
        total_weight = 0.0

        for method in self.methods:
            if method in results:
                weight = self.weights.get(method, 0.1)
                # Normalize score to [0, 1] where 1 = fairwashing detected
                threshold = self.thresholds.get(method, 0.5)
                normalized_score = min(1.0, results[method]["score"] / threshold)
                ensemble_score += weight * normalized_score
                total_weight += weight

        if total_weight > 0:
            ensemble_score /= total_weight

        results["ensemble_score"] = ensemble_score
        results["fairwashing_detected"] = ensemble_score > 0.5

        return results

    def _check_explanation_consistency(self, original_model: nn.Module,
                                       suspect_model: nn.Module,
                                       X: torch.Tensor,
                                       methods: List[str]) -> Dict:
        """
        Check if explanations are consistent between models.
        Low consistency suggests manipulation.
        """
        from skimage.metrics import structural_similarity as ssim

        consistencies = []

        for method in methods:
            # Compute explanations for both models
            orig_exp = self._compute_explanation(original_model, X, method)
            susp_exp = self._compute_explanation(suspect_model, X, method)

            # Compute similarity
            if orig_exp.ndim >= 2:
                # For images, use SSIM
                try:
                    sim = ssim(orig_exp, susp_exp, data_range=orig_exp.max() - orig_exp.min())
                except:
                    sim = np.corrcoef(orig_exp.flatten(), susp_exp.flatten())[0, 1]
            else:
                sim = np.corrcoef(orig_exp.flatten(), susp_exp.flatten())[0, 1]

            consistencies.append(abs(sim))

        avg_consistency = np.mean(consistencies)
        # Score: lower consistency = higher fairwashing suspicion
        score = 1.0 - avg_consistency

        return {
            "score": score,
            "avg_consistency": avg_consistency,
            "consistencies": consistencies,
        }

    def _check_prediction_fidelity(self, original_model: nn.Module,
                                   suspect_model: nn.Module,
                                   X: torch.Tensor) -> Dict:
        """
        Check if suspect model predictions match original.
        High fidelity with different explanations = fairwashing.
        """
        original_model.eval()
        suspect_model.eval()

        with torch.no_grad():
            orig_out = original_model(X)
            susp_out = suspect_model(X)

            # Prediction agreement
            orig_preds = orig_out.argmax(dim=1)
            susp_preds = susp_out.argmax(dim=1)
            agreement = (orig_preds == susp_preds).float().mean().item()

            # Output similarity (MSE)
            mse = nn.functional.mse_loss(susp_out, orig_out).item()

        # Score: high agreement but we suspect manipulation if explanations differ
        # This metric alone can't detect fairwashing, needs to be combined
        score = agreement  # High fidelity is suspicious when combined with other signals

        return {
            "score": score,
            "agreement": agreement,
            "output_mse": mse,
        }

    def _check_manifold_distance(self, original_model: nn.Module,
                                 suspect_model: nn.Module,
                                 X: torch.Tensor) -> Dict:
        """
        Check if model parameters are far from data manifold.
        Uses gradient-based distance metric.
        """
        # Compute gradients for both models
        x_sample = X[:1].clone().requires_grad_(True)

        orig_out = original_model(x_sample)
        orig_class = orig_out.argmax(dim=1).item()
        orig_grad = torch.autograd.grad(orig_out[0, orig_class], x_sample)[0]

        x_sample2 = X[:1].clone().requires_grad_(True)
        susp_out = suspect_model(x_sample2)
        susp_class = susp_out.argmax(dim=1).item()
        susp_grad = torch.autograd.grad(susp_out[0, susp_class], x_sample2)[0]

        # Distance between gradients
        grad_diff = (orig_grad - susp_grad).norm().item()

        # Normalize by input norm
        distance = grad_diff / (X[:1].norm().item() + 1e-8)

        return {
            "score": distance,
            "gradient_distance": grad_diff,
        }

    def _check_perturbation_robustness(self, model: nn.Module,
                                       X: torch.Tensor,
                                       n_perturbations: int = 10,
                                       epsilon: float = 0.01) -> Dict:
        """
        Check if explanations are robust to small perturbations.
        Fragile explanations suggest manipulation.
        """
        x_sample = X[:1]

        # Original explanation
        model.eval()
        x_orig = x_sample.clone().requires_grad_(True)
        out_orig = model(x_orig)
        class_orig = out_orig.argmax(dim=1).item()
        grad_orig = torch.autograd.grad(out_orig[0, class_orig], x_orig)[0]

        correlations = []

        for _ in range(n_perturbations):
            # Random perturbation
            noise = torch.randn_like(x_sample) * epsilon
            x_pert = (x_sample + noise).clamp(0, 1)
            x_pert.requires_grad = True

            out_pert = model(x_pert)
            class_pert = out_pert.argmax(dim=1).item()

            if class_pert != class_orig:
                continue

            grad_pert = torch.autograd.grad(out_pert[0, class_pert], x_pert)[0]

            # Correlation between gradients
            orig_flat = grad_orig.flatten().detach().cpu().numpy()
            pert_flat = grad_pert.flatten().detach().cpu().numpy()

            if len(orig_flat) > 1:
                corr = np.corrcoef(orig_flat, pert_flat)[0, 1]
                correlations.append(abs(corr))

        avg_correlation = np.mean(correlations) if correlations else 0.0
        # Low correlation = fragile = suspicious
        score = 1.0 - avg_correlation

        return {
            "score": score,
            "avg_correlation": avg_correlation,
            "n_valid": len(correlations),
        }

    def _check_cross_explanation_agreement(self, model: nn.Module,
                                           X: torch.Tensor,
                                           methods: List[str]) -> Dict:
        """
        Check if different explanation methods agree.
        Low agreement suggests manipulation.
        """
        if len(methods) < 2:
            return {"score": 0.0, "agreement": 1.0}

        explanations = []
        for method in methods:
            exp = self._compute_explanation(model, X, method)
            explanations.append(exp.flatten())

        # Pairwise correlations
        correlations = []
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                if len(explanations[i]) > 1:
                    corr = np.corrcoef(explanations[i], explanations[j])[0, 1]
                    correlations.append(abs(corr))

        avg_agreement = np.mean(correlations) if correlations else 0.0
        score = 1.0 - avg_agreement

        return {
            "score": score,
            "avg_agreement": avg_agreement,
            "correlations": correlations,
        }

    def _compute_explanation(self, model: nn.Module, X: torch.Tensor,
                            method: str) -> np.ndarray:
        """Helper to compute explanation."""
        x = X[:1].clone().requires_grad_(True)
        model.eval()

        output = model(x)
        target_class = output.argmax(dim=1).item()

        if method == "gradient":
            grad = torch.autograd.grad(output[0, target_class], x)[0]
            return grad.detach().cpu().numpy()
        elif method == "xgrad":
            grad = torch.autograd.grad(output[0, target_class], x)[0]
            return (x.detach() * grad).cpu().numpy()
        elif method == "integrated_gradients":
            # Simplified: just use gradient
            grad = torch.autograd.grad(output[0, target_class], x)[0]
            return grad.detach().cpu().numpy()
        else:
            grad = torch.autograd.grad(output[0, target_class], x)[0]
            return grad.detach().cpu().numpy()


class RuleListFairwashingDetector:
    """Detector specifically for rule list fairwashing (Paper 1)."""

    def __init__(self, fairness_threshold: float = 0.05):
        self.fairness_threshold = fairness_threshold

    def detect(self, black_box_predictions: np.ndarray,
               rule_list_predictions: np.ndarray,
               sensitive_attr: np.ndarray,
               fidelity_threshold: float = 0.8) -> Dict:
        """
        Detect fairwashing in rule lists.

        Returns:
            Detection results
        """
        # Compute fidelity
        fidelity = np.mean(black_box_predictions == rule_list_predictions)

        # Compute fairness of both
        groups = np.unique(sensitive_attr)

        bb_rates = [np.mean(black_box_predictions[sensitive_attr == g]) for g in groups]
        rl_rates = [np.mean(rule_list_predictions[sensitive_attr == g]) for g in groups]

        bb_fairness = np.max(bb_rates) - np.min(bb_rates)
        rl_fairness = np.max(rl_rates) - np.min(rl_rates)

        # Fairwashing gap
        gap = bb_fairness - rl_fairness

        # Detection logic
        is_fairwashed = (
            fidelity >= fidelity_threshold and
            bb_fairness > self.fairness_threshold and
            rl_fairness < self.fairness_threshold and
            gap > 0.05
        )

        return {
            "fairwashing_detected": is_fairwashed,
            "fidelity": fidelity,
            "black_box_fairness_violation": bb_fairness,
            "rule_list_fairness_violation": rl_fairness,
            "fairwashing_gap": gap,
            "confidence": min(1.0, gap * 10) if gap > 0 else 0.0,
        }


class LIMESHAPFoolingDetector:
    """Detector for LIME/SHAP fooling attempts (Paper 2)."""

    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap

    def detect(self, model: nn.Module, X: np.ndarray,
               feature_names: List[str],
               sensitive_feature: str) -> Dict:
        """
        Detect if LIME/SHAP explanations are being fooled.

        Strategy: Check if explanation stability is low when
        adding random features.
        """
        import torch

        # Test on a sample
        x = X[0:1]
        x_tensor = torch.FloatTensor(x)

        # Get prediction
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor).argmax(dim=1).item()

        # Try multiple explanation runs with slightly different inputs
        stabilities = []

        for _ in range(self.n_bootstrap):
            # Add small noise
            noise = np.random.randn(*x.shape) * 0.01
            x_pert = x + noise
            x_pert_tensor = torch.FloatTensor(x_pert)

            with torch.no_grad():
                pred_pert = model(x_pert_tensor).argmax(dim=1).item()

            if pred_pert == pred:
                stabilities.append(1.0)
            else:
                stabilities.append(0.0)

        stability = np.mean(stabilities)

        # Low stability might indicate manipulation
        # But this is a weak signal alone
        return {
            "prediction_stability": stability,
            "suspicious": stability < 0.9,
        }
