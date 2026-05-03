"""
Paper 2: Fooling LIME and SHAP - Adversarial Attacks on Post hoc Explanation Methods
(Slack et al., AIES 2020)

This module implements:
1. Scaffolding attack to fool LIME/SHAP
2. OOD detection using perturbations
3. Decoy feature generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class OODDetector:
    """
    Out-of-Distribution detector using perturbation-based method.
    From Section 4.2 of the paper.
    """

    def __init__(self, n_clusters: int = 10, perturbation_std: float = 0.1):
        self.n_clusters = n_clusters
        self.perturbation_std = perturbation_std
        self.kmeans = None
        self.threshold = None

    def fit(self, X: np.ndarray):
        """Fit OOD detector on training data."""
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)

        # Compute distances to cluster centers
        distances = self.kmeans.transform(X)
        min_distances = np.min(distances, axis=1)

        # Set threshold as 95th percentile
        self.threshold = np.percentile(min_distances, 95)

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        """Check if samples are OOD."""
        if self.kmeans is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        distances = self.kmeans.transform(X)
        min_distances = np.min(distances, axis=1)
        return min_distances > self.threshold

    def compute_ood_score(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD score (higher = more OOD)."""
        distances = self.kmeans.transform(X)
        min_distances = np.min(distances, axis=1)
        return min_distances


class ScaffoldingAttack:
    """
    Scaffolding attack to fool LIME and SHAP explanations.

    The attack adds uncorrelated decoy features that the explanation
    methods incorrectly attribute importance to, hiding the true
    bias in the black-box model.
    """

    def __init__(self, base_classifier: nn.Module, 
                 n_decoy_features: int = 2,
                 decoy_std: float = 0.1,
                 fidelity_threshold: float = 0.95):
        """
        Args:
            base_classifier: The biased classifier to hide
            n_decoy_features: Number of decoy features to add
            decoy_std: Standard deviation of decoy features
            fidelity_threshold: Minimum fidelity to base classifier
        """
        self.base_classifier = base_classifier
        self.n_decoy_features = n_decoy_features
        self.decoy_std = decoy_std
        self.fidelity_threshold = fidelity_threshold

        # Freeze base classifier
        for param in self.base_classifier.parameters():
            param.requires_grad = False

    def add_decoy_features(self, X: torch.Tensor) -> torch.Tensor:
        """Add uncorrelated decoy features to input."""
        batch_size = X.shape[0]
        decoy = torch.randn(batch_size, self.n_decoy_features, 
                           device=X.device) * self.decoy_std
        return torch.cat([X, decoy], dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using scaffolded classifier."""
        # Only use original features for prediction
        X_orig = X[:, :-self.n_decoy_features] if self.n_decoy_features > 0 else X
        return self.base_classifier(X_orig)

    def evaluate_fidelity(self, X: torch.Tensor) -> float:
        """Check that scaffolded predictions match base classifier."""
        with torch.no_grad():
            base_preds = self.base_classifier(X).argmax(dim=1)
            scaffold_preds = self.predict(
                self.add_decoy_features(X)
            ).argmax(dim=1)
            fidelity = (base_preds == scaffold_preds).float().mean().item()
        return fidelity

    def fool_lime_explanation(self, X_train: np.ndarray, 
                             x_instance: np.ndarray,
                             feature_names: List[str],
                             num_samples: int = 5000) -> Dict:
        """
        Demonstrate how LIME is fooled by decoy features.

        Returns:
            Dictionary with original and fooled explanations
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")

        # Add decoy features to training data
        X_train_augmented = np.concatenate([
            X_train,
            np.random.randn(X_train.shape[0], self.n_decoy_features) * self.decoy_std
        ], axis=1)

        # Add decoy features to instance
        x_augmented = np.concatenate([
            x_instance,
            np.random.randn(self.n_decoy_features) * self.decoy_std
        ])

        # Extended feature names
        augmented_features = feature_names + [f"decoy_{i}" for i in range(self.n_decoy_features)]

        # Create explainer on augmented data
        explainer = LimeTabularExplainer(
            X_train_augmented,
            feature_names=augmented_features,
            class_names=["class_0", "class_1"],
            discretize_continuous=True
        )

        def predict_fn(X_aug):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_aug)
                # Only use original features for prediction
                X_orig = X_tensor[:, :-self.n_decoy_features]
                output = self.base_classifier(X_orig)
                probs = F.softmax(output, dim=1)
                return probs.numpy()

        # Get explanation
        exp = explainer.explain_instance(
            x_augmented,
            predict_fn,
            num_features=len(augmented_features),
            top_labels=1
        )

        # Parse explanations
        explanation_dict = {}
        for feature, weight in exp.as_list(label=exp.available_labels()[0]):
            explanation_dict[feature] = weight

        return {
            "explanation": explanation_dict,
            "decoy_features": [f"decoy_{i}" for i in range(self.n_decoy_features)],
            "n_decoy_features": self.n_decoy_features,
        }

    def fool_shap_explanation(self, X_background: np.ndarray,
                             x_instance: np.ndarray) -> Dict:
        """
        Demonstrate how SHAP is fooled by decoy features.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        # Add decoy features
        X_bg_aug = np.concatenate([
            X_background,
            np.random.randn(X_background.shape[0], self.n_decoy_features) * self.decoy_std
        ], axis=1)

        x_aug = np.concatenate([
            x_instance,
            np.random.randn(self.n_decoy_features) * self.decoy_std
        ])

        def predict_fn(X_aug):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_aug)
                X_orig = X_tensor[:, :-self.n_decoy_features]
                output = self.base_classifier(X_orig)
                probs = F.softmax(output, dim=1)
                return probs.numpy()

        explainer = shap.KernelExplainer(predict_fn, X_bg_aug)
        shap_values = explainer.shap_values(x_aug.reshape(1, -1), nsamples=100)

        return {
            "shap_values": shap_values,
            "n_features": len(x_aug),
            "n_decoy": self.n_decoy_features,
        }


class BiasedClassifierFactory:
    """Factory to create intentionally biased classifiers for experiments."""

    @staticmethod
    def create_gender_biased_classifier(input_dim: int, 
                                       gender_idx: int = 0,
                                       bias_strength: float = 0.9) -> nn.Module:
        """Create a classifier heavily biased on gender feature."""
        model = nn.Linear(input_dim, 2)

        with torch.no_grad():
            # Set large weight on gender feature
            weight = torch.zeros(2, input_dim)
            weight[:, gender_idx] = torch.tensor([bias_strength, -bias_strength])
            # Small random weights for other features
            weight[:, [i for i in range(input_dim) if i != gender_idx]] =                 torch.randn(2, input_dim - 1) * 0.1
            model.weight.copy_(weight)
            model.bias.zero_()

        return model

    @staticmethod
    def create_race_biased_classifier(input_dim: int,
                                     race_idx: int = 0,
                                     bias_strength: float = 0.9) -> nn.Module:
        """Create a classifier heavily biased on race feature."""
        return BiasedClassifierFactory.create_gender_biased_classifier(
            input_dim, race_idx, bias_strength
        )


def run_scaffolding_experiment(X_train, y_train, X_test, sensitive_idx: int,
                               feature_names: List[str],
                               n_decoy: int = 2) -> Dict:
    """
    Run complete scaffolding experiment.

    Returns:
        Experiment results showing LIME/SHAP fooling
    """
    input_dim = X_train.shape[1]

    # Create biased classifier
    biased_model = BiasedClassifierFactory.create_gender_biased_classifier(
        input_dim, sensitive_idx, bias_strength=0.9
    )

    # Create scaffolding attack
    attack = ScaffoldingAttack(biased_model, n_decoy_features=n_decoy)

    # Test on a sample instance
    x_instance = X_test[0]

    # Fool LIME
    lime_result = attack.fool_lime_explanation(X_train, x_instance, feature_names)

    # Fool SHAP
    shap_result = attack.fool_shap_explanation(X_train[:100], x_instance)

    # Check fidelity
    X_test_tensor = torch.FloatTensor(X_test)
    fidelity = attack.evaluate_fidelity(X_test_tensor)

    return {
        "lime_fooling": lime_result,
        "shap_fooling": shap_result,
        "fidelity": fidelity,
        "n_decoy_features": n_decoy,
        "biased_on_feature": feature_names[sensitive_idx] if sensitive_idx < len(feature_names) else f"feature_{sensitive_idx}",
    }
