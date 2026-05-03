"""
Experiment script for Paper 2: Fooling LIME and SHAP
Adversarial Attacks on Post hoc Explanation Methods
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import PAPER2_CONFIG, RANDOM_SEED, DEVICE
from src.utils.helpers import set_seed
from src.data.datasets import get_tabular_loaders
from src.models.architectures import BiasedClassifier
from src.attacks.paper2_fooling_lime_shap import (
    ScaffoldingAttack, BiasedClassifierFactory, OODDetector,
    run_scaffolding_experiment
)
from src.detection.detector import LIMESHAPFoolingDetector
from src.evaluation.metrics import ExperimentEvaluator


def run_experiment(dataset_name="compas", n_decoy=2):
    """Run complete Paper 2 experiment."""
    print(f"\n{'='*60}")
    print(f"Paper 2: Fooling LIME and SHAP - {dataset_name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # Load data
    train_loader, test_loader, metadata = get_tabular_loaders(dataset_name, batch_size=256)

    # Extract numpy arrays
    X_train, y_train, s_train = [], [], []
    for batch in train_loader:
        X_train.append(batch["features"].numpy())
        y_train.append(batch["label"].numpy())
        s_train.append(batch["sensitive"].numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    s_train = np.concatenate(s_train)

    X_test, y_test, s_test = [], [], []
    for batch in test_loader:
        X_test.append(batch["features"].numpy())
        y_test.append(batch["label"].numpy())
        s_test.append(batch["sensitive"].numpy())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    s_test = np.concatenate(s_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Feature names
    feature_names = [f"feature_{i}" for i in range(metadata["n_features"])]

    # Create biased classifier
    sensitive_idx = 0
    biased_model = BiasedClassifierFactory.create_gender_biased_classifier(
        metadata["n_features"], sensitive_idx, bias_strength=0.9
    )

    # Test biased model
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        biased_preds = biased_model(X_test_tensor).argmax(dim=1).numpy()

    biased_accuracy = np.mean(biased_preds == y_test)
    print(f"\nBiased model accuracy: {biased_accuracy:.4f}")

    # Compute bias
    from src.utils.helpers import compute_demographic_parity
    bias = compute_demographic_parity(biased_preds, s_test)
    print(f"Biased model fairness violation: {bias:.4f}")

    # Create scaffolding attack
    print(f"\nCreating scaffolding attack with {n_decoy} decoy features...")
    attack = ScaffoldingAttack(biased_model, n_decoy_features=n_decoy)

    # Test fidelity
    fidelity = attack.evaluate_fidelity(X_test_tensor)
    print(f"Scaffolding fidelity: {fidelity:.4f}")

    # Fool LIME on a sample instance
    print(f"\nFooling LIME explanation...")
    try:
        lime_result = attack.fool_lime_explanation(
            X_train, X_test[0], feature_names
        )

        print(f"LIME fooled successfully!")
        print(f"Decoy features in explanation:")
        for feature, weight in list(lime_result["explanation"].items())[:10]:
            marker = " [DECOY]" if "decoy" in str(feature) else ""
            print(f"  {feature}: {weight:.4f}{marker}")
    except Exception as e:
        print(f"LIME fooling failed: {e}")
        lime_result = None

    # Fool SHAP on a sample instance
    print(f"\nFooling SHAP explanation...")
    try:
        shap_result = attack.fool_shap_explanation(X_train[:100], X_test[0])
        print(f"SHAP fooled successfully!")
    except Exception as e:
        print(f"SHAP fooling failed: {e}")
        shap_result = None

    # OOD Detection
    print(f"\nTesting OOD detection...")
    ood_detector = OODDetector(n_clusters=10)
    ood_detector.fit(X_train)

    # Normal data
    ood_scores_normal = ood_detector.compute_ood_score(X_test[:50])
    # OOD data (perturbed)
    X_ood = X_test[:50] + np.random.randn(50, X_test.shape[1]) * 0.5
    ood_scores_ood = ood_detector.compute_ood_score(X_ood)

    print(f"Normal data OOD score: {ood_scores_normal.mean():.4f} (+/- {ood_scores_normal.std():.4f})")
    print(f"OOD data OOD score: {ood_scores_ood.mean():.4f} (+/- {ood_scores_ood.std():.4f})")

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper2(
        lime_result, shap_result, fidelity, save_prefix="paper2"
    )

    # Detection
    print(f"\nRunning LIME/SHAP fooling detection...")
    detector = LIMESHAPFoolingDetector(n_bootstrap=50)
    detection = detector.detect(biased_model, X_test, feature_names, "feature_0")
    print(f"Prediction stability: {detection['prediction_stability']:.4f}")
    print(f"Suspicious: {detection['suspicious']}")

    return {
        "metrics": metrics,
        "lime_result": lime_result,
        "shap_result": shap_result,
        "fidelity": fidelity,
        "ood_scores": {
            "normal_mean": ood_scores_normal.mean(),
            "ood_mean": ood_scores_ood.mean(),
        },
        "detection": detection,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="compas")
    parser.add_argument("--n_decoy", type=int, default=2)
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.n_decoy)
