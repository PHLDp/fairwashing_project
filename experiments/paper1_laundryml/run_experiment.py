"""
Experiment script for Paper 1: LaundryML
Fairwashing: The Risk of Rationalization
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.config import PAPER1_CONFIG, RANDOM_SEED, DEVICE
from src.utils.helpers import set_seed, compute_demographic_parity
from src.data.datasets import get_tabular_loaders
from src.models.architectures import BiasedClassifier
from src.attacks.paper1_laundryml import LaundryML, RuleList
from src.detection.detector import RuleListFairwashingDetector
from src.evaluation.metrics import ExperimentEvaluator


def create_biased_black_box(input_dim, sensitive_idx=0, bias_strength=0.9):
    """Create a biased classifier."""
    model = BiasedClassifier(input_dim, sensitive_weight=bias_strength, 
                            other_weight=0.1, num_classes=2)
    # Set weights manually
    with torch.no_grad():
        weight = torch.zeros(2, input_dim)
        weight[:, sensitive_idx] = torch.tensor([bias_strength, -bias_strength])
        weight[:, [i for i in range(input_dim) if i != sensitive_idx]] =             torch.randn(2, input_dim - 1) * 0.1
        model.fc.weight.copy_(weight)
        model.fc.bias.zero_()
    return model


def run_experiment(dataset_name="adult_income", n_candidates=50):
    """Run complete Paper 1 experiment."""
    print(f"\n{'='*60}")
    print(f"Paper 1: LaundryML - {dataset_name}")
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
    print(f"Sensitive attr distribution: {np.bincount(s_train)}")

    # Create biased black-box
    sensitive_idx = 0  # Assume first feature is sensitive after standardization
    black_box = create_biased_black_box(metadata["n_features"], sensitive_idx)
    black_box.eval()

    # Get black-box predictions
    with torch.no_grad():
        bb_train_preds = black_box(torch.FloatTensor(X_train)).argmax(dim=1).numpy()
        bb_test_preds = black_box(torch.FloatTensor(X_test)).argmax(dim=1).numpy()

    bb_train_fairness = compute_demographic_parity(bb_train_preds, s_train)
    bb_test_fairness = compute_demographic_parity(bb_test_preds, s_test)

    print(f"\nBlack-box fairness violation (train): {bb_train_fairness:.4f}")
    print(f"Black-box fairness violation (test): {bb_test_fairness:.4f}")
    print(f"Black-box accuracy (train): {accuracy_score(y_train, bb_train_preds):.4f}")
    print(f"Black-box accuracy (test): {accuracy_score(y_test, bb_test_preds):.4f}")

    # Create DataFrames
    feature_names = [f"f{i}" for i in range(metadata["n_features"])]
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["sensitive"] = s_train
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["sensitive"] = s_test

    # Run LaundryML
    print(f"\nEnumerating {n_candidates} rule list candidates...")
    laundry = LaundryML(black_box, "sensitive", fairness_metric="demographic_parity")

    candidates = laundry.enumerate_rule_lists(
        df_train, y_train,
        beta_values=PAPER1_CONFIG["beta_range"],
        n_models=n_candidates
    )

    print(f"Generated {len(candidates)} candidates")

    # Analyze on test set
    analysis = laundry.analyze_fairwashing_gap(df_test, y_test)

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper1(candidates, bb_test_fairness, save_prefix="paper1")

    # Detection
    detector = RuleListFairwashingDetector()
    best = laundry.find_best_rationalization()

    if best:
        rl_test_preds = best["rule_list"].predict(df_test)
        detection = detector.detect(bb_test_preds, rl_test_preds, s_test)

        print(f"\nBest Rationalization:")
        print(f"  Fidelity: {best['fidelity']:.4f}")
        print(f"  Fairness violation: {best['fairness_violation']:.4f}")
        print(f"  Objective: {best['objective']:.4f}")
        print(f"\nDetection:")
        print(f"  Fairwashing detected: {detection['fairwashing_detected']}")
        print(f"  Fairwashing gap: {detection['fairwashing_gap']:.4f}")
        print(f"  Confidence: {detection['confidence']:.4f}")

    # Print some rule lists
    print(f"\nSample Rule Lists (top 3 by objective):")
    sorted_candidates = sorted(candidates, key=lambda c: c["objective"], reverse=True)[:3]
    for i, c in enumerate(sorted_candidates):
        print(f"\nCandidate {i+1} (fidelity={c['fidelity']:.3f}, fairness={c['fairness_violation']:.3f}):")
        print(f"  {c['rule_list']}")

    return {
        "candidates": candidates,
        "metrics": metrics,
        "detection": detection if best else {},
        "black_box_fairness": bb_test_fairness,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="adult_income")
    parser.add_argument("--n_candidates", type=int, default=50)
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.n_candidates)
