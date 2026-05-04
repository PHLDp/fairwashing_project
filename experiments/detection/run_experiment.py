"""
Experiment script for Fairwashing Detection Module.
Tests all detection methods on synthetic fairwashing scenarios.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.config import DETECTION_CONFIG, RANDOM_SEED, DEVICE
from src.utils.helpers import set_seed
from src.data.datasets import get_tabular_loaders, get_image_loaders
from src.models.architectures import Conv4, TabularClassifier, BiasedClassifier
from src.explanations.methods import get_explainer
from src.attacks.paper3_off_manifold import OffManifoldFairwasher, create_target_explanation
from src.detection.detector import FairwashingDetector, RuleListFairwashingDetector
from src.evaluation.metrics import ExperimentEvaluator


def create_fairwashed_tabular_model(original_model, X_train, n_epochs=10):
    """Create a fairwashed version of a tabular model."""
    # Clone architecture
    fairwashed = TabularClassifier(
        input_dim=original_model.fc.in_features,
        hidden_dims=[64, 32],
        num_classes=2
    ).to(DEVICE)

    # Train to match original outputs but with different internals
    optimizer = torch.optim.Adam(fairwashed.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    original_model.eval()
    fairwashed.train()

    for epoch in range(n_epochs):
        # Get original predictions
        with torch.no_grad():
            orig_out = original_model(X_train)

        # Train fairwashed to match
        fairwashed_out = fairwashed(X_train)
        loss = criterion(fairwashed_out, orig_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return fairwashed


def create_fairwashed_image_model(teacher, student, train_loader, n_epochs=10):
    """Create a fairwashed image model using off-manifold technique."""
    target_exp = create_target_explanation((1, 28, 28), pattern="center")
    target_exp = target_exp.to(DEVICE)

    fairwasher = OffManifoldFairwasher(
        teacher, student,
        explanation_method="gradient",
        alpha=0.8,
        device=DEVICE
    )

    history = fairwasher.train(
        train_loader, target_exp,
        n_epochs=n_epochs,
        lr=5e-5
    )

    return student, history


def run_tabular_detection_experiment(dataset_name="german_credit"):
    """Run detection on tabular data."""
    print(f"\n{'='*60}")
    print(f"Detection: Tabular - {dataset_name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Load data
    train_loader, test_loader, metadata = get_tabular_loaders(dataset_name, batch_size=128)

    # Get test data
    X_test, y_test = [], []
    for batch in test_loader:
        X_test.append(batch["features"])
        y_test.append(batch["label"])
    X_test = torch.cat(X_test).to(DEVICE)
    y_test = torch.cat(y_test)

    # Create original biased model
    original_model = BiasedClassifier(
        metadata["n_features"],
        sensitive_weight=0.9,
        other_weight=0.1
    ).to(DEVICE)
    original_model.eval()

    # Create fairwashed model
    print("Creating fairwashed model...")
    fairwashed_model = create_fairwashed_tabular_model(original_model, X_test[:500])
    fairwashed_model.eval()

    # Test prediction agreement
    with torch.no_grad():
        orig_preds = original_model(X_test).argmax(dim=1)
        fair_preds = fairwashed_model(X_test).argmax(dim=1)
        agreement = (orig_preds == fair_preds).float().mean().item()

    print(f"Prediction agreement: {agreement:.4f}")

    # Run detector
    print("\nRunning ensemble detector...")
    detector = FairwashingDetector(
        methods=DETECTION_CONFIG["methods"],
        weights=DETECTION_CONFIG["ensemble_weights"],
        thresholds=DETECTION_CONFIG["thresholds"]
    )

    detection_results = detector.detect(
        original_model, fairwashed_model, X_test,
        explanation_methods=["gradient", "xgrad"]
    )

    print(f"\nDetection Results:")
    print(f"  Ensemble score: {detection_results['ensemble_score']:.4f}")
    print(f"  Fairwashing detected: {detection_results['fairwashing_detected']}")

    for method, result in detection_results.items():
        if isinstance(result, dict) and "score" in result:
            print(f"  {method}:")
            print(f"    Score: {result['score']:.4f}")
            for key, val in result.items():
                if key != "score":
                    print(f"    {key}: {val}")

    return detection_results


def run_image_detection_experiment(dataset_name="mnist"):
    """Run detection on image data."""
    print(f"\n{'='*60}")
    print(f"Detection: Image - {dataset_name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Load data
    train_loader, test_loader = get_image_loaders(dataset_name, batch_size=128)

    # Create models
    teacher = Conv4(num_classes=10).to(DEVICE)
    student = Conv4(num_classes=10).to(DEVICE)

    # Brief training for teacher
    print("Training teacher model...")
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = teacher(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx >= 50:
                break

    teacher.eval()

    # Create fairwashed model
    print("Creating fairwashed model...")
    fairwashed_model, history = create_fairwashed_image_model(
        teacher, student, train_loader, n_epochs=5
    )
    fairwashed_model.eval()

    # Get test batch
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(DEVICE)

    # Test prediction agreement
    with torch.no_grad():
        orig_out = teacher(x_test)
        fair_out = fairwashed_model(x_test)
        orig_preds = orig_out.argmax(dim=1)
        fair_preds = fair_out.argmax(dim=1)
        agreement = (orig_preds == fair_preds).float().mean().item()
        output_mse = nn.functional.mse_loss(fair_out, orig_out).item()

    print(f"Prediction agreement: {agreement:.4f}")
    print(f"Output MSE: {output_mse:.6f}")

    # Run detector
    print("\nRunning ensemble detector...")
    detector = FairwashingDetector()

    detection_results = detector.detect(
        teacher, fairwashed_model, x_test,
        explanation_methods=["gradient", "xgrad"]
    )

    print(f"\nDetection Results:")
    print(f"  Ensemble score: {detection_results['ensemble_score']:.4f}")
    print(f"  Fairwashing detected: {detection_results['fairwashing_detected']}")

    for method, result in detection_results.items():
        if isinstance(result, dict) and "score" in result:
            print(f"  {method}: {result['score']:.4f}")

    return detection_results


def run_comprehensive_detection_benchmark():
    """Run comprehensive benchmark across multiple scenarios."""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE DETECTION BENCHMARK")
    print(f"{'='*60}")

    results = {}

    # Tabular scenarios
    for dataset in ["german_credit", "compas"]:
        try:
            results[f"tabular_{dataset}"] = run_tabular_detection_experiment(dataset)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            results[f"tabular_{dataset}"] = {"error": str(e)}

    # Image scenarios
    for dataset in ["mnist"]:
        try:
            results[f"image_{dataset}"] = run_image_detection_experiment(dataset)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            results[f"image_{dataset}"] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for scenario, result in results.items():
        if "error" not in result:
            print(f"\n{scenario}:")
            print(f"  Ensemble score: {result.get('ensemble_score', 'N/A')}")
            print(f"  Detected: {result.get('fairwashing_detected', 'N/A')}")

    # Plot comparison
    scenarios = [k for k, v in results.items() if "error" not in v]
    scores = [results[s].get("ensemble_score", 0) for s in scenarios]

    if scores:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(scenarios)), scores, color=['blue', 'green', 'red'])
        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
        plt.ylabel("Ensemble Detection Score")
        plt.axhline(y=0.5, color='r', linestyle='--', label="Detection Threshold")
        plt.title("Fairwashing Detection Scores Across Scenarios")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/detection_benchmark.png", dpi=150)
        plt.close()

        print(f"\nBenchmark plot saved to results/detection_benchmark.png")

    # Generate summary.txt
    print(f"\nGenerating summary report...")
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("FAIRWASHING DETECTION FRAMEWORK - EXPERIMENT SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append("This project implements and evaluates four fairwashing attack")
    summary_lines.append("papers and a unified detection module.")
    summary_lines.append("")
    summary_lines.append("Papers:")
    summary_lines.append("  1. LaundryML (Aivodji et al., 2019) - Rule list rationalization")
    summary_lines.append("  2. Fooling LIME & SHAP (Slack et al., 2020) - Scaffolding attack")
    summary_lines.append("  3. Off-Manifold Detergent (Anders et al., 2020) - Explanation manipulation")
    summary_lines.append("  4. Fragile Interpretations (Ghorbani et al., 2019) - Adversarial perturbations")
    summary_lines.append("")
    summary_lines.append("-" * 40)
    summary_lines.append("DETECTION BENCHMARK RESULTS")
    summary_lines.append("-" * 40)
    for scenario, result in results.items():
        summary_lines.append(f"")
        summary_lines.append(f"Scenario: {scenario}")
        if "error" in result:
            summary_lines.append(f"  Error: {result['error']}")
        else:
            summary_lines.append(f"  Ensemble Score: {result.get('ensemble_score', 'N/A')}")
            summary_lines.append(f"  Fairwashing Detected: {result.get('fairwashing_detected', 'N/A')}")
    summary_lines.append("")
    summary_lines.append("-" * 40)
    summary_lines.append("GENERATED OUTPUT FILES")
    summary_lines.append("-" * 40)
    summary_lines.append("  results/paper1_pareto.png         - Fidelity vs fairness Pareto plot")
    summary_lines.append("  results/paper1_candidates.csv     - All candidate rule list metrics")
    summary_lines.append("  results/paper2_lime.png           - LIME explanation comparison")
    summary_lines.append("  results/paper2_shap.png           - SHAP feature attribution")
    summary_lines.append("  results/paper4_comparison.png     - Original vs perturbed explanations")
    summary_lines.append("  results/paper4_aggregate.png      - Aggregate attack metrics")
    summary_lines.append("  results/detection_benchmark.png   - Detection scores across scenarios")
    summary_lines.append("  results/summary.txt               - This summary file")
    summary_lines.append("")
    summary_lines.append("Note: Paper 3 (Off-Manifold Detergent) requires GPU and is skipped.")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    with open("results/summary.txt", "w") as f:
        f.write(summary_text)
    print(f"Summary saved to results/summary.txt")
    print(summary_text)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="comprehensive",
                       choices=["tabular", "image", "comprehensive"])
    parser.add_argument("--dataset", default="german_credit")
    args = parser.parse_args()

    if args.mode == "tabular":
        results = run_tabular_detection_experiment(args.dataset)
    elif args.mode == "image":
        results = run_image_detection_experiment(args.dataset)
    else:
        results = run_comprehensive_detection_benchmark()
