"""
Main pipeline for Fairwashing Detection Framework.
Runs all four papers' experiments and detection.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import *
from src.utils.helpers import set_seed
from src.data.datasets import get_tabular_loaders, get_image_loaders
from src.models.architectures import get_model
from src.explanations.methods import get_explainer
from src.attacks.paper1_laundryml import LaundryML, run_laundryml_experiment
from src.attacks.paper2_fooling_lime_shap import ScaffoldingAttack, BiasedClassifierFactory, run_scaffolding_experiment
from src.attacks.paper3_off_manifold import OffManifoldFairwasher, TangentSpaceProjector, TSPDefender, create_target_explanation
from src.attacks.paper4_fragile_interpretation import TopKAttack, MassCenterAttack, RandomSignPerturbation, compute_interpretation_metrics
from src.detection.detector import FairwashingDetector, RuleListFairwashingDetector, LIMESHAPFoolingDetector
from src.evaluation.metrics import ExperimentEvaluator


def run_paper1_experiment(dataset_name: str = "adult_income") -> Dict:
    """
    Run Paper 1 experiment: LaundryML fairwashing.

    Steps:
    1. Load biased black-box model
    2. Enumerate rule lists with different fairness-accuracy trade-offs
    3. Find rationalizations that appear fair
    4. Detect fairwashing gap
    """
    print("
" + "="*60)
    print("PAPER 1: LaundryML - Fairwashing: The Risk of Rationalization")
    print("="*60)

    set_seed(RANDOM_SEED)

    # Load data
    train_loader, test_loader, metadata = get_tabular_loaders(dataset_name, batch_size=64)

    # Get full data for rule list training
    X_train = []
    y_train = []
    s_train = []
    for batch in train_loader:
        X_train.append(batch["features"].numpy())
        y_train.append(batch["label"].numpy())
        s_train.append(batch["sensitive"].numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    s_train = np.concatenate(s_train)

    X_test = []
    y_test = []
    s_test = []
    for batch in test_loader:
        X_test.append(batch["features"].numpy())
        y_test.append(batch["label"].numpy())
        s_test.append(batch["sensitive"].numpy())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    s_test = np.concatenate(s_test)

    # Create biased black-box model
    black_box = get_model("biased", input_dim=metadata["n_features"], 
                         sensitive_weight=0.9, other_weight=0.1)

    # Create DataFrames for rule list
    import pandas as pd
    feature_names = [f"f{i}" for i in range(metadata["n_features"])]
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["sensitive"] = s_train
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["sensitive"] = s_test

    # Run LaundryML
    sensitive_attr = "sensitive"
    results = run_laundryml_experiment(
        df_train, y_train, df_test, y_test,
        black_box, sensitive_attr,
        beta_values=PAPER1_CONFIG["beta_range"],
    )

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")

    # Compute black-box fairness
    bb_preds = black_box(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
    from src.utils.helpers import compute_demographic_parity
    bb_fairness = compute_demographic_parity(bb_preds, s_test)

    metrics = evaluator.evaluate_paper1(
        results["candidates"], bb_fairness, save_prefix="paper1"
    )

    # Detection
    detector = RuleListFairwashingDetector()

    best_candidate = min(results["candidates"], 
                        key=lambda c: c["fairness_violation"])
    rl_preds = best_candidate["rule_list"].predict(df_test)

    detection = detector.detect(
        bb_preds, rl_preds, s_test
    )

    print(f"
Paper 1 Results:")
    print(f"  Candidates generated: {metrics['n_candidates']}")
    print(f"  Best fidelity: {metrics['best_fidelity']:.3f}")
    print(f"  Best fairness violation: {metrics['best_fairness']:.3f}")
    print(f"  Fairwashing gap: {metrics['fairwashing_gap']:.3f}")
    print(f"  Fairwashing detected: {detection['fairwashing_detected']}")

    return {
        "metrics": metrics,
        "detection": detection,
        "results": results,
    }


def run_paper2_experiment(dataset_name: str = "compas") -> Dict:
    """
    Run Paper 2 experiment: Fooling LIME and SHAP.

    Steps:
    1. Create biased classifier
    2. Add decoy features
    3. Show LIME/SHAP are fooled
    4. Demonstrate OOD detection
    """
    print("
" + "="*60)
    print("PAPER 2: Fooling LIME and SHAP")
    print("="*60)

    set_seed(RANDOM_SEED)

    # Load data
    train_loader, test_loader, metadata = get_tabular_loaders(dataset_name, batch_size=64)

    # Get numpy arrays
    X_train = []
    for batch in train_loader:
        X_train.append(batch["features"].numpy())
    X_train = np.vstack(X_train)

    X_test = []
    for batch in test_loader:
        X_test.append(batch["features"].numpy())
    X_test = np.vstack(X_test)

    # Feature names
    feature_names = [f"feature_{i}" for i in range(metadata["n_features"])]

    # Run scaffolding experiment
    sensitive_idx = 0  # Assume first feature is sensitive
    results = run_scaffolding_experiment(
        X_train, None, X_test, sensitive_idx, feature_names, n_decoy=2
    )

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper2(
        results.get("lime_fooling"),
        results.get("shap_fooling"),
        results["fidelity"],
        save_prefix="paper2"
    )

    print(f"
Paper 2 Results:")
    print(f"  Fidelity: {results['fidelity']:.3f}")
    print(f"  Decoy features: {results['n_decoy_features']}")
    print(f"  LIME fooled: {metrics['lime_fooled']}")
    print(f"  SHAP fooled: {metrics['shap_fooled']}")

    return {
        "metrics": metrics,
        "results": results,
    }


def run_paper3_experiment(dataset_name: str = "fashion_mnist") -> Dict:
    """
    Run Paper 3 experiment: Off-Manifold Fairwashing.

    Steps:
    1. Train teacher model on image data
    2. Train student model to match outputs but different explanations
    3. Evaluate explanation manipulation
    4. Test TSP defense
    """
    print("
" + "="*60)
    print("PAPER 3: Fairwashing with Off-Manifold Detergent")
    print("="*60)

    set_seed(RANDOM_SEED)

    # Load data
    train_loader, test_loader = get_image_loaders(dataset_name, batch_size=128)

    # Get a batch for quick experiment
    x_batch, y_batch = next(iter(test_loader))

    # Create models
    if dataset_name in ["mnist", "fashion_mnist"]:
        teacher = get_model("conv4", num_classes=10)
        student = get_model("conv4", num_classes=10)
    else:
        teacher = get_model("vgg16", num_classes=10)
        student = get_model("vgg16", num_classes=10)

    # Create fairwasher
    fairwasher = OffManifoldFairwasher(
        teacher, student, 
        explanation_method="gradient",
        alpha=0.8,
        device=DEVICE
    )

    # Create target explanation (shape of "42")
    target_exp = create_target_explanation(
        x_batch[0].shape, pattern="number_42"
    )

    # Train (reduced epochs for demo)
    print("Training fairwashed model...")
    history = fairwasher.train(
        train_loader, target_exp,
        n_epochs=10,  # Reduced for demo
        lr=5e-5
    )

    # Evaluate
    eval_results = fairwasher.evaluate(test_loader, target_exp)

    # Test TSP defense
    print("Computing tangent space projector...")
    # Get data for projector
    X_data = []
    for batch in train_loader:
        X_data.append(batch[0].numpy())
        if len(X_data) > 10:  # Limit for demo
            break
    X_data = np.vstack(X_data)
    X_flat = X_data.reshape(X_data.shape[0], -1)

    projector = TangentSpaceProjector(n_neighbors=16, n_components=8)
    projector.fit(X_flat)

    tsp_defender = TSPDefender(projector)

    # Evaluate with TSP
    x_test = x_batch[:1].to(DEVICE)
    target_class = teacher(x_test).argmax(dim=1).item()

    tsp_exp = tsp_defender.compute_tsp_explanation(student, x_test, target_class)

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper3(
        history, eval_results, save_prefix="paper3"
    )

    print(f"
Paper 3 Results:")
    print(f"  Final output loss: {history[-1]['output_loss']:.4f}")
    print(f"  Final explanation loss: {history[-1]['explanation_loss']:.4f}")
    print(f"  Test output MSE: {eval_results['output_mse']:.4f}")
    print(f"  Test explanation MSE: {eval_results['explanation_mse']:.4f}")

    return {
        "metrics": metrics,
        "history": history,
        "evaluation": eval_results,
    }


def run_paper4_experiment(dataset_name: str = "mnist") -> Dict:
    """
    Run Paper 4 experiment: Fragile Interpretation.

    Steps:
    1. Load model
    2. Apply top-k, mass-center, and random attacks
    3. Measure explanation fragility
    """
    print("
" + "="*60)
    print("PAPER 4: Interpretation of Neural Networks is Fragile")
    print("="*60)

    set_seed(RANDOM_SEED)

    # Load data
    _, test_loader = get_image_loaders(dataset_name, batch_size=1)

    # Load model
    model = get_model("conv4", num_classes=10)
    model.eval()

    # Get a test sample
    x, y = next(iter(test_loader))
    x = x.to(DEVICE)

    # Original prediction and explanation
    with torch.no_grad():
        orig_pred = model(x).argmax(dim=1).item()

    x_grad = x.clone().requires_grad_(True)
    output = model(x_grad)
    orig_exp = torch.autograd.grad(output[0, orig_pred], x_grad)[0]
    orig_exp_np = orig_exp.detach().cpu().numpy()

    print(f"Original prediction: {orig_pred}")

    results = {}

    # Random sign perturbation
    print("Applying random sign perturbation...")
    random_attack = RandomSignPerturbation(model, epsilon=8.0, device=DEVICE)
    x_random = random_attack.attack(x)

    with torch.no_grad():
        random_pred = model(x_random).argmax(dim=1).item()

    x_grad_r = x_random.clone().requires_grad_(True)
    output_r = model(x_grad_r)
    random_exp = torch.autograd.grad(output_r[0, random_pred], x_grad_r)[0]
    random_exp_np = random_exp.detach().cpu().numpy()

    random_metrics = compute_interpretation_metrics(
        orig_exp_np, random_exp_np, k=100
    )
    results["random"] = random_metrics

    # Top-k attack (simplified - fewer iterations for demo)
    print("Applying top-k attack...")
    topk_attack = TopKAttack(model, k=100, epsilon=8.0, 
                             n_iterations=50, device=DEVICE)
    x_topk = topk_attack.attack(x, explanation_method="gradient")

    with torch.no_grad():
        topk_pred = model(x_topk).argmax(dim=1).item()

    x_grad_t = x_topk.clone().requires_grad_(True)
    output_t = model(x_grad_t)
    topk_exp = torch.autograd.grad(output_t[0, topk_pred], x_grad_t)[0]
    topk_exp_np = topk_exp.detach().cpu().numpy()

    topk_metrics = compute_interpretation_metrics(
        orig_exp_np, topk_exp_np, k=100
    )
    results["top_k"] = topk_metrics

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper4(
        x.cpu().numpy(),
        x_topk.cpu().numpy(),
        orig_exp_np,
        topk_exp_np,
        topk_metrics,
        save_prefix="paper4"
    )

    print(f"
Paper 4 Results:")
    print(f"  Random - Spearman: {random_metrics['spearman_correlation']:.3f}")
    print(f"  Random - Top-k intersection: {random_metrics['top_k_intersection']:.3f}")
    print(f"  Top-k attack - Spearman: {topk_metrics['spearman_correlation']:.3f}")
    print(f"  Top-k attack - Top-k intersection: {topk_metrics['top_k_intersection']:.3f}")
    print(f"  Prediction preserved: {orig_pred == topk_pred}")

    return {
        "metrics": metrics,
        "random_metrics": random_metrics,
        "topk_metrics": topk_metrics,
        "results": results,
    }


def run_detection_experiment() -> Dict:
    """
    Run detection experiments across all papers.
    """
    print("
" + "="*60)
    print("DETECTION MODULE")
    print("="*60)

    set_seed(RANDOM_SEED)

    # Create a simple test case
    # Original biased model vs fairwashed model

    # Load small dataset
    _, test_loader, metadata = get_tabular_loaders("german_credit", batch_size=32)

    X_test = []
    for batch in test_loader:
        X_test.append(batch["features"])
    X_test = torch.cat(X_test)

    # Original biased model
    original_model = get_model("biased", input_dim=metadata["n_features"])

    # "Fairwashed" model (for demo, just a different model)
    suspect_model = get_model("tabular", input_dim=metadata["n_features"])

    # Run detector
    detector = FairwashingDetector()

    detection_results = detector.detect(
        original_model, suspect_model, X_test,
        explanation_methods=["gradient", "xgrad"]
    )

    print(f"
Detection Results:")
    print(f"  Ensemble score: {detection_results['ensemble_score']:.3f}")
    print(f"  Fairwashing detected: {detection_results['fairwashing_detected']}")

    for method, result in detection_results.items():
        if isinstance(result, dict) and "score" in result:
            print(f"  {method}: {result['score']:.3f}")

    return detection_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fairwashing Detection Framework")
    parser.add_argument("--paper", type=str, default="all",
                       choices=["all", "1", "2", "3", "4", "detection"],
                       help="Which paper experiment to run")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to use")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    if args.paper in ["all", "1"]:
        dataset = args.dataset or "adult_income"
        all_results["paper1"] = run_paper1_experiment(dataset)

    if args.paper in ["all", "2"]:
        dataset = args.dataset or "compas"
        all_results["paper2"] = run_paper2_experiment(dataset)

    if args.paper in ["all", "3"]:
        dataset = args.dataset or "fashion_mnist"
        all_results["paper3"] = run_paper3_experiment(dataset)

    if args.paper in ["all", "4"]:
        dataset = args.dataset or "mnist"
        all_results["paper4"] = run_paper4_experiment(dataset)

    if args.paper in ["all", "detection"]:
        all_results["detection"] = run_detection_experiment()

    # Generate summary
    evaluator = ExperimentEvaluator(save_dir=args.output)
    summary = evaluator.generate_summary_report(all_results)

    print("
" + "="*60)
    print(summary)
    print("="*60)

    # Save summary
    with open(f"{args.output}/summary.txt", "w") as f:
        f.write(summary)

    print(f"
Results saved to {args.output}/")


if __name__ == "__main__":
    main()
