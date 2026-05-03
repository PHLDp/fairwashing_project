"""
Experiment script for Paper 4: Interpretation of Neural Networks is Fragile
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.config import PAPER4_CONFIG, RANDOM_SEED, DEVICE
from src.utils.helpers import set_seed
from src.data.datasets import get_image_loaders
from src.models.architectures import Conv4
from src.attacks.paper4_fragile_interpretation import (
    TopKAttack, MassCenterAttack, RandomSignPerturbation,
    compute_interpretation_metrics
)
from src.evaluation.metrics import ExperimentEvaluator


def run_experiment(dataset_name="mnist", n_samples=5):
    """Run complete Paper 4 experiment."""
    print(f"\n{'='*60}")
    print(f"Paper 4: Fragile Interpretation - {dataset_name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # Load data
    _, test_loader = get_image_loaders(dataset_name, batch_size=1)

    # Load model
    model = Conv4(num_classes=10).to(DEVICE)
    model.eval()

    # Brief training for demo (in practice, use pretrained)
    print("Training model briefly for demo...")
    train_loader, _ = get_image_loaders(dataset_name, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx >= 100:
                break

    model.eval()

    # Test on multiple samples
    all_results = {
        "random": [],
        "top_k": [],
        "mass_center": [],
    }

    for sample_idx in range(n_samples):
        print(f"\n--- Sample {sample_idx + 1}/{n_samples} ---")

        # Get test sample
        x, y = next(iter(test_loader))
        x = x.to(DEVICE)

        # Original prediction
        with torch.no_grad():
            orig_out = model(x)
            orig_pred = orig_out.argmax(dim=1).item()
            orig_conf = torch.softmax(orig_out, dim=1)[0, orig_pred].item()

        print(f"Original prediction: {orig_pred} (confidence: {orig_conf:.4f})")

        # Original explanation
        x_grad = x.clone().requires_grad_(True)
        output = model(x_grad)
        orig_exp = torch.autograd.grad(output[0, orig_pred], x_grad)[0]
        orig_exp_np = orig_exp.detach().cpu().numpy()

        # Random sign perturbation
        print("  Random sign perturbation...")
        random_attack = RandomSignPerturbation(model, epsilon=8.0, device=DEVICE)
        x_random = random_attack.attack(x)

        with torch.no_grad():
            random_out = model(x_random)
            random_pred = random_out.argmax(dim=1).item()
            random_conf = torch.softmax(random_out, dim=1)[0, random_pred].item()

        x_grad_r = x_random.clone().requires_grad_(True)
        output_r = model(x_grad_r)
        random_exp = torch.autograd.grad(output_r[0, random_pred], x_grad_r)[0]
        random_exp_np = random_exp.detach().cpu().numpy()

        random_metrics = compute_interpretation_metrics(
            orig_exp_np, random_exp_np, k=100
        )
        random_metrics["pred_preserved"] = (orig_pred == random_pred)
        random_metrics["conf_change"] = abs(orig_conf - random_conf)
        all_results["random"].append(random_metrics)

        print(f"    Pred preserved: {orig_pred == random_pred}")
        print(f"    Spearman: {random_metrics['spearman_correlation']:.4f}")
        print(f"    Top-k intersection: {random_metrics['top_k_intersection']:.4f}")

        # Top-k attack
        print("  Top-k attack...")
        topk_attack = TopKAttack(model, k=100, epsilon=8.0, 
                                n_iterations=100, step_size=0.5, device=DEVICE)
        x_topk = topk_attack.attack(x, explanation_method="gradient")

        with torch.no_grad():
            topk_out = model(x_topk)
            topk_pred = topk_out.argmax(dim=1).item()
            topk_conf = torch.softmax(topk_out, dim=1)[0, topk_pred].item()

        x_grad_t = x_topk.clone().requires_grad_(True)
        output_t = model(x_grad_t)
        topk_exp = torch.autograd.grad(output_t[0, topk_pred], x_grad_t)[0]
        topk_exp_np = topk_exp.detach().cpu().numpy()

        topk_metrics = compute_interpretation_metrics(
            orig_exp_np, topk_exp_np, k=100
        )
        topk_metrics["pred_preserved"] = (orig_pred == topk_pred)
        topk_metrics["conf_change"] = abs(orig_conf - topk_conf)
        all_results["top_k"].append(topk_metrics)

        print(f"    Pred preserved: {orig_pred == topk_pred}")
        print(f"    Spearman: {topk_metrics['spearman_correlation']:.4f}")
        print(f"    Top-k intersection: {topk_metrics['top_k_intersection']:.4f}")

        # Mass-center attack
        print("  Mass-center attack...")
        center_attack = MassCenterAttack(model, epsilon=8.0, 
                                        n_iterations=100, device=DEVICE)
        x_center = center_attack.attack(x, explanation_method="gradient")

        with torch.no_grad():
            center_out = model(x_center)
            center_pred = center_out.argmax(dim=1).item()
            center_conf = torch.softmax(center_out, dim=1)[0, center_pred].item()

        x_grad_c = x_center.clone().requires_grad_(True)
        output_c = model(x_grad_c)
        center_exp = torch.autograd.grad(output_c[0, center_pred], x_grad_c)[0]
        center_exp_np = center_exp.detach().cpu().numpy()

        center_metrics = compute_interpretation_metrics(
            orig_exp_np, center_exp_np, k=100
        )
        center_metrics["pred_preserved"] = (orig_pred == center_pred)
        center_metrics["conf_change"] = abs(orig_conf - center_conf)
        all_results["mass_center"].append(center_metrics)

        print(f"    Pred preserved: {orig_pred == center_pred}")
        print(f"    Spearman: {center_metrics['spearman_correlation']:.4f}")
        print(f"    Center shift: {center_metrics['center_shift']:.2f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    for attack_name, results in all_results.items():
        print(f"\n{attack_name.upper()}:")

        spearmans = [r["spearman_correlation"] for r in results]
        topk_inters = [r["top_k_intersection"] for r in results]
        pred_preserved = [r["pred_preserved"] for r in results]

        print(f"  Spearman correlation: {np.mean(spearmans):.4f} (+/- {np.std(spearmans):.4f})")
        print(f"  Top-k intersection: {np.mean(topk_inters):.4f} (+/- {np.std(topk_inters):.4f})")
        print(f"  Prediction preserved: {np.mean(pred_preserved):.2%}")

        if "center_shift" in results[0]:
            shifts = [r["center_shift"] for r in results]
            print(f"  Center shift: {np.mean(shifts):.2f} (+/- {np.std(shifts):.2f})")

    # Plot aggregate comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    attack_names = list(all_results.keys())
    spearman_means = [np.mean([r["spearman_correlation"] for r in all_results[name]]) 
                     for name in attack_names]
    spearman_stds = [np.std([r["spearman_correlation"] for r in all_results[name]]) 
                    for name in attack_names]

    axes[0].bar(attack_names, spearman_means, yerr=spearman_stds, capsize=5)
    axes[0].set_ylabel("Spearman Correlation")
    axes[0].set_title("Explanation Rank Correlation After Attack")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    topk_means = [np.mean([r["top_k_intersection"] for r in all_results[name]]) 
                 for name in attack_names]
    topk_stds = [np.std([r["top_k_intersection"] for r in all_results[name]]) 
                for name in attack_names]

    axes[1].bar(attack_names, topk_means, yerr=topk_stds, capsize=5)
    axes[1].set_ylabel("Top-k Intersection")
    axes[1].set_title("Top-k Feature Overlap After Attack")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/paper4_aggregate.png", dpi=150)
    plt.close()

    print(f"\nAggregate plot saved to results/paper4_aggregate.png")

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

    return {
        "all_results": all_results,
        "aggregate": {
            "random": {
                "spearman_mean": np.mean([r["spearman_correlation"] for r in all_results["random"]]),
                "topk_mean": np.mean([r["top_k_intersection"] for r in all_results["random"]]),
            },
            "top_k": {
                "spearman_mean": np.mean([r["spearman_correlation"] for r in all_results["top_k"]]),
                "topk_mean": np.mean([r["top_k_intersection"] for r in all_results["top_k"]]),
            },
            "mass_center": {
                "spearman_mean": np.mean([r["spearman_correlation"] for r in all_results["mass_center"]]),
                "topk_mean": np.mean([r["top_k_intersection"] for r in all_results["mass_center"]]),
            },
        },
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.n_samples)
