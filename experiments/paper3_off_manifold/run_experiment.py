"""
Experiment script for Paper 3: Fairwashing with Off-Manifold Detergent
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.config import PAPER3_CONFIG, RANDOM_SEED, DEVICE
from src.utils.helpers import set_seed, compute_explanation_similarity
from src.data.datasets import get_image_loaders
from src.models.architectures import Conv4, VGG16
from src.explanations.methods import get_explainer
from src.attacks.paper3_off_manifold import (
    OffManifoldFairwasher, TangentSpaceProjector, TSPDefender,
    create_target_explanation
)
from src.evaluation.metrics import ExperimentEvaluator


def run_experiment(dataset_name="fashion_mnist", n_epochs=20):
    """Run complete Paper 3 experiment."""
    print(f"\n{'='*60}")
    print(f"Paper 3: Off-Manifold Fairwashing - {dataset_name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # Load data
    train_loader, test_loader = get_image_loaders(dataset_name, batch_size=128)

    # Get sample batch
    x_batch, y_batch = next(iter(test_loader))
    print(f"Data shape: {x_batch.shape}, Labels: {y_batch.shape}")

    # Create models
    if dataset_name in ["mnist", "fashion_mnist"]:
        teacher = Conv4(num_classes=10).to(DEVICE)
        student = Conv4(num_classes=10).to(DEVICE)
    else:
        teacher = VGG16(num_classes=10).to(DEVICE)
        student = VGG16(num_classes=10).to(DEVICE)

    # Train teacher model briefly (or load pretrained)
    print(f"\nTraining teacher model...")
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):  # Brief training for demo
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = teacher(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx >= 50:  # Limit for demo
                break
        print(f"  Teacher Epoch {epoch+1}: Loss={total_loss/(batch_idx+1):.4f}")

    teacher.eval()

    # Evaluate teacher
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = teacher(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            if total >= 1000:
                break
    print(f"Teacher accuracy: {correct/total:.4f}")

    # Get original explanation
    x_test = x_batch[:1].to(DEVICE)
    x_test.requires_grad = True
    teacher_out = teacher(x_test)
    target_class = teacher_out.argmax(dim=1).item()
    teacher_grad = torch.autograd.grad(teacher_out[0, target_class], x_test)[0]

    print(f"\nOriginal explanation stats:")
    print(f"  Mean: {teacher_grad.mean().item():.4f}")
    print(f"  Std: {teacher_grad.std().item():.4f}")
    print(f"  Max: {teacher_grad.abs().max().item():.4f}")

    # Create target explanation
    target_exp = create_target_explanation(x_test[0].shape, pattern="center")
    target_exp = target_exp.to(DEVICE)
    print(f"\nTarget explanation shape: {target_exp.shape}")

    # Create fairwasher
    print(f"\nTraining fairwashed student model...")
    fairwasher = OffManifoldFairwasher(
        teacher, student,
        explanation_method="gradient",
        alpha=PAPER3_CONFIG["fairwashing"]["alpha"],
        device=DEVICE
    )

    # Train
    history = fairwasher.train(
        train_loader, target_exp,
        n_epochs=n_epochs,
        lr=PAPER3_CONFIG["fairwashing"]["lr"]
    )

    # Evaluate
    print(f"\nEvaluating fairwashed model...")
    eval_results = fairwasher.evaluate(test_loader, target_exp)

    # Get student explanation
    student.eval()
    x_test_s = x_batch[:1].to(DEVICE)
    x_test_s.requires_grad = True
    student_out = student(x_test_s)
    student_class = student_out.argmax(dim=1).item()
    student_grad = torch.autograd.grad(student_out[0, student_class], x_test_s)[0]

    # Compute explanation similarity
    teacher_exp_np = teacher_grad.detach().cpu().numpy()
    student_exp_np = student_grad.detach().cpu().numpy()
    target_exp_np = target_exp.cpu().numpy()

    sim_teacher_student = compute_explanation_similarity(
        teacher_exp_np, student_exp_np, metric="pcc"
    )
    sim_student_target = compute_explanation_similarity(
        student_exp_np, target_exp_np, metric="pcc"
    )

    print(f"\nExplanation similarities:")
    print(f"  Teacher-Student: {sim_teacher_student:.4f}")
    print(f"  Student-Target: {sim_student_target:.4f}")

    # TSP Defense
    print(f"\nTesting Tangent Space Projection defense...")

    # Collect data for projector
    X_data = []
    for data, _ in train_loader:
        X_data.append(data.numpy())
        if len(X_data) > 5:
            break
    X_data = np.vstack(X_data)
    X_flat = X_data.reshape(X_data.shape[0], -1)[:500]  # Limit for demo

    projector = TangentSpaceProjector(
        n_neighbors=PAPER3_CONFIG["tangent_space_projection"]["neighbours"],
        n_components=PAPER3_CONFIG["tangent_space_projection"]["d_singular"]
    )
    projector.fit(X_flat)

    tsp_defender = TSPDefender(projector)

    # Compute TSP explanation
    x_tsp = x_batch[:1].to(DEVICE)
    tsp_exp = tsp_defender.compute_tsp_explanation(student, x_tsp, student_class)

    tsp_np = tsp_exp.cpu().numpy()
    sim_tsp_target = compute_explanation_similarity(tsp_np, target_exp_np, metric="pcc")
    sim_tsp_teacher = compute_explanation_similarity(tsp_np, teacher_exp_np, metric="pcc")

    print(f"\nTSP Explanation similarities:")
    print(f"  TSP-Target: {sim_tsp_target:.4f}")
    print(f"  TSP-Teacher: {sim_tsp_teacher:.4f}")

    # Evaluate
    evaluator = ExperimentEvaluator(save_dir="results")
    metrics = evaluator.evaluate_paper3(history, eval_results, save_prefix="paper3")

    # Plot explanations
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(x_test[0, 0].detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(np.abs(teacher_exp_np[0, 0]), cmap="hot")
    axes[1].set_title("Teacher Explanation")
    axes[1].axis("off")

    axes[2].imshow(np.abs(student_exp_np[0, 0]), cmap="hot")
    axes[2].set_title("Student Explanation")
    axes[2].axis("off")

    axes[3].imshow(np.abs(target_exp_np[0, 0]), cmap="hot")
    axes[3].set_title("Target Explanation")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("results/paper3_explanations.png", dpi=150)
    plt.close()

    print(f"\nExplanation comparison saved to results/paper3_explanations.png")

    return {
        "metrics": metrics,
        "history": history,
        "evaluation": eval_results,
        "similarities": {
            "teacher_student": sim_teacher_student,
            "student_target": sim_student_target,
            "tsp_target": sim_tsp_target,
            "tsp_teacher": sim_tsp_teacher,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--n_epochs", type=int, default=20)
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.n_epochs)
