"""
Evaluation metrics and visualization for fairwashing experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch


class ExperimentEvaluator:
    """Evaluator for fairwashing experiments."""

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        self.results = {}

    def evaluate_paper1(self, candidates: List[Dict], 
                        black_box_fairness: float,
                        save_prefix: str = "paper1") -> Dict:
        """
        Evaluate Paper 1 (LaundryML) results.

        Args:
            candidates: List of candidate rule lists
            black_box_fairness: Fairness violation of black-box model

        Returns:
            Evaluation metrics
        """
        if not candidates:
            return {"error": "No candidates"}

        fidelities = [c["fidelity"] for c in candidates]
        fairness_violations = [c["fairness_violation"] for c in candidates]

        # Find Pareto frontier
        pareto = self._compute_pareto_frontier(fidelities, fairness_violations)

        # Best fairwashing candidate
        best_fairwashing = min(candidates, 
                              key=lambda c: c["fairness_violation"] if c["fidelity"] > 0.8 else float('inf'))

        metrics = {
            "n_candidates": len(candidates),
            "avg_fidelity": np.mean(fidelities),
            "avg_fairness_violation": np.mean(fairness_violations),
            "best_fidelity": np.max(fidelities),
            "best_fairness": np.min(fairness_violations),
            "pareto_size": len(pareto),
            "fairwashing_gap": black_box_fairness - best_fairwashing["fairness_violation"],
            "best_candidate": best_fairwashing,
        }

        # Plot
        self._plot_pareto(fidelities, fairness_violations, pareto, 
                         black_box_fairness, save_prefix)

        return metrics

    def evaluate_paper2(self, lime_result: Dict, shap_result: Dict,
                       fidelity: float, save_prefix: str = "paper2") -> Dict:
        """Evaluate Paper 2 (Fooling LIME/SHAP) results."""
        metrics = {
            "fidelity": fidelity,
            "lime_fooled": lime_result is not None,
            "shap_fooled": shap_result is not None,
            "n_decoy_features": lime_result.get("n_decoy_features", 0) if lime_result else 0,
        }

        # Plot explanation comparison if available
        if lime_result and "explanation" in lime_result:
            self._plot_lime_explanation(lime_result, save_prefix)

        return metrics

    def evaluate_paper3(self, history: List[Dict],
                       evaluation: Dict,
                       save_prefix: str = "paper3") -> Dict:
        """Evaluate Paper 3 (Off-Manifold) results."""
        if not history:
            return {"error": "No training history"}

        final_losses = history[-1]

        metrics = {
            "final_total_loss": final_losses.get("total_loss", 0),
            "final_output_loss": final_losses.get("output_loss", 0),
            "final_explanation_loss": final_losses.get("explanation_loss", 0),
            "test_output_mse": evaluation.get("output_mse", 0),
            "test_explanation_mse": evaluation.get("explanation_mse", 0),
            "n_epochs": len(history),
        }

        # Plot training curves
        self._plot_training_curves(history, save_prefix)

        return metrics

    def evaluate_paper4(self, original_img: np.ndarray,
                       perturbed_img: np.ndarray,
                       orig_exp: np.ndarray,
                       pert_exp: np.ndarray,
                       metrics: Dict,
                       save_prefix: str = "paper4") -> Dict:
        """Evaluate Paper 4 (Fragile Interpretation) results."""
        # Plot comparison
        self._plot_explanation_comparison(
            original_img, perturbed_img, orig_exp, pert_exp, save_prefix
        )

        return metrics

    def evaluate_detection(self, detection_results: Dict,
                          ground_truth_fairwashed: bool,
                          save_prefix: str = "detection") -> Dict:
        """Evaluate detection performance."""
        detected = detection_results.get("fairwashing_detected", False)

        tp = detected and ground_truth_fairwashed
        fp = detected and not ground_truth_fairwashed
        tn = not detected and not ground_truth_fairwashed
        fn = not detected and ground_truth_fairwashed

        return {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "correct": (tp or tn),
            "ensemble_score": detection_results.get("ensemble_score", 0),
        }

    def _compute_pareto_frontier(self, fidelities: List[float],
                                 fairness_violations: List[float]) -> List[Tuple[float, float]]:
        """Compute Pareto frontier (maximize fidelity, minimize fairness violation)."""
        points = list(zip(fidelities, fairness_violations))
        pareto = []

        for i, (f1, fv1) in enumerate(points):
            dominated = False
            for j, (f2, fv2) in enumerate(points):
                if i != j:
                    # f2 >= f1 and fv2 <= fv1 means point i is dominated
                    if f2 >= f1 and fv2 <= fv1 and (f2 > f1 or fv2 < fv1):
                        dominated = True
                        break
            if not dominated:
                pareto.append((f1, fv1))

        return pareto

    def _plot_pareto(self, fidelities, fairness_violations, pareto,
                    black_box_fairness, save_prefix):
        """Plot Pareto frontier."""
        plt.figure(figsize=(8, 6))
        plt.scatter(fidelities, fairness_violations, alpha=0.5, label="Candidates")

        if pareto:
            pf = sorted(pareto, key=lambda x: x[0])
            plt.plot([p[0] for p in pf], [p[1] for p in pf], 
                    'r-', linewidth=2, label="Pareto Frontier")

        plt.axhline(y=black_box_fairness, color='g', linestyle='--', 
                   label=f"Black-box fairness violation ({black_box_fairness:.3f})")
        plt.xlabel("Fidelity to Black-box")
        plt.ylabel("Fairness Violation")
        plt.title("Fairwashing: Fidelity vs Fairness Trade-off")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_prefix}_pareto.png", dpi=150)
        plt.close()

    def _plot_training_curves(self, history: List[Dict], save_prefix):
        """Plot training loss curves."""
        epochs = range(1, len(history) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        losses = {
            "Total Loss": [h.get("total_loss", 0) for h in history],
            "Output Loss": [h.get("output_loss", 0) for h in history],
            "Explanation Loss": [h.get("explanation_loss", 0) for h in history],
        }

        for ax, (name, values) in zip(axes, losses.items()):
            ax.plot(epochs, values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_prefix}_training.png", dpi=150)
        plt.close()

    def _plot_explanation_comparison(self, orig_img, pert_img, orig_exp, pert_exp, save_prefix):
        """Plot original vs perturbed explanations."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(orig_img.squeeze(), cmap="gray" if orig_img.squeeze().ndim == 2 else None)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pert_img.squeeze(), cmap="gray" if pert_img.squeeze().ndim == 2 else None)
        axes[0, 1].set_title("Perturbed Image")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(np.abs(orig_exp).squeeze(), cmap="hot")
        axes[1, 0].set_title("Original Explanation")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(np.abs(pert_exp).squeeze(), cmap="hot")
        axes[1, 1].set_title("Perturbed Explanation")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_prefix}_comparison.png", dpi=150)
        plt.close()

    def _plot_lime_explanation(self, lime_result: Dict, save_prefix):
        """Plot LIME explanation weights."""
        explanation = lime_result.get("explanation", {})
        if not explanation:
            return

        features = list(explanation.keys())[:20]  # Top 20
        weights = [explanation[f] for f in features]

        plt.figure(figsize=(10, 6))
        colors = ['red' if 'decoy' in str(f) else 'blue' for f in features]
        plt.barh(range(len(features)), weights, color=colors)
        plt.yticks(range(len(features)), [str(f)[:30] for f in features])
        plt.xlabel("LIME Weight")
        plt.title("LIME Explanation (Red = Decoy Features)")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_prefix}_lime.png", dpi=150)
        plt.close()

    def generate_summary_report(self, all_results: Dict) -> str:
        """Generate a text summary of all experiments."""
        report = []
        report.append("=" * 60)
        report.append("FAIRWASHING DETECTION FRAMEWORK - EXPERIMENT SUMMARY")
        report.append("=" * 60)
        report.append("")

        for paper, results in all_results.items():
            report.append(f"\n{paper.upper()}")
            report.append("-" * 40)
            for key, value in results.items():
                if isinstance(value, (int, float, bool, str)):
                    report.append(f"  {key}: {value}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
