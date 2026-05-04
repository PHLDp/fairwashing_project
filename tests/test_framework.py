"""
Unit tests for Fairwashing Detection Framework.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.utils.helpers import set_seed, compute_demographic_parity, compute_explanation_similarity
from src.models.architectures import TabularClassifier, Conv4, BiasedClassifier
from src.explanations.methods import GradientExplainer, XGradExplainer
from src.attacks.paper1_laundryml import RuleList, LaundryML
from src.attacks.paper2_fooling_lime_shap import ScaffoldingAttack, BiasedClassifierFactory
from src.attacks.paper3_off_manifold import TangentSpaceProjector, create_target_explanation
from src.attacks.paper4_fragile_interpretation import compute_interpretation_metrics
from src.detection.detector import FairwashingDetector, RuleListFairwashingDetector


class TestHelpers(unittest.TestCase):
    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_demographic_parity(self):
        y_pred = np.array([1, 1, 0, 0, 1, 1])
        sensitive = np.array([0, 0, 0, 1, 1, 1])
        dp = compute_demographic_parity(y_pred, sensitive)
        self.assertAlmostEqual(dp, 1/3, places=5)

    def test_explanation_similarity(self):
        exp1 = np.array([1, 2, 3, 4, 5])
        exp2 = np.array([1, 2, 3, 4, 5])
        sim = compute_explanation_similarity(exp1, exp2, metric="pcc")
        self.assertAlmostEqual(sim, 1.0, places=5)


class TestModels(unittest.TestCase):
    def test_tabular_classifier(self):
        model = TabularClassifier(input_dim=10, hidden_dims=[20, 10], num_classes=2)
        x = torch.randn(5, 10)
        out = model(x)
        self.assertEqual(out.shape, (5, 2))

    def test_conv4(self):
        model = Conv4(num_classes=10, input_channels=1)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_biased_classifier(self):
        model = BiasedClassifier(10, sensitive_weight=0.9)
        model.set_biased_weights(sensitive_idx=0)
        x = torch.randn(5, 10)
        out = model(x)
        self.assertEqual(out.shape, (5, 2))


class TestExplanations(unittest.TestCase):
    def test_gradient_explainer(self):
        model = TabularClassifier(10, num_classes=2)
        explainer = GradientExplainer(model)
        x = torch.randn(1, 10)
        exp = explainer.explain(x)
        self.assertEqual(exp.shape, (10,))

    def test_xgrad_explainer(self):
        model = TabularClassifier(10, num_classes=2)
        explainer = XGradExplainer(model)
        x = torch.randn(1, 10)
        exp = explainer.explain(x)
        self.assertEqual(exp.shape, (10,))


class TestPaper1(unittest.TestCase):
    def test_rule_list(self):
        rules = [("f0 > 0.5", 1), ("f1 <= 0.3", 0)]
        rl = RuleList(rules, default_class=0)

        import pandas as pd
        df = pd.DataFrame({"f0": [0.6, 0.4], "f1": [0.2, 0.5]})
        preds = rl.predict(df)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_laundryml_init(self):
        model = TabularClassifier(5, num_classes=2)
        laundry = LaundryML(model, "sensitive")
        self.assertIsNotNone(laundry)


class TestPaper2(unittest.TestCase):
    def test_scaffolding_attack(self):
        base_model = BiasedClassifierFactory.create_gender_biased_classifier(5, 0, 0.9)
        attack = ScaffoldingAttack(base_model, n_decoy_features=2)

        x = torch.randn(10, 5)
        x_aug = attack.add_decoy_features(x)
        self.assertEqual(x_aug.shape, (10, 7))

    def test_fidelity(self):
        base_model = BiasedClassifierFactory.create_gender_biased_classifier(5, 0, 0.9)
        attack = ScaffoldingAttack(base_model, n_decoy_features=2)

        x = torch.randn(10, 5)
        fidelity = attack.evaluate_fidelity(x)
        self.assertEqual(fidelity, 1.0)  # Should be perfect


class TestPaper3(unittest.TestCase):
    def test_tangent_space_projector(self):
        X = np.random.randn(100, 20)
        projector = TangentSpaceProjector(n_neighbors=10, n_components=5)
        projector.fit(X)

        x = np.random.randn(20)
        proj = projector.project(x)
        self.assertEqual(proj.shape, (20,))

    def test_create_target_explanation(self):
        exp = create_target_explanation((1, 28, 28), pattern="center")
        self.assertEqual(exp.shape, (1, 28, 28))
        self.assertTrue(exp.abs().sum() > 0)


class TestPaper4(unittest.TestCase):
    def test_interpretation_metrics(self):
        orig = np.array([1, 2, 3, 4, 5])
        pert = np.array([5, 4, 3, 2, 1])
        metrics = compute_interpretation_metrics(orig, pert, k=3)

        self.assertIn("spearman_correlation", metrics)
        self.assertIn("top_k_intersection", metrics)
        self.assertIn("center_shift", metrics)


class TestDetection(unittest.TestCase):
    def test_rule_list_detector(self):
        bb_preds = np.array([1, 1, 1, 0, 0, 0])
        rl_preds = np.array([0, 0, 0, 0, 0, 0])  # Appears fair
        sensitive = np.array([0, 0, 1, 0, 0, 1])

        detector = RuleListFairwashingDetector()
        result = detector.detect(bb_preds, rl_preds, sensitive)

        self.assertIn("fairwashing_detected", result)
        self.assertIn("fairwashing_gap", result)

    def test_ensemble_detector(self):
        model1 = TabularClassifier(5, num_classes=2)
        model2 = TabularClassifier(5, num_classes=2)
        x = torch.randn(10, 5)

        detector = FairwashingDetector()
        result = detector.detect(model1, model2, x)

        self.assertIn("ensemble_score", result)
        self.assertIn("fairwashing_detected", result)


if __name__ == "__main__":
    unittest.main()
