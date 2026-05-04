#!/usr/bin/env python3
"""
Validation script for Fairwashing Detection Framework.
Tests all imports and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    errors = []

    try:
        from config import PAPER1_CONFIG, PAPER2_CONFIG, PAPER3_CONFIG, PAPER4_CONFIG
        print("  ✓ config.py")
    except Exception as e:
        errors.append(f"config.py: {e}")
        print(f"  ✗ config.py: {e}")

    try:
        from utils.helpers import set_seed, compute_demographic_parity
        print("  ✓ utils/helpers.py")
    except Exception as e:
        errors.append(f"utils/helpers.py: {e}")
        print(f"  ✗ utils/helpers.py: {e}")

    try:
        from data.datasets import get_tabular_loaders, get_image_loaders
        print("  ✓ data/datasets.py")
    except Exception as e:
        errors.append(f"data/datasets.py: {e}")
        print(f"  ✗ data/datasets.py: {e}")

    try:
        from models.architectures import TabularClassifier, Conv4, BiasedClassifier
        print("  ✓ models/architectures.py")
    except Exception as e:
        errors.append(f"models/architectures.py: {e}")
        print(f"  ✗ models/architectures.py: {e}")

    try:
        from explanations.methods import get_explainer, GradientExplainer, LRPExplainer
        print("  ✓ explanations/methods.py")
    except Exception as e:
        errors.append(f"explanations/methods.py: {e}")
        print(f"  ✗ explanations/methods.py: {e}")

    try:
        from explanations.lrp_full import FullLRPExplainer, LRPRule
        print("  ✓ explanations/lrp_full.py (Task 1)")
    except Exception as e:
        errors.append(f"explanations/lrp_full.py: {e}")
        print(f"  ✗ explanations/lrp_full.py: {e}")

    try:
        from explanations.image_lime_shap import ImageLIMEExplainer, ImageSHAPExplainer, ImageSegmenter
        print("  ✓ explanations/image_lime_shap.py (Task 7)")
    except Exception as e:
        errors.append(f"explanations/image_lime_shap.py: {e}")
        print(f"  ✗ explanations/image_lime_shap.py: {e}")

    try:
        from attacks.paper1_laundryml import LaundryML, RuleList
        print("  ✓ attacks/paper1_laundryml.py")
    except Exception as e:
        errors.append(f"attacks/paper1_laundryml.py: {e}")
        print(f"  ✗ attacks/paper1_laundryml.py: {e}")

    try:
        from attacks.corels_wrapper import CORELSWrapper, FairCORELS
        print("  ✓ attacks/corels_wrapper.py (Task 3)")
    except Exception as e:
        errors.append(f"attacks/corels_wrapper.py: {e}")
        print(f"  ✗ attacks/corels_wrapper.py: {e}")

    try:
        from attacks.paper2_fooling_lime_shap import ScaffoldingAttack, OODDetector
        print("  ✓ attacks/paper2_fooling_lime_shap.py")
    except Exception as e:
        errors.append(f"attacks/paper2_fooling_lime_shap.py: {e}")
        print(f"  ✗ attacks/paper2_fooling_lime_shap.py: {e}")

    try:
        from attacks.paper3_off_manifold import OffManifoldFairwasher, TangentSpaceProjector
        print("  ✓ attacks/paper3_off_manifold.py")
    except Exception as e:
        errors.append(f"attacks/paper3_off_manifold.py: {e}")
        print(f"  ✗ attacks/paper3_off_manifold.py: {e}")

    try:
        from attacks.paper4_fragile_interpretation import TopKAttack, MassCenterAttack
        print("  ✓ attacks/paper4_fragile_interpretation.py")
    except Exception as e:
        errors.append(f"attacks/paper4_fragile_interpretation.py: {e}")
        print(f"  ✗ attacks/paper4_fragile_interpretation.py: {e}")

    try:
        from detection.detector import FairwashingDetector, RuleListFairwashingDetector
        print("  ✓ detection/detector.py")
    except Exception as e:
        errors.append(f"detection/detector.py: {e}")
        print(f"  ✗ detection/detector.py: {e}")

    try:
        from evaluation.metrics import ExperimentEvaluator
        print("  ✓ evaluation/metrics.py")
    except Exception as e:
        errors.append(f"evaluation/metrics.py: {e}")
        print(f"  ✗ evaluation/metrics.py: {e}")

    return errors


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    errors = []

    import torch
    import numpy as np

    # Test model creation
    try:
        from models.architectures import TabularClassifier, Conv4
        model = TabularClassifier(10, num_classes=2)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 2)
        print("  ✓ TabularClassifier works")
    except Exception as e:
        errors.append(f"TabularClassifier: {e}")
        print(f"  ✗ TabularClassifier: {e}")

    # Test explanation
    try:
        from explanations.methods import GradientExplainer
        explainer = GradientExplainer(model)
        exp = explainer.explain(x[:1])
        assert exp.shape == (10,)
        print("  ✓ GradientExplainer works")
    except Exception as e:
        errors.append(f"GradientExplainer: {e}")
        print(f"  ✗ GradientExplainer: {e}")

    # Test full LRP
    try:
        from explanations.lrp_full import FullLRPExplainer
        lrp = FullLRPExplainer(model, rule="z+")
        exp = lrp.explain(x[:1])
        assert exp.shape == (1, 10)
        print("  ✓ FullLRPExplainer works (Task 1)")
    except Exception as e:
        errors.append(f"FullLRPExplainer: {e}")
        print(f"  ✗ FullLRPExplainer: {e}")

    # Test image segmenter
    try:
        from explanations.image_lime_shap import ImageSegmenter
        segmenter = ImageSegmenter(method="slic", n_segments=10)
        img = np.random.rand(28, 28)
        segments = segmenter.segment(img)
        assert segments.shape == (28, 28)
        print("  ✓ ImageSegmenter works (Task 7)")
    except Exception as e:
        errors.append(f"ImageSegmenter: {e}")
        print(f"  ✗ ImageSegmenter: {e}")

    # Test CORELS wrapper
    try:
        from attacks.corels_wrapper import CORELSWrapper
        wrapper = CORELSWrapper(max_length=5)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        wrapper.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])
        preds = wrapper.predict(X)
        assert len(preds) == 50
        print("  ✓ CORELSWrapper works (Task 3)")
    except Exception as e:
        errors.append(f"CORELSWrapper: {e}")
        print(f"  ✗ CORELSWrapper: {e}")

    # Test detection
    try:
        from detection.detector import FairwashingDetector
        detector = FairwashingDetector()
        model1 = TabularClassifier(5, num_classes=2)
        model2 = TabularClassifier(5, num_classes=2)
        x_test = torch.randn(5, 5)
        result = detector.detect(model1, model2, x_test)
        assert "ensemble_score" in result
        print("  ✓ FairwashingDetector works")
    except Exception as e:
        errors.append(f"FairwashingDetector: {e}")
        print(f"  ✗ FairwashingDetector: {e}")

    return errors


def main():
    print("="*60)
    print("FAIRWASHING DETECTION FRAMEWORK - VALIDATION")
    print("="*60)

    import_errors = test_imports()
    func_errors = test_basic_functionality()

    all_errors = import_errors + func_errors

    print("\n" + "="*60)
    if not all_errors:
        print(" ALL TESTS PASSED!")
        print("The framework is ready to use.")
    else:
        print(f" {len(all_errors)} ERROR(S) FOUND:")
        for err in all_errors:
            print(f"  - {err}")
    print("="*60)

    return len(all_errors)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
