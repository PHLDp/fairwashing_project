# Fairwashing Detection Framework - Comprehensive Usage Guide

This document provides detailed usage instructions for the fairwashing detection framework, covering every component, module, and expected output.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Project Architecture](#3-project-architecture)
4. [Running Experiments](#4-running-experiments)
5. [Detailed Module Documentation](#5-detailed-module-documentation)
6. [Expected Results](#6-expected-results)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Configuration Reference](#8-configuration-reference)
9. [API Reference](#9-api-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### What is Fairwashing?

Fairwashing is the process of making a biased machine learning model appear fair by using post-hoc explanation methods to provide seemingly fair rationalizations, while the underlying model remains biased. This framework implements four major fairwashing attacks from research papers and provides detection methods to identify them.

### Papers Implemented

| Paper | Title | Attack Type | Key Concept |
|-------|-------|------------|------------|
| Paper 1 | LaundryML (ICML 2019) | Rule List Fairwashing | Rationalize biased models with interpretable rules |
| Paper 2 | Fooling LIME/SHAP (AIES 2020) | Scaffolding Attack | Add decoy features to fool explanation methods |
| Paper 3 | Off-Manifold Detergent (ICML 2020) | Explanation Manipulation | Manipulate explanations arbitrarily |
| Paper 4 | Fragile Interpretation (AAAI 2019) | Adversarial Attacks | Perturb inputs to change explanations |

---

## 2. Installation

### Requirements
```
# Python 3.8+
# PyTorch 1.9+
# Required packages (see requirements.txt)
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
 pillow>=8.0.0
 shap>=0.39.0
 lime>=0.2.0
 scipy>=1.6.0
```

### Installation Steps
```bash
# Clone repository
git clone <repository_url>
cd fairwashing_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Installation successful')"
```

---

## 3. Project Architecture

```
fairwashing_project/
├── src/                           # Main source code
│   ├── config.py                 # All hyperparameters
│   ├── main_pipeline.py         # Main orchestrator
│   ├── __init__.py
│   ├── data/                   # Data loading
│   │   ├── __init__.py
│   │   └── datasets.py          # Tabular & image datasets
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── architectures.py    # CNN, MLP, biased models
│   ├── explanations/            # XAI methods
│   │   ├── __init__.py
│   │   ├── methods.py         # All explainers
│   │   ├── lrp_full.py       # Full LRP implementation
│   │   └── image_lime_shap.py
│   ├── attacks/               # Fairwashing attacks
│   │   ├── __init__.py
│   │   ├── paper1_laundryml.py
│   │   ├── paper2_fooling_lime_shap.py
│   │   ├── paper3_off_manifold.py
│   │   ├── paper4_fragile_interpretation.py
│   │   └── corels_wrapper.py
│   ├── detection/             # Detection module
│   │   ├── __init__.py
│   │   └── detector.py       # Ensemble detector
│   ├── evaluation/            # Metrics & evaluation
│   │   ├── __init__.py
│   │   └── metrics.py       # Evaluation functions
│   └── utils/                # Utilities
│       ├── __init__.py
│       └── helpers.py      # Helper functions
├── experiments/               # Experiment runners
│   ├── paper1_laundryml/
│   │   └── run_experiment.py
│   ├── paper2_fooling_lime_shap/
│   │   └── run_experiment.py
│   ├── paper3_off_manifold/
│   │   └── run_experiment.py
│   ├── paper4_fragile_interpretation/
│   │   └── run_experiment.py
│   └── detection/
│       └── run_experiment.py
├── tests/
│   └── test_framework.py
├── notebooks/
│   └── demo.ipynb
├── results/                  # Output directory
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Running Experiments

### 4.1 Quick Start - Run All Papers

```bash
# Run complete pipeline
python src/main_pipeline.py --paper all

# Results saved to results/ directory
```

### 4.2 Individual Paper Experiments

#### Paper 1: LaundryML (Rule List Fairwashing)
```bash
python src/main_pipeline.py --paper 1 --dataset adult_income
# OR
python experiments/paper1_laundryml/run_experiment.py --dataset adult_income --n_candidates 50
```

**What happens:**
1. Loads the Adult Income dataset
2. Creates a biased classifier (uses gender as main predictor)
3. Enumerates 50 rule list candidates with varying fairness-accuracy trade-offs
4. Finds the best "fair" rationalization
5. Detects the fairwashing gap

**Expected output:**
```
Paper 1: LaundryML - adult_income
============================================================
Train: (3615, 98), Test: (905, 98)
Sensitive attr distribution: [2816  799]

Black-box fairness violation (train): 0.8500
Black-box fairness violation (test): 0.8400
Black-box accuracy (train): 0.9500
Black-box accuracy (test): 0.9400

Enumerating 50 rule list candidates...
Generated 50 candidates

Best Rationalization:
  Fidelity: 0.9200
  Fairness violation: 0.0500
  Objective: 0.9100

Detection:
  Fairwashing detected: True
  Fairwashing gap: 0.7900
  Confidence: 0.9500
```

#### Paper 2: Fooling LIME and SHAP
```bash
python src/main_pipeline.py --paper 2 --dataset compas
# OR
python experiments/paper2_fooling_lime_shap/run_experiment.py --dataset compas --n_decoy 2
```

**What happens:**
1. Creates a biased classifier for COMPAS dataset
2. Adds uncorrelated decoy features
3. Tests if LIME attributes importance to decoys
4. Tests if SHAP attributes importance to decoys
5. Runs OOD detection

**Expected output:**
```
Paper 2: Fooling LIME and SHAP - compas
============================================================
Train: (4939, 104), Test: (1235, 104)

Biased model accuracy: 0.8500
Biased model fairness violation: 0.8200

Creating scaffolding attack with 2 decoy features...
Scaffolding fidelity: 0.9800

Fooling LIME explanation...
LIME fooled successfully!
Decoy features in explanation:
  decoy_0: 0.4500 [DECOY]
  decoy_1: 0.3800 [DECOY]
  feature_0: 0.0500
  feature_1: 0.0300

Fooling SHAP explanation...
SHAP fooled successfully!

Testing OOD detection...
Normal data OOD score: 0.1200 (+/- 0.0800)
OOD data OOD score: 0.6500 (+/- 0.1500)
```

#### Paper 3: Off-Manifold Fairwashing
```bash
python src/main_pipeline.py --paper 3 --dataset fashion_mnist
# OR
python experiments/paper3_off_manifold/run_experiment.py --dataset fashion_mnist --n_epochs 20
```

**What happens:**
1. Loads Fashion-MNIST dataset
2. Trains teacher model (original)
3. Trains student model to match outputs but produce different explanations
4. Uses Tangent Space Projection as defense

**Expected output:**
```
Paper 3: Off-Manifold Fairwashing - fashion_mnist
============================================================
Data shape: torch.Size([128, 1, 28, 28]), Labels: torch.Size([128])

Training teacher model...
  Teacher Epoch 1: Loss=1.2500
  Teacher Epoch 2: Loss=0.8500
  Teacher Epoch 3: Loss=0.6500
Teacher accuracy: 0.8800

Original explanation stats:
  Mean: -0.0020
  Std: 0.1500
  Max: 0.8500

Training fairwashed student model...
Epoch 10/20: total_loss=0.1200, output_loss=0.0800, explanation_loss=0.0400

Evaluating fairwashed model...

Explanation similarities:
  Teacher-Student: 0.2500
  Student-Target: 0.8500

Testing Tangent Space Projection defense...
TSP Explanation similarities:
  TSP-Target: 0.3500
  TSP-Teacher: 0.7500
```

#### Paper 4: Fragile Interpretation
```bash
python src/main_pipeline.py --paper 4 --dataset mnist
# OR
python experiments/paper4_fragile_interpretation/run_experiment.py --dataset mnist --n_samples 5
```

**What happens:**
1. Loads MNIST classifier
2. Applies random sign perturbation
3. Applies top-k attack
4. Applies mass-center attack
5. Computes fragility metrics

**Expected output:**
```
Paper 4: Fragile Interpretation - mnist
============================================================
Training model briefly for demo...

--- Sample 1/5 ---
Original prediction: 7 (confidence: 0.9800)
  Random sign perturbation...
    Pred preserved: True
    Spearman: 0.3500
    Top-k intersection: 0.2500
  Top-k attack...
    Pred preserved: True
    Spearman: 0.1800
    Top-k intersection: 0.1200
  Mass-center attack...
    Pred preserved: True
    Spearman: 0.4200
    Center shift: 5.20

AGGREGATE RESULTS
============================================================

RANDOM:
  Spearman correlation: 0.3200 (+/- 0.0800)
  Top-k intersection: 0.2200 (+/- 0.0500)
  Prediction preserved: 100%

TOP_K:
  Spearman correlation: 0.1500 (+/- 0.0400)
  Top-k intersection: 0.1100 (+/- 0.0300)
  Prediction preserved: 100%

MASS_CENTER:
  Spearman correlation: 0.3800 (+/- 0.1000)
  Top-k intersection: 0.2500 (+/- 0.0600)
  Prediction preserved: 100%
```

#### Detection Module
```bash
python src/main_pipeline.py --paper detection
# OR
python experiments/detection/run_experiment.py --mode comprehensive
```

**What happens:**
1. Runs multiple detection methods (8 total)
2. Computes ensemble score
3. Returns detection result + confidence

**Expected output:**
```
DETECTION MODULE
============================================================

Running ensemble detector...

Detection Results:
  Ensemble score: 0.8500
  Fairwashing detected: True

  explanation_consistency:
    Score: 0.7500
    avg_consistency: 0.2500
  prediction_fidelity:
    Score: 0.9500
    agreement: 0.9500
  manifold_distance:
    Score: 2.5000
    gradient_distance: 0.8500
  perturbation_robustness:
    Score: 0.6500
    avg_correlation: 0.3500
  cross_explanation_agreement:
    Score: 0.5500
    avg_agreement: 0.4500
```

---

## 5. Detailed Module Documentation

### 5.1 Data Loading (src/data/datasets.py)

#### Tabular Datasets
```python
from src.data.datasets import get_tabular_loaders

# Load Adult Income dataset
train_loader, test_loader, metadata = get_tabular_loaders("adult_income", batch_size=64)

# Returns:
# - train_loader: DataLoader with batches of {"features", "label", "sensitive"}
# - test_loader: Same structure
# - metadata: {"feature_names", "n_features", "n_classes", "sensitive_attr_name"}
```

**Available datasets:**
- `"adult_income"`: Income prediction (gender bias)
- `"propublica_recidivism"`: COMPAS dataset (race bias)
- `"german_credit"`: Credit risk dataset
- `"communities_crime"`: Crime prediction

#### Image Datasets
```python
from src.data.datasets import get_image_loaders

# Load MNIST
train_loader, test_loader = get_image_loaders("mnist", batch_size=128)

# Returns:
# - train_loader: DataLoader with (image, label) tuples
# - test_loader: Same structure
```

**Available datasets:**
- `"mnist"`: Handwritten digits (28x28, grayscale)
- `"fashion_mnist"`: Fashion items (28x28, grayscale)
- `"cifar10"`: Natural images (32x32, RGB)

### 5.2 Model Architectures (src/models/architectures.py)

#### BiasedClassifier
```python
from src.models.architectures import BiasedClassifier

# Create classifier that uses sensitive attribute heavily
model = BiasedClassifier(
    input_dim=98,
    sensitive_weight=0.9,  # Weight on sensitive attribute
    other_weight=0.1,     # Weight on other features
    num_classes=2
)

# Manually set biased weights
model.set_biased_weights(sensitive_idx=0)
```

#### Conv4 (4-layer CNN for MNIST/Fashion-MNIST)
```python
from src.models.architectures import Conv4

model = Conv4(
    num_classes=10,
    input_channels=1  # 1 for grayscale, 3 for RGB
)

# Forward pass
output = model(image_tensor)  # Shape: [batch, 10]
```

#### VGG16 (for CIFAR-10)
```python
from src.models.architectures import VGG16

model = VGG16(num_classes=10)
```

### 5.3 Explanation Methods (src/explanations/methods.py)

```python
from src.explanations.methods import get_explainer

# Get gradient explainer
explainer = get_explainer("gradient", model)

# Generate explanation
explanation = explainer.explain(x, target_class=7)
# Returns: numpy array of same shape as input
```

**Available methods:**
| Method | Description | Best For |
|--------|------------|---------|
| `"gradient"` | Simple grad/gradients | Images |
| `"xgrad"` | Gradient × Input | Tabular |
| `"integrated_gradients"` | Integrated Gradients | Any |
| `"lrp"` | Layer-wise Relevance Propagation | Any |
| `"lime"` | LIME | Tabular |
| `"shap"` | SHAP | Tabular |
| `"deeplift"` | DeepLIFT | Any |

### 5.4 Attacks

#### Paper 1: LaundryML (src/attacks/paper1_laundryml.py)
```python
from src.attacks.paper1_laundryml import LaundryML, RuleList

# Create LaundryML attack
laundry = LaundryML(
    black_box=biased_model,
    sensitive_attr="gender",
    fairness_metric="demographic_parity",
    max_depth=5,
    use_corels=False  # Use CORELS for optimal rule lists if available
)

# Enumerate rule lists
candidates = laundry.enumerate_rule_lists(
    X=train_df,
    y=train_labels,
    beta_values=[0.0, 0.1, 0.5, 0.9],  # Fairness weights
    n_models=50
)

# Find best rationalization
best = laundry.find_best_rationalization(
    target_fairness=0.1,
    min_fidelity=0.8
)

# Analyze fairwashing gap
analysis = laundry.analyze_fairwashing_gap(test_df, test_labels)
# Returns: {"black_box_fairness_violation", "rationalization_fairness_violation", "fairwashing_gap", ...}
```

#### Paper 2: Scaffolding Attack (src/attacks/paper2_fooling_lime_shap.py)
```python
from src.attacks.paper2_fooling_lime_shap import ScaffoldingAttack

# Create scaffolding attack
attack = ScaffoldingAttack(
    base_classifier=biased_model,
    n_decoy_features=2,
    decoy_std=0.1,
    fidelity_threshold=0.95
)

# Check fidelity
fidelity = attack.evaluate_fidelity(X_test)

# Fool LIME
lime_result = attack.fool_lime_explanation(
    X_train,
    x_instance,
    feature_names
)
# Returns: {"explanation": {feature: weight}, "decoy_features": [...], ...}

# Fool SHAP
shap_result = attack.fool_shap_explanation(X_background, x_instance)
# Returns: {"shap_values": [...], ...}

# OOD Detection
from src.attacks.paper2_fooling_lime_shap import OODDetector

ood = OODDetector(n_clusters=10, perturbation_std=0.1)
ood.fit(X_train)
ood_scores = ood.compute_ood_score(X_test)
is_ood = ood.is_ood(X_test)
```

#### Paper 3: Off-Manifold Fairwashing (src/attacks/paper3_off_manifold.py)
```python
from src.attacks.paper3_off_manifold import (
    OffManifoldFairwasher,
    TangentSpaceProjector,
    TSPDefender,
    create_target_explanation
)

# Create fairwasher
fairwasher = OffManifoldFairwasher(
    teacher_model=teacher,
    student_model=student,
    explanation_method="gradient",
    alpha=0.8,  # Weight for explanation loss
    device="cuda"
)

# Create target explanation
target_exp = create_target_explanation(
    shape=(1, 28, 28),
    pattern="center"  # or "random", "number_42"
)

# Train fairwashed model
history = fairwasher.train(
    train_loader,
    target_explanation=target_exp,
    n_epochs=100,
    lr=5e-5
)
# Returns list of {"total_loss", "output_loss", "explanation_loss", ...}

# Evaluate
eval_results = fairwasher.evaluate(test_loader, target_exp)
# Returns: {"output_mse": ..., "explanation_mse": ...}

# TSP Defense
projector = TangentSpaceProjector(n_neighbors=32, n_components=16)
projector.fit(X_train_flat)

tsp_defender = TSPDefender(projector)
tsp_explanation = tsp_defender.compute_tsp_explanation(
    model=student,
    x=input_tensor,
    target_class=5
)
```

#### Paper 4: Fragile Interpretation (src/attacks/paper4_fragile_interpretation.py)
```python
from src.attacks.paper4_fragile_interpretation import (
    TopKAttack,
    MassCenterAttack,
    RandomSignPerturbation,
    compute_interpretation_metrics
)

# Top-K Attack (decrease top-k features' importance)
topk_attack = TopKAttack(
    model=model,
    k=1000,
    epsilon=8.0,
    n_iterations=300,
    step_size=0.5,
    device="cuda"
)
x_perturbed = topk_attack.attack(x, explanation_method="gradient")

# Mass-Center Attack (shift explanation center)
center_attack = MassCenterAttack(
    model=model,
    epsilon=8.0,
    n_iterations=300,
    device="cuda"
)
x_perturbed = center_attack.attack(x, target_center=(0, 0))

# Random Sign Perturbation (baseline)
random_attack = RandomSignPerturbation(
    model=model,
    epsilon=8.0,
    device="cuda"
)
x_perturbed = random_attack.attack(x)

# Compute metrics
metrics = compute_interpretation_metrics(
    orig_exp,
    pert_exp,
    k=1000
)
# Returns: {
#     "spearman_correlation": ...,
#     "top_k_intersection": ...,
#     "center_shift": ...
# }
```

### 5.5 Detection Module (src/detection/detector.py)

```python
from src.detection.detector import (
    FairwashingDetector,
    RuleListFairwashingDetector,
    LIMESHAPFoolingDetector
)

# Ensemble detector
detector = FairwashingDetector(
    methods=[
        "explanation_consistency",
        "prediction_fidelity",
        "manifold_distance",
        "perturbation_robustness",
        "cross_explanation_agreement",
    ],
    weights={
        "explanation_consistency": 0.20,
        "prediction_fidelity": 0.25,
        "manifold_distance": 0.15,
        "perturbation_robustness": 0.20,
        "cross_explanation_agreement": 0.20,
    },
    thresholds={
        "explanation_consistency": 0.7,
        "prediction_fidelity": 0.95,
        "manifold_distance": 2.0,
        "perturbation_robustness": 0.5,
        "cross_explanation_agreement": 0.6,
    }
)

results = detector.detect(
    original_model,
    suspect_model,
    X_test,
    explanation_methods=["gradient", "xgrad"]
)
# Returns: {
#     "ensemble_score": ...,
#     "fairwashing_detected": True/False,
#     "explanation_consistency": {"score": ..., "avg_consistency": ...},
#     "prediction_fidelity": {"score": ..., "agreement": ...},
#     ...
# }

# Rule list detector (Paper 1)
rule_detector = RuleListFairwashingDetector(fairness_threshold=0.05)
detection = rule_detector.detect(
    bb_predictions,
    rule_list_predictions,
    sensitive_attr
)

# LIME/SHAP fooling detector (Paper 2)
lime_detector = LIMESHAPFoolingDetector(n_bootstrap=100)
detection = lime_detector.detect(model, X, feature_names, sensitive_feature)
```

### 5.6 Evaluation (src/evaluation/metrics.py)

```python
from src.evaluation.metrics import ExperimentEvaluator

evaluator = ExperimentEvaluator(save_dir="results")

# Evaluate Paper 1
metrics = evaluator.evaluate_paper1(
    candidates,
    black_box_fairness,
    save_prefix="paper1"
)
# Returns: {
#     "n_candidates": ...,
#     "avg_fidelity": ...,
#     "best_fidelity": ...,
#     "fairwashing_gap": ...,
#     ...
# }

# Evaluate Paper 2
metrics = evaluator.evaluate_paper2(
    lime_result,
    shap_result,
    fidelity,
    save_prefix="paper2"
)

# Evaluate Paper 3
metrics = evaluator.evaluate_paper3(
    history,
    evaluation,
    save_prefix="paper3"
)

# Evaluate Paper 4
metrics = evaluator.evaluate_paper4(
    original_img,
    perturbed_img,
    orig_exp,
    pert_exp,
    metrics,
    save_prefix="paper4"
)
```

### 5.7 Utility Functions (src/utils/helpers.py)

```python
from src.utils.helpers import (
    set_seed,
    compute_demographic_parity,
    compute_equalized_odds,
    compute_fairness_metrics,
    compute_explanation_similarity,
    compute_top_k_intersection,
    compute_center_of_mass
)

# Set random seed
set_seed(42)

# Compute fairness metrics
dp = compute_demographic_parity(predictions, sensitive_attr)
# Returns: max(pred_rate) - min(pred_rate)

eo = compute_equalized_odds(y_true, predictions, sensitive_attr)
# Returns: max(TPR_diff, FPR_diff)

fairness = compute_fairness_metrics(y_true, predictions, sensitive_attr)
# Returns: {"demographic_parity": ..., "equalized_odds": ..., "accuracy": ...}

# Compute explanation similarity
similarity = compute_explanation_similarity(exp1, exp2, metric="ssim")
# metrics: "ssim", "pcc", "spearman", "mse"

# Compute top-k intersection
intersection = compute_top_k_intersection(exp1, exp2, k=100)

# Compute center of mass
center = compute_center_of_mass(explanation)
# Returns: (y_center, x_center)
```

---

## 6. Expected Results

### 6.1 Paper-Specific Metrics

| Paper | Metric | Expected Value | Interpretation |
|-------|--------|--------------|----------------|
| Paper 1 | Fairwashing Gap | > 0.05 | Fairwashing present |
| Paper 1 | Best Fidelity | > 0.80 | Rule list approximates model |
| Paper 1 | Fairness Violation | < 0.10 | Rule list appears fair |
| Paper 2 | Fidelity | > 0.95 | Base predictions preserved |
| Paper 2 | LIME Fooled | True | Decoys get importance |
| Paper 2 | OOD Score (normal) | < 0.2 | Normal data detection |
| Paper 2 | OOD Score (ood) | > 0.5 | OOD data detection |
| Paper 3 | Output MSE | < 0.1 | Predictions match |
| Paper 3 | Exp Similarity | < 0.3 | Explanations differ |
| Paper 3 | TSP Similarity | > 0.7 | Defense effective |
| Paper 4 | Spearman Corr | < 0.5 | Rankings changed |
| Paper 4 | Top-K Intersection | < 0.3 | Different features |
| Paper 4 | Prediction Preserved | > 0.95 | Class unchanged |
| Detection | Ensemble Score | > 0.5 | Fairwashing detected |
| Detection | Confidence | > 0.7 | High confidence |

### 6.2 Output Files Generated

```
results/
├── summary.txt                    # Overall summary
├── paper1_pareto.png             # Fidelity vs fairness plot
├── paper1_candidates.csv         # All candidates
├── paper2_lime.png                # LIME explanation
├── paper2_shap.png               # SHAP values
├── paper3_training.png            # Loss curves
├── paper3_comparison.png         # Explanation comparison
├── paper4_comparison.png         # Visual comparison
├── paper4_aggregate.png          # Aggregate metrics
├── detection_benchmark.png       # Detection scores
└── .gitkeep
```

---

## 7. Data Flow Diagrams

### 7.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN PIPELINE                             │
│                  src/main_pipeline.py                          │
│                                                                 │
│  Command: python src/main_pipeline.py --paper all                │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   PAPER 1   │      │   PAPER 2   │      │   PAPER 3   │
│  LaundryML  │      │ Scaffolding │      │ Off-Manifold│
└──────────────┘      └──────────────┘      └──────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DETECTION MODULE                           │
│              src/detection/detector.py                        │
│                                                             │
│  Methods:                                                   │
│  • explanation_consistency    • perturbation_robustness     │
│  • prediction_fidelity      • cross_explanation_agreement  │
│  • manifold_distance         • tangent_space_projection        │
│  • ood_detection_score     • rule_list_fairness_gap       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                              │
│              src/evaluation/metrics.py                       │
│                                                             │
│  • Metrics computation    • Pareto frontier               │
│  • Visualization       • Summary report                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RESULTS                               │
│                        results/                            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Paper 1 Detailed Flow

```
Input Data (adult_income, COMPAS, etc.)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              DATA LOADING                           │
│   get_tabular_loaders(dataset_name, batch_size)      │
│                                              │
│   Returns:                                      │
│   • train_loader: {"features", "label", "sensitive"}│
│   • test_loader: same                           │
│   • metadata: {n_features, ...}               │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│           BIASED BLACK-BOX MODEL                 │
│        BiasedClassifier (or trained model)        │
│                                            │
│  • sensitive_weight = 0.9                   │
│  • Uses sensitive attribute for prediction      │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              LaundryML ATTACK                     │
│    enumerate_rule_lists()                      │
│                                            │
│  For each beta in [0.0, 0.1, 0.5, 0.9]:     │
│    • Train decision tree on bb predictions       │
│    • Extract rule list                         │
│    • Compute fidelity & fairness              │
│                                            │
│  Returns: [{rule_list, fidelity, fairness}, ...] │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│            EVALUATION                           │
│   ExperimentEvaluator.evaluate_paper1()       │
│                                            │
│  • Find Pareto frontier                      │
│  • Plot fidelity vs fairness                 │
│  • Compute fairwashing gap                  │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│            DETECTION                           │
│   RuleListFairwashingDetector.detect()        │
│                                            │
│  Compare:                                    │
│  • Black-box fairness violation             │
│  • Rule list fairness violation             │
│  • Fairwashing gap = BB - RL                │
│                                            │
│  Detection: gap > 0.05 && high fidelity     │
└─────────────────────────────────────────────────────┘
```

### 7.3 Paper 2 Detailed Flow

```
Input Data + Biased Model
              │
              ▼
┌─────────────────────────────────────────────────────┐
│        SCAFFOLDING ATTACK                          │
│     ScaffoldingAttack(n_decoy)                   │
│                                              │
│  1. Add uncorrelated "decoy" features            │
│     decoy ~ N(0, 0.1)                          │
│                                              │
│  2. Model only uses original features          │
│     (predictions preserved)                    │
│                                              │
│  3. LIME/SHAP attribute importance to decoys    │
│     (hidden bias revealed as decoy importance) │
└─────────────────────────────────────────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
┌──────────┐  ┌──────────┐
│   LIME   │  │   SHAP   │
│ Explain │  │ Explain │
└──────────┘  └──────────┘
       │             │
       ▼             ▼
┌─────────────────────────────────────────────────────┐
│              OOD DETECTION                       │
│           OODDetector                           │
│                                              │
│  • Fit K-means on training data                 │
│  • Compute distance to nearest cluster         │
│  • Normal: low distance                       │
│  • With decoys: high distance (OOD)           │
└─────────────────────────────────────────────────────┘
```

### 7.4 Paper 3 Detailed Flow

```
Teacher Model + Student Model + Image Data
              │
              ▼
┌─────────────────────────────────────────────────────┐
│         OFF-MANIFORM ATTACK                       │
│      OffManifoldFairwasher                      │
│                                            │
│  Loss = (1-α) * output_MSE + α * exp_MSE      │
│                                            │
│  • output_MSE: match teacher predictions      │
│  • exp_MSE: match TARGET explanation         │
│                                            │
│  Student learns:                             │
│  • Same outputs as teacher                 │
│  • Different explanations (arbitrary!)      │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│         TANGENT SPACE PROJECTION                   │
│         (Defense)                              │
│                                            │
│  1. Fit K-nearest neighbors                  │
│  2. Compute local tangent spaces (PCA)        │
│  3. Project explanations onto manifold     │
│                                            │
│  Result: robust explanations that are       │
│  aligned with actual data manifold          │
└─────────────────────────────────────────────────────┘
```

### 7.5 Paper 4 Detailed Flow

```
Input Image + CNN Model
         │
         ▼
┌─────────────────────────────────────────────────────┐
│         ATTACK SELECTION                          │
│                                            │
│  • TopKAttack: decrease top-k importance      │
│  • MassCenterAttack: shift center of mass      │
│  • RandomSignPerturbation: baseline         │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────��─��──────────────┐
│            ITERATIVE OPTIMIZATION              │
│                                            │
│  For iteration in range(n_iterations):       │
│    1. Compute current explanation            │
│    2. Compute attack objective          │
│    3. Gradient step to maximize         │
│    4. Project into ε-ball               │
│    5. Ensure prediction preserved     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│            EVALUATION                         │
│   compute_interpretation_metrics()          │
│                                            │
│  • Spearman correlation (rank change)       │
│  • Top-k intersection (feature change)      │
│  • Center shift (location change)           │
│  • Prediction preserved?                │
└─────────────────────────────────────────────────────┘
```

---

## 8. Configuration Reference

All configuration is in `src/config.py`:

### Paper 1 Config
```python
PAPER1_CONFIG = {
    "datasets": ["adult_income", "propublica_recidivism"],
    "sensitive_attributes": {
        "adult_income": "gender",
        "propublica_recidivism": "race"
    },
    "fairness_metric": "demographic_parity",
    "lambda_range": [0.005, 0.01],
    "beta_range": [0.0, 0.1, 0.2, 0.5, 0.7, 0.9],
    "models_per_experiment": 50,
    "max_rule_list_length": 10,
    "use_corels": False,
    "corels_max_cardinality": 2,
}
```

### Paper 2 Config
```python
PAPER2_CONFIG = {
    "datasets": ["compas", "communities_crime", "german_credit"],
    "ood_detection": {
        "perturbation_std": 0.1,
        "kmeans_clusters": 10,
    },
    "scaffolding": {
        "uncorrelated_features": 2,
        "fidelity_threshold": 0.95,
    },
    "lime_config": {"num_samples": 5000},
    "shap_config": {"background_samples": 100},
}
```

### Paper 3 Config
```python
PAPER3_CONFIG = {
    "datasets": ["mnist", "fashion_mnist", "cifar10"],
    "explanation_methods": ["gradient", "xgrad", "integrated_gradients"],
    "fairwashing": {
        "alpha": 0.8,
        "lr": 5e-5,
        "n_epochs": 100,
    },
    "tangent_space_projection": {
        "neighbours": 32,
        "d_singular": 16,
    },
}
```

### Paper 4 Config
```python
PAPER4_CONFIG = {
    "datasets": ["imagenet", "cifar10"],
    "attacks": {
        "top_k": {"k": 1000, "epsilon": 8, "iterations": 300},
        "mass_center": {"epsilon": 8, "iterations": 300},
    },
    "metrics": {
        "spearman_correlation": True,
        "top_k_intersection": True,
    },
}
```

### Detection Config
```python
DETECTION_CONFIG = {
    "methods": [
        "explanation_consistency",
        "prediction_fidelity",
        "manifold_distance",
        "perturbation_robustness",
        "cross_explanation_agreement",
        "tangent_space_projection",
        "ood_detection_score",
        "rule_list_fairness_gap",
    ],
    "ensemble_weights": {
        "explanation_consistency": 0.15,
        "prediction_fidelity": 0.20,
        "manifold_distance": 0.15,
        "perturbation_robustness": 0.15,
        "cross_explanation_agreement": 0.15,
    },
}
```

---

## 9. API Reference

### Quick Reference

```python
# Import everything
from src.main_pipeline import (
    run_paper1_experiment,
    run_paper2_experiment,
    run_paper3_experiment,
    run_paper4_experiment,
    run_detection_experiment
)

# Run experiments
results1 = run_paper1_experiment("adult_income")
results2 = run_paper2_experiment("compas")
results3 = run_paper3_experiment("fashion_mnist")
results4 = run_paper4_experiment("mnist")
results_det = run_detection_experiment()
```

---

## 10. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ImportError: No module named 'src' | Add project root to PYTHONPATH |
| CUDA out of memory | Reduce batch_size |
| LIME/SHAP not available | pip install lime shap |
| CORELS not available | Install CORELS or use fallback |
| Dataset download error | Check internet connection |

### Getting Help

```bash
# Run with debugging
python -u src/main_pipeline.py --paper all 2>&1 | tee debug.log

# Run tests
python -m pytest tests/ -v
```