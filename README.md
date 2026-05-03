# Fairwashing Detection Framework

A comprehensive codebase implementing and detecting fairwashing attacks across four major research papers in ML explainability and fairness.

## Overview

This framework implements:
1. **Paper 1**: LaundryML - Fairwashing via rule list rationalization (Aivodji et al., ICML 2019)
2. **Paper 2**: Fooling LIME and SHAP via scaffolding attacks (Slack et al., AIES 2020)
3. **Paper 3**: Off-manifold fairwashing with tangent space projection defense (Anders et al., ICML 2020)
4. **Paper 4**: Fragile interpretation via adversarial perturbations (Ghorbani et al., AAAI 2019)

Plus a **Detection Module** that combines multiple signals to detect fairwashing attempts.

## Project Structure

```
fairwashing_project/
├── src/
│   ├── config.py                          # Configuration for all papers
│   ├── data/
│   │   └── datasets.py                    # Data loaders (tabular + image)
│   ├── models/
│   │   └── architectures.py             # Model architectures
│   ├── explanations/
│   │   └── methods.py                     # Explanation methods (Grad, IG, LRP, LIME, SHAP)
│   ├── attacks/
│   │   ├── paper1_laundryml.py          # Rule list fairwashing
│   │   ├── paper2_fooling_lime_shap.py   # Scaffolding attacks
│   │   ├── paper3_off_manifold.py        # Off-manifold manipulation + TSP
│   │   └── paper4_fragile_interpretation.py # Adversarial attacks on explanations
│   ├── detection/
│   │   └── detector.py                    # Ensemble fairwashing detector
│   ├── evaluation/
│   │   └── metrics.py                     # Evaluation and visualization
│   ├── utils/
│   │   └── helpers.py                     # Utility functions
│   └── main_pipeline.py                   # Main pipeline runner
├── experiments/
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
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
cd fairwashing_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Individual Paper Experiments

```bash
# Paper 1: LaundryML - Rule list fairwashing
python experiments/paper1_laundryml/run_experiment.py --dataset adult_income --n_candidates 50

# Paper 2: Fooling LIME and SHAP
python experiments/paper2_fooling_lime_shap/run_experiment.py --dataset compas --n_decoy 2

# Paper 3: Off-Manifold Fairwashing
python experiments/paper3_off_manifold/run_experiment.py --dataset fashion_mnist --n_epochs 20

# Paper 4: Fragile Interpretation
python experiments/paper4_fragile_interpretation/run_experiment.py --dataset mnist --n_samples 5

# Detection Module
python experiments/detection/run_experiment.py --mode comprehensive
```

### Run Full Pipeline

```bash
# Run all experiments
python src/main_pipeline.py --paper all

# Run specific paper
python src/main_pipeline.py --paper 1 --dataset adult_income

# Run only detection
python src/main_pipeline.py --paper detection
```

## Module Details

### Paper 1: LaundryML (Aivodji et al.)

**What it does:**
- Creates a biased black-box classifier
- Enumerates interpretable rule lists that approximate the black-box
- Finds rule lists that appear fair while maintaining high fidelity
- Demonstrates the "fairwashing gap" between true and apparent fairness

**Key classes:**
- `RuleList`: Interpretable rule list representation
- `LaundryML`: Main fairwashing algorithm
- `RuleListFairwashingDetector`: Detection for rule list fairwashing

**Usage:**
```python
from src.attacks.paper1_laundryml import LaundryML
from src.models.architectures import BiasedClassifier

# Create biased model
black_box = BiasedClassifier(input_dim=10, sensitive_weight=0.9)

# Run LaundryML
laundry = LaundryML(black_box, "gender")
candidates = laundry.enumerate_rule_lists(X_train, y_train, beta_values=[0.0, 0.5, 0.9])

# Detect fairwashing
best = laundry.find_best_rationalization()
```

### Paper 2: Fooling LIME and SHAP (Slack et al.)

**What it does:**
- Creates a biased classifier
- Adds uncorrelated decoy features to the input
- Shows that LIME and SHAP attribute importance to decoy features
- Hides the true bias in the black-box model
- Includes OOD detection mechanism

**Key classes:**
- `ScaffoldingAttack`: Main attack class
- `OODDetector`: Out-of-distribution detector
- `BiasedClassifierFactory`: Creates intentionally biased models

**Usage:**
```python
from src.attacks.paper2_fooling_lime_shap import ScaffoldingAttack

# Create attack
attack = ScaffoldingAttack(biased_model, n_decoy_features=2)

# Fool LIME
lime_result = attack.fool_lime_explanation(X_train, x_instance, feature_names)

# Fool SHAP
shap_result = attack.fool_shap_explanation(X_background, x_instance)
```

### Paper 3: Off-Manifold Detergent (Anders et al.)

**What it does:**
- Trains a student model to match teacher outputs on data manifold
- Manipulates explanations orthogonal to data manifold
- Demonstrates arbitrary explanation manipulation
- Implements Tangent Space Projection (TSP) defense

**Key classes:**
- `OffManifoldFairwasher`: Main fairwashing trainer
- `TangentSpaceProjector`: Computes tangent space projectors
- `TSPDefender`: Defense using tangent space projection

**Usage:**
```python
from src.attacks.paper3_off_manifold import OffManifoldFairwasher, TangentSpaceProjector

# Create fairwasher
fairwasher = OffManifoldFairwasher(teacher, student, alpha=0.8)

# Train
history = fairwasher.train(train_loader, target_explanation, n_epochs=100)

# TSP Defense
projector = TangentSpaceProjector(n_neighbors=32, n_components=16)
projector.fit(X_train)
tsp_defender = TSPDefender(projector)
```

### Paper 4: Fragile Interpretation (Ghorbani et al.)

**What it does:**
- Generates adversarial perturbations that preserve predictions
- Dramatically changes feature importance explanations
- Implements top-k, mass-center, and targeted attacks
- Measures fragility via rank correlation and top-k intersection

**Key classes:**
- `TopKAttack`: Decrease top-k feature importance
- `MassCenterAttack`: Shift explanation center of mass
- `RandomSignPerturbation`: Baseline random perturbation
- `InfluenceFunctionAttack`: Attack on influence functions

**Usage:**
```python
from src.attacks.paper4_fragile_interpretation import TopKAttack, compute_interpretation_metrics

# Create attack
attack = TopKAttack(model, k=1000, epsilon=8.0, n_iterations=300)

# Generate perturbed image
x_perturbed = attack.attack(x, explanation_method="gradient")

# Compute metrics
metrics = compute_interpretation_metrics(orig_exp, pert_exp, k=1000)
```

### Detection Module

**What it does:**
- Combines multiple detection signals into ensemble score
- Detects fairwashing across all four paper scenarios
- Provides confidence scores and per-method diagnostics

**Detection methods:**
1. **Explanation Consistency**: Compare explanations between models
2. **Prediction Fidelity**: Check output agreement
3. **Manifold Distance**: Measure gradient differences
4. **Perturbation Robustness**: Test explanation stability
5. **Cross-Explanation Agreement**: Check if different methods agree
6. **Tangent Space Projection**: Project onto data manifold
7. **OOD Detection Score**: Detect out-of-distribution inputs
8. **Rule List Fairness Gap**: Detect fairness gap in rule lists

**Usage:**
```python
from src.detection.detector import FairwashingDetector

# Create detector
detector = FairwashingDetector(
    methods=["explanation_consistency", "prediction_fidelity", "manifold_distance"],
    weights={"explanation_consistency": 0.3, "prediction_fidelity": 0.4, "manifold_distance": 0.3}
)

# Detect
results = detector.detect(original_model, suspect_model, X_test)
print(f"Fairwashing detected: {results['fairwashing_detected']}")
print(f"Confidence: {results['ensemble_score']:.3f}")
```

## Running Tests

```bash
python -m pytest tests/test_framework.py -v
```

## Datasets

### Tabular
- **Adult Income**: Gender-based income prediction
- **COMPAS (ProPublica Recidivism)**: Race-based recidivism prediction
- **German Credit**: Credit risk with gender bias
- **Communities and Crime**: Crime prediction with racial bias

### Image
- **MNIST**: Handwritten digits
- **Fashion-MNIST**: Fashion items
- **CIFAR-10**: Natural images

## Citation

If you use this framework, please cite the original papers:

```bibtex
@inproceedings{aivodji2019fairwashing,
  title={Fairwashing: the risk of rationalization},
  author={A{"i}vodji, Ulrich and Arai, Hiromi and Fortineau, Olivier and Gambs, S{'e}bastien and Hara, Satoshi and Tapp, Alain},
  booktitle={International Conference on Machine Learning},
  pages={161--170},
  year={2019}
}

@inproceedings{slack2020fooling,
  title={Fooling lime and shap: Adversarial attacks on post hoc explanation methods},
  author={Slack, Dylan and Hilgard, Sophie and Jia, Emily and Singh, Sameer and Lakkaraju, Himabindu},
  booktitle={AAAI/ACM Conference on AI, Ethics, and Society},
  year={2020}
}

@inproceedings{anders2020fairwashing,
  title={Fairwashing explanations with off-manifold detergent},
  author={Anders, Christopher J and Pasliev, Plamen and Dombrowski, Ann-Kathrin and M{"u}ller, Klaus-Robert and Kessel, Pan},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

@inproceedings{ghorbani2019interpretation,
  title={Interpretation of neural networks is fragile},
  author={Ghorbani, Amirata and Abid, Abubakar and Zou, James},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3681--3688},
  year={2019}
}
```

