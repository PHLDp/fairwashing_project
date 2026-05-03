#!/bin/bash
# Run all experiments for Fairwashing Detection Framework

echo "=========================================="
echo "Fairwashing Detection Framework"
echo "Running All Experiments"
echo "=========================================="

# Create results directory
mkdir -p results

# Paper 1: LaundryML
echo ""
echo "Running Paper 1: LaundryML..."
python experiments/paper1_laundryml/run_experiment.py --dataset adult_income --n_candidates 50

# Paper 2: Fooling LIME and SHAP
echo ""
echo "Running Paper 2: Fooling LIME and SHAP..."
python experiments/paper2_fooling_lime_shap/run_experiment.py --dataset compas --n_decoy 2

# Paper 3: Off-Manifold Fairwashing
echo ""
echo "Running Paper 3: Off-Manifold Fairwashing..."
python experiments/paper3_off_manifold/run_experiment.py --dataset fashion_mnist --n_epochs 20

# Paper 4: Fragile Interpretation
echo ""
echo "Running Paper 4: Fragile Interpretation..."
python experiments/paper4_fragile_interpretation/run_experiment.py --dataset mnist --n_samples 5

# Detection Module
echo ""
echo "Running Detection Module..."
python experiments/detection/run_experiment.py --mode comprehensive

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in results/ directory"
echo "=========================================="
