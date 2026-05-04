"""Fairwashing Detection and Analysis Framework."""

from src.attacks.paper1_laundryml import LaundryML, run_laundryml_experiment
from src.attacks.paper2_fooling_lime_shap import ScaffoldingAttack, BiasedClassifierFactory, run_scaffolding_experiment
from src.attacks.paper3_off_manifold import OffManifoldFairwasher, TangentSpaceProjector, TSPDefender, create_target_explanation
from src.attacks.paper4_fragile_interpretation import TopKAttack, MassCenterAttack, RandomSignPerturbation, compute_interpretation_metrics
