"""
Configuration file for Fairwashing Detection Framework.
Contains all hyperparameters and settings for the four papers.
"""

import torch

__all__ = [
    "RANDOM_SEED", "DEVICE",
    "PAPER1_CONFIG", "PAPER2_CONFIG", "PAPER3_CONFIG", "PAPER4_CONFIG",
    "DETECTION_CONFIG", "MODEL_CONFIGS",
]

# General settings
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PAPER 1: LaundryML - Fairwashing: The Risk of Rationalization (Aivodji et al.)
# =============================================================================
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
    "scenarios": ["global", "local"],
    # NEW: CORELS settings
    "use_corels": False,  # Set to True to use optimal rule list enumeration
    "corels_max_cardinality": 2,
    "corels_min_support": 0.01,
    "corels_c": 0.01,
}

# =============================================================================
# PAPER 2: Fooling LIME and SHAP (Slack et al.)
# =============================================================================
PAPER2_CONFIG = {
    "datasets": ["compas", "communities_crime", "german_credit"],
    "sensitive_attributes": {
        "compas": "race",
        "communities_crime": "race",
        "german_credit": "gender"
    },
    "ood_detection": {
        "perturbation_std": 0.1,
        "kmeans_clusters": 10,
        "kernel_width_factor": 0.75,
    },
    "scaffolding": {
        "uncorrelated_features": 2,
        "fidelity_threshold": 0.95,
    },
    "explanation_methods": ["lime", "shap"],
    "lime_config": {
        "num_samples": 5000,
        "kernel_width": None,
    },
    "shap_config": {
        "background_samples": 100,
        "nsamples": 100,
    },
    # NEW: Image scaffolding settings
    "image_scaffolding": {
        "n_decoy_segments": 5,
        "segmentation_method": "slic",
        "n_segments": 50,
    }
}

# =============================================================================
# PAPER 3: Fairwashing with Off-Manifold Detergent (Anders et al.)
# =============================================================================
PAPER3_CONFIG = {
    "datasets": ["mnist", "fashion_mnist", "cifar10"],
    "explanation_methods": ["gradient", "xgrad", "integrated_gradients", "lrp"],
    "fairwashing": {
        "alpha": 0.8,
        "lr": 5e-5,
        "n_epochs": 100,
        "batch_size": 128,
        "optimizer": "adam",
        "loss_type": "mse",
    },
    "tangent_space_projection": {
        "neighbours": 32,
        "d_singular": 16,
        "strict_mode": False,
    },
    "target_explanation": "42.png",
    "models": {
        "mnist": "conv4",
        "fashion_mnist": "conv4",
        "cifar10": "vgg16"
    },
    # NEW: Full LRP settings
    "lrp": {
        "rule": "z+",
        "epsilon": 1e-6,
        "first_layer_rule": "zB",
    }
}

# =============================================================================
# PAPER 4: Interpretation of Neural Networks is Fragile (Ghorbani et al.)
# =============================================================================
PAPER4_CONFIG = {
    "datasets": ["imagenet", "cifar10"],
    "explanation_methods": ["simple_gradient", "integrated_gradients", "deeplift"],
    "attacks": {
        "top_k": {
            "k": 1000,
            "epsilon": 8,
            "iterations": 300,
            "step_size": 0.5,
        },
        "mass_center": {
            "epsilon": 8,
            "iterations": 300,
            "step_size": 0.5,
        },
        "targeted": {
            "epsilon": 8,
            "iterations": 300,
            "step_size": 0.5,
            "target_region": None,
        },
        "random_sign": {
            "epsilon": 8,
        }
    },
    "metrics": {
        "spearman_correlation": True,
        "top_k_intersection": True,
        "center_shift": True,
    },
    "influence_functions": {
        "top_influential": 3,
        "hessian_approximation": "cg",
    },
    # NEW: Image LIME/SHAP settings
    "image_explanations": {
        "segmentation_method": "slic",
        "n_segments": 50,
        "compactness": 10.0,
    }
}

# =============================================================================
# DETECTION MODULE CONFIGURATION
# =============================================================================
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
    "thresholds": {
        "explanation_consistency": 0.7,
        "prediction_fidelity": 0.95,
        "manifold_distance": 2.0,
        "perturbation_robustness": 0.5,
        "cross_explanation_agreement": 0.6,
    },
    "ensemble_weights": {
        "explanation_consistency": 0.15,
        "prediction_fidelity": 0.20,
        "manifold_distance": 0.15,
        "perturbation_robustness": 0.15,
        "cross_explanation_agreement": 0.15,
        "tangent_space_projection": 0.10,
        "ood_detection_score": 0.10,
    }
}

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================
MODEL_CONFIGS = {
    "conv4_mnist": {
        "conv_layers": [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ],
        "pool_size": 2,
        "fc_layers": [128, 10],
        "dropout": 0.5,
    },
    "vgg16_cifar": {
        "type": "vgg16",
        "num_classes": 10,
        "pretrained": False,
    }
}
