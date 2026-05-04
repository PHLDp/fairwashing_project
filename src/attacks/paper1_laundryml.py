"""
Paper 1: LaundryML - Fairwashing: The Risk of Rationalization (Aivodji et al., ICML 2019)

This module implements:
1. Rule list enumeration for rationalization
2. Global and local fairwashing scenarios
3. Fidelity vs fairness trade-off analysis
4. CORELS integration for optimal rule lists (NEW)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

# Try to import CORELS wrapper
try:
    from .corels_wrapper import CORELSWrapper, FairCORELS, LaundryML_CORELS
    CORELS_AVAILABLE = True
except ImportError:
    CORELS_AVAILABLE = False
    warnings.warn("CORELS wrapper not available. Using DecisionTree fallback.")


class RuleList:
    """Represents an interpretable rule list."""

    def __init__(self, rules: List[Tuple[str, int]], default_class: int):
        """
        Args:
            rules: List of (condition, prediction) tuples
            default_class: Default prediction if no rule fires
        """
        self.rules = rules
        self.default_class = default_class

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using rule list."""
        predictions = []
        for idx in range(len(X)):
            x = X.iloc[idx]
            pred = self.default_class
            for condition, class_pred in self.rules:
                if self._evaluate_condition(condition, x):
                    pred = class_pred
                    break
            predictions.append(pred)
        return np.array(predictions)

    def _evaluate_condition(self, condition: str, x: pd.Series) -> bool:
        """Evaluate a condition on a data point. Supports compound AND conditions."""
        try:
            # Handle compound AND conditions
            if " AND " in condition:
                sub_conditions = condition.split(" AND ")
                return all(self._evaluate_single_condition(sc.strip(), x)
                           for sc in sub_conditions)
            else:
                return self._evaluate_single_condition(condition, x)
        except:
            return False

    def _evaluate_single_condition(self, condition: str, x: pd.Series) -> bool:
        """Evaluate a single condition like 'feature > value'."""
        try:
            if "<=" in condition:
                parts = condition.split("<=")
                feature = parts[0].strip()
                value = float(parts[1].strip())
                return x[feature] <= value
            elif ">" in condition:
                parts = condition.split(">")
                feature = parts[0].strip()
                value = float(parts[1].strip())
                return x[feature] > value
            elif "==" in condition:
                parts = condition.split("==")
                feature = parts[0].strip()
                value = parts[1].strip().strip("'")
                return str(x[feature]) == value
        except:
            return False
        return False

    def __str__(self):
        lines = ["IF"]
        for i, (condition, pred) in enumerate(self.rules):
            prefix = "   " if i > 0 else ""
            lines.append(f"{prefix}{condition} THEN predict {pred}")
        lines.append(f"ELSE predict {self.default_class}")
        return " ".join(lines)


class LaundryML:
    """
    LaundryML: Regularized rule list enumeration for fairwashing.

    Simplified implementation that extracts rule lists from decision trees
    and evaluates their fidelity and fairness.

    Now supports CORELS for optimal rule list enumeration.
    """

    def __init__(self, black_box, sensitive_attr: str,
                 fairness_metric: str = "demographic_parity",
                 max_depth: int = 5, min_samples_leaf: int = 10,
                 use_corels: bool = False):
        """
        Args:
            black_box: Black-box model to rationalize
            sensitive_attr: Name of sensitive attribute
            fairness_metric: Fairness metric to optimize
            max_depth: Maximum depth of decision trees
            min_samples_leaf: Minimum samples per leaf
            use_corels: Whether to use CORELS for optimal rule lists
        """
        self.black_box = black_box
        self.sensitive_attr = sensitive_attr
        self.fairness_metric = fairness_metric
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.use_corels = use_corels and CORELS_AVAILABLE
        self.candidates = []

        if self.use_corels:
            print("Using CORELS for optimal rule list enumeration.")
        else:
            print("Using DecisionTree fallback for rule list enumeration.")

    def enumerate_rule_lists(self, X: pd.DataFrame, y: np.ndarray,
                            beta_values: List[float] = [0.0, 0.5, 0.9],
                            n_models: int = 50) -> List[Dict]:
        """
        Enumerate rule lists with different fairness-accuracy trade-offs.

        If use_corels=True, uses CORELS for optimal enumeration.
        Otherwise, uses DecisionTree-based fallback.

        Args:
            X: Feature dataframe
            y: True labels
            beta_values: Different fairness weights to try
            n_models: Number of models per beta

        Returns:
            List of candidate models with metrics
        """
        if self.use_corels:
            return self._enumerate_with_corels(X, y, beta_values, n_models)
        else:
            return self._enumerate_with_trees(X, y, beta_values, n_models)

    def _enumerate_with_corels(self, X: pd.DataFrame, y: np.ndarray,
                              beta_values: List[float],
                              n_models: int) -> List[Dict]:
        """Enumerate using CORELS (optimal)."""
        # Get black-box predictions
        bb_preds = self._get_black_box_predictions(X)

        # Prepare data
        feature_names = [c for c in X.columns if c != self.sensitive_attr]
        X_np = X[feature_names].values
        sensitive = X[self.sensitive_attr].values

        candidates = []

        for beta in beta_values:
            # Create FairCORELS with different fairness weights
            fair_corels = FairCORELS(
                fairness_metric=self.fairness_metric,
                fairness_weight=beta,
                max_length=self.max_depth,
                c=0.01 * (1 + beta)
            )

            # Fit on black-box predictions
            results = fair_corels.fit(X_np, bb_preds, sensitive, feature_names)

            for result in results:
                wrapper = result["rule_list"]
                rl_preds = wrapper.predict(X_np)

                fidelity = accuracy_score(bb_preds, rl_preds)
                fairness = result["fairness_violation"]

                candidates.append({
                    "rule_list": wrapper,
                    "fidelity": fidelity,
                    "fairness_violation": fairness,
                    "accuracy": accuracy_score(y, rl_preds),
                    "beta": beta,
                    "objective": beta * (1 - fairness) + (1 - beta) * fidelity,
                    "rule_list_str": wrapper.get_rule_list_str(),
                    "is_corels": True,
                })

        self.candidates = candidates
        return candidates

    def _enumerate_with_trees(self, X: pd.DataFrame, y: np.ndarray,
                             beta_values: List[float],
                             n_models: int) -> List[Dict]:
        """Enumerate using DecisionTree fallback with maximum diversity."""
        # Get black-box predictions
        bb_preds = self._get_black_box_predictions(X)

        candidates = []
        n_per_beta = max(1, -(-n_models // len(beta_values)))  # Ceiling division

        criteria = ["gini", "entropy"]
        splitters = ["best", "random"]

        for beta in beta_values:
            for i in range(n_per_beta):
                # Widely vary tree parameters
                depth = np.random.randint(1, self.max_depth + 3)
                min_samples = np.random.randint(2, 50)
                max_features_options = [None, "sqrt", "log2",
                                        max(1, int(0.3 * X.shape[1])),
                                        max(1, int(0.5 * X.shape[1])),
                                        max(1, int(0.7 * X.shape[1]))]
                max_feat = max_features_options[i % len(max_features_options)]
                criterion = criteria[i % len(criteria)]
                splitter = splitters[(i // 2) % len(splitters)]

                # Use beta to create sample weights that emphasize/
                # de-emphasize sensitive-attribute-correlated samples
                sensitive_col = self.sensitive_attr
                if sensitive_col in X.columns:
                    sensitive_vals = X[sensitive_col].values
                    # beta > 0 → upweight minority group for fairer trees
                    group_counts = np.bincount(sensitive_vals.astype(int))
                    minority_group = np.argmin(group_counts)
                    sample_weight = np.ones(len(X))
                    if beta > 0:
                        boost = 1.0 + beta * (2.0 + i * 0.1)
                        sample_weight[sensitive_vals.astype(int) == minority_group] = boost
                    # Also add small random noise to weights for diversity
                    sample_weight *= (1.0 + np.random.randn(len(X)) * 0.05 * (i + 1))
                    sample_weight = np.clip(sample_weight, 0.1, 10.0)
                else:
                    sample_weight = None

                # Train decision tree on black-box predictions
                dt = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_samples,
                    max_features=max_feat,
                    criterion=criterion,
                    splitter=splitter,
                    random_state=42 + i * 7 + int(beta * 100)
                )
                dt.fit(X, bb_preds, sample_weight=sample_weight)

                # Extract rule list
                rule_list = self._tree_to_rule_list(dt, X.columns)

                # Compute metrics
                rl_preds = rule_list.predict(X)
                fidelity = accuracy_score(bb_preds, rl_preds)

                fairness = self._compute_fairness(rl_preds, X)
                accuracy = accuracy_score(y, rl_preds)

                # Combined objective: beta * fairness + (1-beta) * fidelity
                objective = beta * (1 - fairness) + (1 - beta) * fidelity

                candidates.append({
                    "rule_list": rule_list,
                    "fidelity": fidelity,
                    "fairness_violation": fairness,
                    "accuracy": accuracy,
                    "beta": beta,
                    "objective": objective,
                    "depth": depth,
                    "is_corels": False,
                })

        self.candidates = candidates
        return candidates

    def _get_black_box_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from black-box model."""
        # Handle different model types
        if hasattr(self.black_box, 'predict'):
            return self.black_box.predict(X)
        else:
            # PyTorch model
            import torch
            self.black_box.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X.values)
                output = self.black_box(X_tensor)
                return output.argmax(dim=1).numpy()

    def _tree_to_rule_list(self, tree: DecisionTreeClassifier, 
                          feature_names: List[str]) -> RuleList:
        """Extract rule list from decision tree."""
        from sklearn.tree import _tree

        tree_ = tree.tree_
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]

                # Left branch: <= threshold
                left_conditions = conditions + [f"{name} <= {threshold:.3f}"]
                recurse(tree_.children_left[node], left_conditions)

                # Right branch: > threshold
                right_conditions = conditions + [f"{name} > {threshold:.3f}"]
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf node
                prediction = np.argmax(tree_.value[node])
                if conditions:
                    # Combine conditions with AND
                    rule = " AND ".join(conditions)
                    rules.append((rule, prediction))

        recurse(0, [])

        # Default class is majority class
        default_class = np.argmax(tree.tree_.value[0])

        return RuleList(rules, default_class)

    def _compute_fairness(self, predictions: np.ndarray, X: pd.DataFrame) -> float:
        """Compute fairness violation."""
        sensitive = X[self.sensitive_attr].values

        if self.fairness_metric == "demographic_parity":
            groups = np.unique(sensitive)
            rates = [np.mean(predictions[sensitive == g]) for g in groups]
            return np.max(rates) - np.min(rates)

        elif self.fairness_metric == "equalized_odds":
            # Simplified: just demographic parity for now
            groups = np.unique(sensitive)
            rates = [np.mean(predictions[sensitive == g]) for g in groups]
            return np.max(rates) - np.min(rates)

        return 0.0

    def find_best_rationalization(self, target_fairness: float = 0.1,
                                  min_fidelity: float = 0.8) -> Optional[Dict]:
        """
        Find the best rationalization that appears fair while maintaining fidelity.

        Args:
            target_fairness: Target fairness violation (lower is better)
            min_fidelity: Minimum fidelity to black-box

        Returns:
            Best candidate or None
        """
        if not self.candidates:
            raise ValueError("No candidates. Run enumerate_rule_lists first.")

        # Filter by fidelity
        valid = [c for c in self.candidates if c["fidelity"] >= min_fidelity]

        if not valid:
            return None

        # Find the one with best (apparent) fairness
        best = min(valid, key=lambda c: c["fairness_violation"])

        return best

    def analyze_fairwashing_gap(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Analyze the gap between apparent fairness and true fairness.

        Returns metrics showing how much fairwashing is occurring.
        """
        bb_preds = self._get_black_box_predictions(X)
        bb_fairness = self._compute_fairness(bb_preds, X)

        best_rational = self.find_best_rationalization()

        if best_rational is None:
            return {"fairwashing_gap": 0.0, "message": "No valid rationalization found"}

        rl_fairness = best_rational["fairness_violation"]

        return {
            "black_box_fairness_violation": bb_fairness,
            "rationalization_fairness_violation": rl_fairness,
            "fairwashing_gap": bb_fairness - rl_fairness,
            "fidelity": best_rational["fidelity"],
            "fairwashing_detected": bb_fairness > rl_fairness + 0.05,
            "used_corels": best_rational.get("is_corels", False),
        }


def run_laundryml_experiment(X_train, y_train, X_test, y_test, 
                             black_box, sensitive_attr: str,
                             beta_values: List[float] = [0.0, 0.1, 0.5, 0.9],
                             use_corels: bool = False) -> Dict:
    """
    Run a complete LaundryML experiment.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        black_box: Black-box model
        sensitive_attr: Sensitive attribute name
        beta_values: Fairness-accuracy trade-off values
        use_corels: Whether to use CORELS for optimal rule lists

    Returns:
        Dictionary with experiment results
    """
    laundry = LaundryML(black_box, sensitive_attr, use_corels=use_corels)

    # Enumerate candidates
    candidates = laundry.enumerate_rule_lists(X_train, y_train, beta_values)

    # Analyze on test set
    analysis = laundry.analyze_fairwashing_gap(X_test, y_test)

    return {
        "candidates": candidates,
        "analysis": analysis,
        "n_candidates": len(candidates),
        "used_corels": use_corels and CORELS_AVAILABLE,
    }
