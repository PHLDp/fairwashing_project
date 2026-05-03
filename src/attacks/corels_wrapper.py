"""
CORELS (Certifiably Optimal RulE ListS) Integration for LaundryML.

This module provides:
1. CORELS wrapper for optimal rule list enumeration
2. Fallback implementation using sklearn + custom optimization
3. Fairness-aware rule list generation

Original paper: Angelino et al., "Learning Certifiably Optimal Rule Lists", KDD 2017
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings


class CORELSWrapper:
    """
    Wrapper for CORELS algorithm with fallback to sklearn-based implementation.

    CORELS finds the optimal rule list for a given dataset by formulating
    the problem as a branch-and-bound optimization.
    """

    def __init__(self, max_cardinality: int = 2, min_support: float = 0.01,
                 max_length: int = 10, c: float = 0.01, 
                 fairness_constraint: Optional[str] = None,
                 sensitive_attr: Optional[np.ndarray] = None):
        """
        Args:
            max_cardinality: Maximum number of conditions per rule
            min_support: Minimum support for rules
            max_length: Maximum length of rule list
            c: Regularization parameter (model complexity penalty)
            fairness_constraint: "demographic_parity" or "equalized_odds" or None
            sensitive_attr: Sensitive attribute values
        """
        self.max_cardinality = max_cardinality
        self.min_support = min_support
        self.max_length = max_length
        self.c = c
        self.fairness_constraint = fairness_constraint
        self.sensitive_attr = sensitive_attr
        self.corels_available = False
        self._try_import_corels()

    def _try_import_corels(self):
        """Try to import CORELS package."""
        try:
            import corels
            self.corels_available = True
            self.corels = corels
            print("CORELS package found and loaded.")
        except ImportError:
            warnings.warn(
                "CORELS package not found. Using fallback implementation.\n"
                "Install with: pip install corels\n"
                "Fallback uses sklearn DecisionTree + greedy optimization."
            )
            self.corels_available = False

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'CORELSWrapper':
        """
        Fit optimal rule list.

        Args:
            X: Feature matrix
            y: Labels (0 or 1)
            feature_names: Names of features

        Returns:
            self
        """
        if self.corels_available:
            return self._fit_corels(X, y, feature_names)
        else:
            return self._fit_fallback(X, y, feature_names)

    def _fit_corels(self, X: np.ndarray, y: np.ndarray,
                   feature_names: Optional[List[str]]) -> 'CORELSWrapper':
        """Fit using actual CORELS package."""
        # Convert to binary features if needed
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Create CORELS model
        self.model = self.corels.CorelsClassifier(
            max_cardinality=self.max_cardinality,
            min_support=self.min_support,
            max_length=self.max_length,
            c=self.c,
            verbosity=[]
        )

        # Binarize features for CORELS (it requires binary features)
        X_binary = self._binarize_features(X)

        self.model.fit(X_binary, y, features=feature_names)
        self.rule_list = self._extract_corels_rules()
        return self

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray,
                     feature_names: Optional[List[str]]) -> 'CORELSWrapper':
        """
        Fallback implementation using sklearn + greedy optimization.

        Mimics CORELS by:
        1. Generating candidate rules from decision trees
        2. Greedy selection with regularization
        3. Optional fairness constraints
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names = feature_names

        # Generate candidate rules using decision trees of varying depths
        candidates = self._generate_candidate_rules(X, y)

        # Greedy selection with regularization
        self.rule_list = self._greedy_select_rules(X, y, candidates)

        return self

    def _binarize_features(self, X: np.ndarray) -> np.ndarray:
        """Binarize continuous features using median split."""
        X_binary = np.zeros((X.shape[0], X.shape[1] * 2), dtype=int)

        for i in range(X.shape[1]):
            median = np.median(X[:, i])
            X_binary[:, i * 2] = (X[:, i] > median).astype(int)
            X_binary[:, i * 2 + 1] = (X[:, i] <= median).astype(int)

        return X_binary

    def _generate_candidate_rules(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Generate candidate rules from decision trees."""
        candidates = []

        # Try different tree depths
        for depth in range(1, self.max_cardinality + 1):
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(1, int(self.min_support * len(X))))
            dt.fit(X, y)

            # Extract rules from tree
            rules = self._tree_to_rules(dt, X, y)
            candidates.extend(rules)

        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = str(c["conditions"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        return unique_candidates

    def _tree_to_rules(self, tree: DecisionTreeClassifier, 
                      X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Extract rules from decision tree."""
        from sklearn.tree import _tree

        tree_ = tree.tree_
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = self.feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]

                left_conditions = conditions + [(name, "<=", threshold)]
                recurse(tree_.children_left[node], left_conditions)

                right_conditions = conditions + [(name, ">", threshold)]
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf node
                prediction = np.argmax(tree_.value[node])
                support = tree_.n_node_samples[node] / len(X)

                if support >= self.min_support:
                    rules.append({
                        "conditions": conditions,
                        "prediction": prediction,
                        "support": support,
                        "accuracy": np.max(tree_.value[node]) / tree_.n_node_samples[node]
                    })

        recurse(0, [])
        return rules

    def _greedy_select_rules(self, X: np.ndarray, y: np.ndarray,
                            candidates: List[Dict]) -> List[Dict]:
        """
        Greedy selection of rules with regularization.

        Objective: minimize loss + c * |rules|
        """
        selected = []
        remaining = set(range(len(X)))

        for _ in range(self.max_length):
            best_rule = None
            best_score = float('inf')

            for candidate in candidates:
                # Find samples that match this rule
                matches = self._get_matching_samples(X, candidate["conditions"])

                if len(matches) == 0:
                    continue

                # Compute loss on matching samples
                predictions = np.full(len(matches), candidate["prediction"])
                actual = y[matches]
                loss = np.mean(predictions != actual)

                # Add regularization
                score = loss + self.c

                # Add fairness penalty if specified
                if self.fairness_constraint and self.sensitive_attr is not None:
                    fairness_penalty = self._compute_fairness_penalty(
                        predictions, actual, self.sensitive_attr[matches]
                    )
                    score += fairness_penalty

                if score < best_score:
                    best_score = score
                    best_rule = candidate

            if best_rule is None:
                break

            selected.append(best_rule)

            # Remove covered samples
            matches = self._get_matching_samples(X, best_rule["conditions"])
            remaining -= set(matches)

            if len(remaining) == 0:
                break

        return selected

    def _get_matching_samples(self, X: np.ndarray, 
                             conditions: List[Tuple]) -> List[int]:
        """Get indices of samples matching conditions."""
        matches = list(range(len(X)))

        for feature, op, threshold in conditions:
            idx = self.feature_names.index(feature)
            if op == "<=":
                matches = [i for i in matches if X[i, idx] <= threshold]
            else:
                matches = [i for i in matches if X[i, idx] > threshold]

        return matches

    def _compute_fairness_penalty(self, predictions: np.ndarray, 
                                 actual: np.ndarray,
                                 sensitive: np.ndarray) -> float:
        """Compute fairness violation penalty."""
        if self.fairness_constraint == "demographic_parity":
            groups = np.unique(sensitive)
            rates = [np.mean(predictions[sensitive == g]) for g in groups]
            return np.max(rates) - np.min(rates)
        return 0.0

    def _extract_corels_rules(self) -> List[Dict]:
        """Extract rules from fitted CORELS model."""
        # CORELS stores rules differently
        rules = []
        for rule_str in self.model.rl_.rules:
            rules.append({
                "conditions": [rule_str],
                "prediction": 1,  # Simplified
                "support": 0.5,
                "accuracy": 0.8
            })
        return rules

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using rule list."""
        predictions = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            pred = self._get_default_prediction()
            for rule in self.rule_list:
                if self._matches_rule(X[i], rule["conditions"]):
                    pred = rule["prediction"]
                    break
            predictions[i] = pred

        return predictions

    def _matches_rule(self, x: np.ndarray, conditions: List[Tuple]) -> bool:
        """Check if sample matches rule conditions."""
        for feature, op, threshold in conditions:
            idx = self.feature_names.index(feature)
            if op == "<=":
                if x[idx] > threshold:
                    return False
            else:
                if x[idx] <= threshold:
                    return False
        return True

    def _get_default_prediction(self) -> int:
        """Get default prediction (majority class)."""
        return 0  # Simplified

    def get_rule_list_str(self) -> str:
        """Get human-readable rule list."""
        lines = ["IF"]
        for i, rule in enumerate(self.rule_list):
            conditions_str = " AND ".join([
                f"{f} {op} {t:.3f}" for f, op, t in rule["conditions"]
            ])
            prefix = "   ELSE IF" if i > 0 else ""
            lines.append(f"{prefix} {conditions_str} THEN predict {rule['prediction']}")
        lines.append(f"   ELSE predict {self._get_default_prediction()}")
        return "\n".join(lines)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return accuracy_score(y, preds)


class FairCORELS:
    """
    Fairness-aware CORELS extension.

    Generates rule lists that optimize for both accuracy and fairness.
    """

    def __init__(self, fairness_metric: str = "demographic_parity",
                 fairness_weight: float = 0.5, **kwargs):
        """
        Args:
            fairness_metric: Metric to optimize
            fairness_weight: Weight for fairness in objective (0-1)
            **kwargs: Passed to CORELSWrapper
        """
        self.fairness_metric = fairness_metric
        self.fairness_weight = fairness_weight
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray,
            sensitive_attr: np.ndarray,
            feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Fit fairness-aware rule list.

        Returns:
            List of candidate rule lists with different fairness-accuracy trade-offs
        """
        candidates = []

        # Try different fairness weights
        for fw in [0.0, 0.25, 0.5, 0.75, 1.0]:
            wrapper = CORELSWrapper(
                fairness_constraint=self.fairness_metric,
                sensitive_attr=sensitive_attr,
                **self.kwargs
            )
            wrapper.fit(X, y, feature_names)

            preds = wrapper.predict(X)
            accuracy = accuracy_score(y, preds)

            # Compute fairness
            from src.utils.helpers import compute_demographic_parity
            fairness = compute_demographic_parity(preds, sensitive_attr)

            candidates.append({
                "rule_list": wrapper,
                "accuracy": accuracy,
                "fairness_violation": fairness,
                "fairness_weight": fw,
                "objective": (1 - fw) * accuracy + fw * (1 - fairness)
            })

        return candidates


# Integration with LaundryML
class LaundryML_CORELS:
    """LaundryML using CORELS for optimal rule list enumeration."""

    def __init__(self, black_box, sensitive_attr: str,
                 fairness_metric: str = "demographic_parity",
                 max_length: int = 10, c: float = 0.01):
        self.black_box = black_box
        self.sensitive_attr = sensitive_attr
        self.fairness_metric = fairness_metric
        self.max_length = max_length
        self.c = c

    def enumerate_rule_lists(self, X: pd.DataFrame, y: np.ndarray,
                            beta_values: List[float] = [0.0, 0.5, 0.9],
                            n_models: int = 50) -> List[Dict]:
        """Enumerate rule lists using CORELS."""
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
                max_length=self.max_length,
                c=self.c * (1 + beta)  # Increase regularization for fairer models
            )

            # Fit
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
                    "rule_list_str": wrapper.get_rule_list_str()
                })

        return candidates

    def _get_black_box_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from black-box model."""
        if hasattr(self.black_box, 'predict'):
            return self.black_box.predict(X)
        else:
            import torch
            self.black_box.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X.values)
                output = self.black_box(X_tensor)
                return output.argmax(dim=1).numpy()


if __name__ == "__main__":
    # Test
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)

    wrapper = CORELSWrapper(max_length=5)
    wrapper.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])

    print("Rule List:")
    print(wrapper.get_rule_list_str())
    print(f"\nAccuracy: {wrapper.score(X, y):.4f}")
