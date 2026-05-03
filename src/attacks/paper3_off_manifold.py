"""
Paper 3: Fairwashing Explanations with Off-Manifold Detergent
(Anders et al., ICML 2020)

This module implements:
1. Model manipulation to produce arbitrary target explanations
2. Tangent Space Projection (TSP) for robust explanations
3. Linear and non-linear tangent space projectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class TangentSpaceProjector:
    """
    Computes tangent space projectors for data manifolds.
    Uses local PCA on k-nearest neighbors.
    """

    def __init__(self, n_neighbors: int = 32, n_components: int = 16):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.projectors = {}  # class -> projector matrix
        self.knn_models = {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit tangent space projectors.

        Args:
            X: Data points (n_samples, n_features)
            y: Class labels (optional, for class-specific projectors)
        """
        if y is not None:
            # Class-specific projectors
            for cls in np.unique(y):
                X_cls = X[y == cls]
                self._fit_class_projector(X_cls, cls)
        else:
            # Single projector for all data
            self._fit_class_projector(X, "all")

    def _fit_class_projector(self, X_cls: np.ndarray, cls):
        """Fit projector for a single class."""
        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_cls) - 1))
        knn.fit(X_cls)
        self.knn_models[cls] = knn

        # Compute local tangent spaces
        # For each point, find neighbors and compute PCA
        tangent_bases = []
        for i in range(min(1000, len(X_cls))):  # Subsample for efficiency
            point = X_cls[i:i+1]
            distances, indices = knn.kneighbors(point)
            neighbors = X_cls[indices[0]]

            # Center neighbors
            centered = neighbors - neighbors.mean(axis=0)

            # PCA
            if centered.shape[0] > 1 and centered.shape[1] > 1:
                pca = PCA(n_components=min(self.n_components, centered.shape[0] - 1))
                pca.fit(centered)
                tangent_bases.append(pca.components_)

        # Average tangent space (simplified)
        if tangent_bases:
            avg_basis = np.mean(tangent_bases, axis=0)
            # Projector: P = V^T V where V is tangent basis
            self.projectors[cls] = avg_basis.T @ avg_basis
        else:
            self.projectors[cls] = np.eye(X_cls.shape[1])

    def project(self, x: np.ndarray, cls=None) -> np.ndarray:
        """Project vector onto tangent space."""
        key = cls if cls is not None else "all"
        if key not in self.projectors:
            key = "all"

        P = self.projectors[key]
        return x @ P

    def project_orthogonal(self, x: np.ndarray, cls=None) -> np.ndarray:
        """Project vector onto orthogonal complement of tangent space."""
        key = cls if cls is not None else "all"
        if key not in self.projectors:
            key = "all"

        P = self.projectors[key]
        return x - x @ P

    def save(self, path: str):
        """Save projectors to HDF5 file."""
        if H5PY_AVAILABLE:
            with h5py.File(path, 'w') as f:
                for cls, proj in self.projectors.items():
                    f.create_dataset(str(cls), data=proj)
        else:
            import numpy as np
            np.savez(path, **{str(k): v for k, v in self.projectors.items()})

    def load(self, path: str):
        """Load projectors from HDF5 file."""
        self.projectors = {}
        if H5PY_AVAILABLE:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    self.projectors[key] = f[key][:]
        else:
            import numpy as np
            data = np.load(path + '.npz')
            for key in data.files:
                self.projectors[key] = data[key]


class OffManifoldFairwasher:
    """
    Implements off-manifold fairwashing attack.

    Trains a student model to match teacher outputs on data manifold
    while producing arbitrary target explanations.
    """

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 explanation_method: str = "gradient",
                 alpha: float = 0.8, device: str = "cpu"):
        """
        Args:
            teacher_model: Original (biased) model
            student_model: Model to train (manipulated)
            explanation_method: Which explanation to manipulate
            alpha: Weight for explanation loss vs output loss
            device: torch device
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.explanation_method = explanation_method
        self.alpha = alpha
        self.device = device

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_explanation(self, model: nn.Module, x: torch.Tensor,
                           target_class: int) -> torch.Tensor:
        """Compute explanation for a model."""
        x = x.clone().requires_grad_(True)
        output = model(x)
        score = output[:, target_class].sum()

        model.zero_grad()
        grad = torch.autograd.grad(score, x, create_graph=True)[0]

        if self.explanation_method == "xgrad":
            return x.detach() * grad
        elif self.explanation_method == "gradient":
            return grad
        else:
            return grad

    def train_step(self, x: torch.Tensor, target_explanation: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> dict:
        """Single training step."""
        x = x.to(self.device)
        target_explanation = target_explanation.to(self.device)

        # Get teacher predictions
        with torch.no_grad():
            teacher_output = self.teacher(x)
            target_class = teacher_output.argmax(dim=1)

        # Student forward
        student_output = self.student(x)

        # Output loss (match teacher)
        if student_output.shape[1] > 1:
            output_loss = F.mse_loss(student_output, teacher_output)
        else:
            output_loss = F.mse_loss(student_output.squeeze(), teacher_output.squeeze())

        # Explanation loss (match target)
        student_exp = self.compute_explanation(self.student, x, target_class[0])

        # Normalize explanations
        student_exp = student_exp / (student_exp.abs().max() + 1e-8)
        target_exp = target_explanation / (target_explanation.abs().max() + 1e-8)

        exp_loss = F.mse_loss(student_exp, target_exp)

        # Combined loss
        loss = (1 - self.alpha) * output_loss + self.alpha * exp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "total_loss": loss.item(),
            "output_loss": output_loss.item(),
            "explanation_loss": exp_loss.item(),
        }

    def train(self, train_loader, target_explanation: torch.Tensor,
              n_epochs: int = 100, lr: float = 5e-5,
              projector: Optional[TangentSpaceProjector] = None) -> list:
        """
        Train the fairwashed model.

        Args:
            train_loader: Data loader
            target_explanation: Target explanation heatmap
            n_epochs: Number of epochs
            lr: Learning rate
            projector: Optional TSP projector for defense

        Returns:
            Training history
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        history = []

        for epoch in range(n_epochs):
            epoch_losses = []

            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch["features"]
                else:
                    x = batch[0]

                # Expand target to batch size
                target_batch = target_explanation.unsqueeze(0).expand(x.size(0), -1, -1, -1)

                losses = self.train_step(x, target_batch, optimizer)
                epoch_losses.append(losses)

            # Average losses
            avg_losses = {
                k: np.mean([d[k] for d in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            history.append(avg_losses)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: {avg_losses}")

        return history

    def evaluate(self, test_loader, target_explanation: torch.Tensor) -> dict:
        """Evaluate fairwashing quality."""
        self.student.eval()

        output_mses = []
        exp_mses = []
        ssims = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    x = batch["features"]
                else:
                    x = batch[0]

                x = x.to(self.device)
                teacher_output = self.teacher(x)
                student_output = self.student(x)

                output_mses.append(F.mse_loss(student_output, teacher_output).item())

                # Explanation similarity
                target_class = teacher_output.argmax(dim=1)[0]
                x_grad = x.clone().requires_grad_(True)
                student_exp = self.compute_explanation(self.student, x_grad, target_class)

                exp_mses.append(F.mse_loss(student_exp, target_explanation.to(self.device)).item())

        return {
            "output_mse": np.mean(output_mses),
            "explanation_mse": np.mean(exp_mses),
        }


class TSPDefender:
    """
    Tangent Space Projection defense against off-manifold fairwashing.
    """

    def __init__(self, projector: TangentSpaceProjector):
        self.projector = projector

    def compute_tsp_explanation(self, model: nn.Module, x: torch.Tensor,
                                target_class: int) -> torch.Tensor:
        """
        Compute TSP-projected explanation.

        Projects explanation onto tangent space of data manifold,
        making it robust to off-manifold manipulation.
        """
        x = x.clone().requires_grad_(True)
        output = model(x)
        score = output[:, target_class].sum()

        model.zero_grad()
        grad = torch.autograd.grad(score, x)[0]

        # Project gradient onto tangent space
        grad_np = grad.detach().cpu().numpy()

        # Flatten for projection
        original_shape = grad_np.shape
        grad_flat = grad_np.reshape(grad_np.shape[0], -1)

        projected_flat = self.projector.project(grad_flat[0])
        projected = projected_flat.reshape(original_shape)

        return torch.FloatTensor(projected).to(x.device)


def create_target_explanation(shape: Tuple, pattern: str = "random") -> torch.Tensor:
    """
    Create a target explanation heatmap.

    Args:
        shape: Shape of explanation (C, H, W) or (H, W)
        pattern: "random", "center", "checkerboard", or "number_42"

    Returns:
        Target explanation tensor
    """
    if pattern == "random":
        return torch.randn(shape)
    elif pattern == "center":
        exp = torch.zeros(shape)
        if len(shape) == 3:
            _, h, w = shape
            exp[:, h//4:3*h//4, w//4:3*w//4] = 1.0
        else:
            h, w = shape
            exp[h//4:3*h//4, w//4:3*w//4] = 1.0
        return exp
    elif pattern == "number_42":
        # Create a heatmap shaped like "42"
        exp = torch.zeros(shape)
        if len(shape) == 3:
            _, h, w = shape
            # Simple pattern: two bright regions
            exp[:, h//4:h//2, w//4:w//2] = 1.0
            exp[:, h//4:h//2, 3*w//4:w-2] = 0.8
        return exp
    else:
        return torch.randn(shape)
