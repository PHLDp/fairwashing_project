"""
Explanation methods for neural networks.
Implements: Gradients, x*Grad, Integrated Gradients, LRP, LIME, SHAP.

Now includes:
- Full LRP with z-rule, z+-rule, epsilon-rule (lrp_full.py)
- Image LIME/SHAP with segmentation (image_lime_shap.py)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Dict
from abc import ABC, abstractmethod

# Import full LRP implementation
try:
    from .lrp_full import FullLRPExplainer, LRPAnalyzer
    FULL_LRP_AVAILABLE = True
except ImportError:
    FULL_LRP_AVAILABLE = False

# Import image explainers
try:
    from .image_lime_shap import ImageLIMEExplainer, ImageSHAPExplainer, ImageSegmenter
    IMAGE_EXPLAINERS_AVAILABLE = True
except ImportError:
    IMAGE_EXPLAINERS_AVAILABLE = False


class ExplanationMethod(ABC):
    """Base class for explanation methods."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    @abstractmethod
    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate explanation for input x."""
        pass

    def _get_target_class(self, x: torch.Tensor, target_class: Optional[int] = None) -> int:
        """Determine target class for explanation."""
        if target_class is None:
            with torch.no_grad():
                output = self.model(x)
                target_class = output.argmax(dim=1).item()
        return target_class


class GradientExplainer(ExplanationMethod):
    """Simple gradient-based explanation (Simonyan et al.)."""

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        target_class = self._get_target_class(x, target_class)

        x.requires_grad = True
        output = self.model(x)
        score = output[0, target_class]

        self.model.zero_grad()
        score.backward()

        grad = x.grad.data[0].detach().cpu().numpy()
        x.requires_grad = False

        return grad


class XGradExplainer(ExplanationMethod):
    """Gradient * Input explanation (Shrikumar et al.)."""

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        target_class = self._get_target_class(x, target_class)

        x.requires_grad = True
        output = self.model(x)
        score = output[0, target_class]

        self.model.zero_grad()
        score.backward()

        grad = x.grad.data[0].detach().cpu().numpy()
        x_input = x[0].detach().cpu().numpy()

        x.requires_grad = False
        return x_input * grad


class IntegratedGradientsExplainer(ExplanationMethod):
    """Integrated Gradients (Sundararajan et al.)."""

    def __init__(self, model: torch.nn.Module, n_steps: int = 50):
        super().__init__(model)
        self.n_steps = n_steps

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        target_class = self._get_target_class(x, target_class)

        if baseline is None:
            baseline = torch.zeros_like(x)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps).view(-1, 1, 1, 1)
        if x.dim() == 4:  # Image
            alphas = alphas.view(-1, 1, 1, 1)
        else:
            alphas = alphas.view(-1, 1)

        interpolated = baseline + alphas.to(x.device) * (x - baseline)
        interpolated.requires_grad = True

        # Compute gradients
        outputs = self.model(interpolated)
        scores = outputs[:, target_class]

        self.model.zero_grad()
        gradients = torch.autograd.grad(scores.sum(), interpolated)[0]

        # Average gradients
        avg_gradients = gradients.mean(dim=0)

        # Multiply by input difference
        ig = (x[0] - baseline[0]).detach().cpu().numpy() * avg_gradients.detach().cpu().numpy()

        interpolated.requires_grad = False
        return ig


class LRPExplainer(ExplanationMethod):
    """
    Layer-wise Relevance Propagation (Bach et al.).
    Uses full implementation from lrp_full.py if available,
    otherwise falls back to simplified gradient*input proxy.
    """

    def __init__(self, model: torch.nn.Module, rule: str = "z+",
                 epsilon: float = 1e-6, use_full: bool = True):
        super().__init__(model)
        self.rule = rule
        self.epsilon = epsilon
        self.use_full = use_full and FULL_LRP_AVAILABLE

        if self.use_full:
            self.full_explainer = FullLRPExplainer(model, rule=rule, epsilon=epsilon)

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        if self.use_full:
            return self.full_explainer.explain(x, target_class)
        else:
            return self._explain_fallback(x, target_class)

    def _explain_fallback(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Fallback simplified LRP using gradient*input."""
        target_class = self._get_target_class(x, target_class)

        x.requires_grad = True
        output = self.model(x)
        score = output[0, target_class]

        self.model.zero_grad()
        score.backward()

        grad = x.grad.data[0].detach().cpu().numpy()
        x_input = x[0].detach().cpu().numpy()

        x.requires_grad = False
        return x_input * grad


class LIMEExplainerWrapper(ExplanationMethod):
    """Wrapper for LIME explanations (tabular and image)."""

    def __init__(self, model: torch.nn.Module, mode: str = "tabular",
                 segmenter: Optional['ImageSegmenter'] = None):
        super().__init__(model)
        self.mode = mode
        self.segmenter = segmenter

        if mode == "image" and IMAGE_EXPLAINERS_AVAILABLE:
            self.image_explainer = ImageLIMEExplainer(model, segmenter)
        else:
            self.image_explainer = None

        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer
            from lime.lime_image import LimeImageExplainer
            self.lime_available = True
        except ImportError:
            self.lime_available = False
            print("Warning: LIME not installed. Install with: pip install lime")

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                training_data: Optional[np.ndarray] = None,
                feature_names: Optional[list] = None,
                image: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mode == "image" and self.image_explainer is not None:
            # Image mode
            if image is None:
                # Convert tensor to numpy image
                image = x[0].detach().cpu().numpy()
                if image.shape[0] == 1:  # Grayscale
                    image = image[0]
                elif image.shape[0] == 3:  # RGB
                    image = np.transpose(image, (1, 2, 0))

            result = self.image_explainer.explain(image, target_class)
            return result.get("lime_mask", np.zeros_like(image))

        elif self.mode == "tabular":
            # Tabular mode
            if training_data is None:
                raise ValueError("training_data required for tabular LIME")

            from lime.lime_tabular import LimeTabularExplainer

            explainer = LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=["class_0", "class_1"],
                discretize_continuous=True
            )

            def predict_fn(X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    output = self.model(X_tensor)
                    probs = F.softmax(output, dim=1)
                    return probs.numpy()

            exp = explainer.explain_instance(
                x[0].cpu().numpy(),
                predict_fn,
                num_features=len(feature_names) if feature_names else x.shape[1],
                top_labels=1
            )

            # Convert to array
            explanation = np.zeros(x.shape[1])
            for idx, val in exp.as_list(label=exp.available_labels()[0]):
                # Parse feature index
                if isinstance(idx, str):
                    import re
                    match = re.search(r'(\d+)', idx)
                    if match:
                        idx = int(match.group(1))
                explanation[idx] = val

            return explanation
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class SHAPExplainerWrapper(ExplanationMethod):
    """Wrapper for SHAP explanations (tabular and image)."""

    def __init__(self, model: torch.nn.Module, mode: str = "tabular",
                 segmenter: Optional['ImageSegmenter'] = None):
        super().__init__(model)
        self.mode = mode
        self.segmenter = segmenter

        if mode == "image" and IMAGE_EXPLAINERS_AVAILABLE:
            self.image_explainer = ImageSHAPExplainer(model, segmenter)
        else:
            self.image_explainer = None

        try:
            import shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            print("Warning: SHAP not installed. Install with: pip install shap")

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                background_data: Optional[torch.Tensor] = None,
                image: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mode == "image" and self.image_explainer is not None:
            # Image mode
            if image is None:
                image = x[0].detach().cpu().numpy()
                if image.shape[0] == 1:
                    image = image[0]
                elif image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))

            result = self.image_explainer.explain(image, target_class)
            return result.get("shap_values", np.zeros_like(image))

        elif self.mode == "tabular":
            if background_data is None:
                raise ValueError("background_data required for SHAP")

            import shap

            def predict_fn(X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    output = self.model(X_tensor)
                    probs = F.softmax(output, dim=1)
                    return probs.numpy()

            explainer = shap.KernelExplainer(predict_fn, background_data.numpy())
            shap_values = explainer.shap_values(x.numpy(), nsamples=100)

            target_class = self._get_target_class(x, target_class)
            return np.array(shap_values[target_class])[0]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class DeepLIFTExplainer(ExplanationMethod):
    """
    DeepLIFT explanation (Shrikumar et al.).
    Simplified implementation using gradient approximation.
    """

    def __init__(self, model: torch.nn.Module, reference: Optional[torch.Tensor] = None):
        super().__init__(model)
        self.reference = reference

    def explain(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        target_class = self._get_target_class(x, target_class)

        if self.reference is None:
            self.reference = torch.zeros_like(x)

        # Approximate DeepLIFT using integrated gradients with fewer steps
        ig = IntegratedGradientsExplainer(self.model, n_steps=10)
        return ig.explain(x, target_class, baseline=self.reference)


def get_explainer(method: str, model: torch.nn.Module, **kwargs) -> ExplanationMethod:
    """Factory function to get explanation methods."""
    explainers = {
        "gradient": GradientExplainer,
        "xgrad": XGradExplainer,
        "integrated_gradients": IntegratedGradientsExplainer,
        "lrp": LRPExplainer,
        "lime": LIMEExplainerWrapper,
        "shap": SHAPExplainerWrapper,
        "deeplift": DeepLIFTExplainer,
    }

    if method not in explainers:
        raise ValueError(f"Unknown explanation method: {method}")

    return explainers[method](model, **kwargs)
