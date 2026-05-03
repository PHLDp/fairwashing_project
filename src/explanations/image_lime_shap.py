"""
Full image support for LIME and SHAP explanations.
Implements segmentation-based explanations for images.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Callable, List, Tuple, Dict
from skimage.segmentation import slic, quickshift, felzenszwalb
from skimage.util import img_as_float
from skimage.color import gray2rgb
import warnings


class ImageSegmenter:
    """
    Segmentation algorithms for image explanations.
    Wraps skimage segmentation methods.
    """

    def __init__(self, method: str = "slic", n_segments: int = 50,
                 compactness: float = 10.0, sigma: float = 1.0):
        """
        Args:
            method: "slic", "quickshift", or "felzenszwalb"
            n_segments: Target number of segments
            compactness: SLIC compactness parameter
            sigma: Gaussian smoothing parameter
        """
        self.method = method
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image into superpixels.

        Args:
            image: Image array (H, W) or (H, W, C)

        Returns:
            Segmentation mask (H, W) with segment labels
        """
        # Ensure image is float and has channels
        if image.ndim == 2:
            image = gray2rgb(image)

        image = img_as_float(image)

        if self.method == "slic":
            segments = slic(image, n_segments=self.n_segments,
                          compactness=self.compactness,
                          sigma=self.sigma, start_label=0)
        elif self.method == "quickshift":
            segments = quickshift(image, kernel_size=3, max_dist=6,
                                 ratio=0.5, sigma=self.sigma)
        elif self.method == "felzenszwalb":
            segments = felzenszwalb(image, scale=100, sigma=self.sigma,
                                   min_size=50)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

        return segments

    def get_segment_mask(self, segments: np.ndarray, segment_id: int) -> np.ndarray:
        """Get binary mask for a specific segment."""
        return (segments == segment_id).astype(float)

    def count_segments(self, segments: np.ndarray) -> int:
        """Count number of segments."""
        return len(np.unique(segments))


class ImageLIMEExplainer:
    """
    LIME explanations for images with segmentation.
    """

    def __init__(self, model: torch.nn.Module, segmenter: Optional[ImageSegmenter] = None):
        """
        Args:
            model: PyTorch model
            segmenter: Image segmenter (default: SLIC with 50 segments)
        """
        self.model = model
        self.model.eval()
        self.segmenter = segmenter or ImageSegmenter(method="slic", n_segments=50)

        try:
            from lime.lime_image import LimeImageExplainer
            self.lime_available = True
        except ImportError:
            self.lime_available = False
            warnings.warn("LIME not installed. Using custom implementation.")

    def explain(self, image: np.ndarray, target_class: Optional[int] = None,
                n_samples: int = 1000, top_labels: int = 1,
                hide_color: Optional[float] = None) -> Dict:
        """
        Generate LIME explanation for image.

        Args:
            image: Input image (H, W) or (H, W, C)
            target_class: Target class for explanation
            n_samples: Number of perturbed samples
            top_labels: Number of top labels to explain
            hide_color: Color to use for hidden segments (None = mean)

        Returns:
            Dictionary with explanation results
        """
        if self.lime_available:
            return self._explain_lime_package(image, target_class, n_samples, top_labels)
        else:
            return self._explain_custom(image, target_class, n_samples, hide_color)

    def _explain_lime_package(self, image: np.ndarray, target_class: Optional[int],
                             n_samples: int, top_labels: int) -> Dict:
        """Use official LIME package."""
        from lime.lime_image import LimeImageExplainer

        explainer = LimeImageExplainer()

        def predict_fn(images):
            # LIME passes numpy images, need to convert to torch
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
            if images_tensor.shape[1] == 1 and len(image.shape) == 2:
                # Grayscale image
                images_tensor = images_tensor[:, :1, :, :]

            with torch.no_grad():
                output = self.model(images_tensor)
                probs = F.softmax(output, dim=1)
            return probs.numpy()

        # Convert grayscale to RGB if needed
        if image.ndim == 2:
            image_rgb = gray2rgb(image)
        else:
            image_rgb = image

        explanation = explainer.explain_instance(
            image_rgb, predict_fn, top_labels=top_labels,
            hide_color=hide_color, num_samples=n_samples
        )

        return {
            "explanation": explanation,
            "segments": explanation.segments,
            "local_exp": explanation.local_exp,
        }

    def _explain_custom(self, image: np.ndarray, target_class: Optional[int],
                       n_samples: int, hide_color: Optional[float]) -> Dict:
        """Custom LIME implementation without package."""
        # Segment image
        segments = self.segmenter.segment(image)
        n_segments = self.segmenter.count_segments(segments)

        # Get original prediction
        image_tensor = self._preprocess_image(image)
        with torch.no_grad():
            output = self.model(image_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            orig_prob = F.softmax(output, dim=1)[0, target_class].item()

        # Generate perturbed samples
        perturbed_probs = []

        for _ in range(n_samples):
            # Randomly hide segments
            mask = np.random.rand(n_segments) > 0.5
            perturbed = self._perturb_image(image, segments, mask, hide_color)

            perturbed_tensor = self._preprocess_image(perturbed)
            with torch.no_grad():
                output = self.model(perturbed_tensor)
                prob = F.softmax(output, dim=1)[0, target_class].item()

            perturbed_probs.append(prob)

        # Compute segment importance (simplified)
        # In full implementation, would fit linear model
        segment_importance = self._compute_segment_importance(
            image, segments, n_segments, target_class, n_samples, hide_color
        )

        return {
            "target_class": target_class,
            "original_probability": orig_prob,
            "segment_importance": segment_importance,
            "segments": segments,
            "n_segments": n_segments,
        }

    def _perturb_image(self, image: np.ndarray, segments: np.ndarray,
                      mask: np.ndarray, hide_color: Optional[float]) -> np.ndarray:
        """Perturb image by hiding segments."""
        perturbed = image.copy()

        if hide_color is None:
            hide_color = image.mean()

        for i in range(len(mask)):
            if not mask[i]:
                perturbed[segments == i] = hide_color

        return perturbed

    def _compute_segment_importance(self, image: np.ndarray, segments: np.ndarray,
                                   n_segments: int, target_class: int,
                                   n_samples: int, hide_color: Optional[float]) -> Dict:
        """Compute importance of each segment."""
        importances = {}

        for seg_id in range(n_segments):
            # Hide only this segment
            mask = np.ones(n_segments, dtype=bool)
            mask[seg_id] = False

            perturbed = self._perturb_image(image, segments, mask, hide_color)
            perturbed_tensor = self._preprocess_image(perturbed)

            with torch.no_grad():
                output = self.model(perturbed_tensor)
                prob = F.softmax(output, dim=1)[0, target_class].item()

            # Importance = drop in probability when segment is hidden
            importances[seg_id] = prob

        return importances

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        if image.ndim == 2:
            # Grayscale
            tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        else:
            # RGB
            tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)

        return tensor

    def get_explanation_mask(self, explanation: Dict, 
                            positive_only: bool = True) -> np.ndarray:
        """
        Generate explanation heatmap from LIME explanation.

        Args:
            explanation: Output from explain()
            positive_only: Only show positive contributions

        Returns:
            Heatmap array (H, W)
        """
        segments = explanation["segments"]
        importance = explanation["segment_importance"]

        # Create heatmap
        heatmap = np.zeros_like(segments, dtype=float)

        for seg_id, imp in importance.items():
            if positive_only and imp < 0:
                continue
            heatmap[segments == seg_id] = abs(imp)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


class ImageSHAPExplainer:
    """
    SHAP explanations for images with segmentation.
    """

    def __init__(self, model: torch.nn.Module, segmenter: Optional[ImageSegmenter] = None):
        """
        Args:
            model: PyTorch model
            segmenter: Image segmenter
        """
        self.model = model
        self.model.eval()
        self.segmenter = segmenter or ImageSegmenter(method="slic", n_segments=50)

        try:
            import shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            warnings.warn("SHAP not installed. Using custom implementation.")

    def explain(self, image: np.ndarray, target_class: Optional[int] = None,
               n_samples: int = 50, background_images: Optional[np.ndarray] = None) -> Dict:
        """
        Generate SHAP explanation for image.

        Args:
            image: Input image
            target_class: Target class
            n_samples: Number of background samples
            background_images: Background images for SHAP

        Returns:
            Dictionary with SHAP values
        """
        # Segment image
        segments = self.segmenter.segment(image)
        n_segments = self.segmenter.count_segments(segments)

        # Get original prediction
        image_tensor = self._preprocess_image(image)
        with torch.no_grad():
            output = self.model(image_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        if self.shap_available and background_images is not None:
            return self._explain_shap_package(image, segments, n_segments,
                                            target_class, background_images)
        else:
            return self._explain_custom(image, segments, n_segments,
                                       target_class, n_samples)

    def _explain_shap_package(self, image: np.ndarray, segments: np.ndarray,
                           n_segments: int, target_class: int,
                           background_images: np.ndarray) -> Dict:
        """Use SHAP package with segmentation."""
        import shap

        # Create masker for segments
        def masker(mask, x):
            """Mask image segments."""
            masked = x.copy()
            for i in range(len(mask)):
                if not mask[i]:
                    masked[segments == i] = x[segments == i].mean()
            return masked

        # Create explainer
        def f(mask):
            masked_img = masker(mask, image)
            tensor = self._preprocess_image(masked_img)
            with torch.no_grad():
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)
            return probs[0, target_class].item()

        # Compute SHAP values (simplified - full implementation needs KernelExplainer)
        explainer = shap.KernelExplainer(f, np.zeros((1, n_segments)))
        shap_values = explainer.shap_values(np.ones((1, n_segments)), nsamples=n_samples)

        return {
            "shap_values": shap_values,
            "segments": segments,
            "target_class": target_class,
        }

    def _explain_custom(self, image: np.ndarray, segments: np.ndarray,
                       n_segments: int, target_class: int,
                       n_samples: int) -> Dict:
        """Custom SHAP-like implementation."""
        # Compute baseline (all segments hidden)
        baseline = np.zeros_like(image) + image.mean()
        baseline_tensor = self._preprocess_image(baseline)

        with torch.no_grad():
            baseline_output = self.model(baseline_tensor)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()

        # Compute marginal contributions
        shap_values = np.zeros(n_segments)

        for seg_id in range(n_segments):
            # Image with only this segment
            mask = np.zeros(n_segments, dtype=bool)
            mask[seg_id] = True

            perturbed = self._perturb_image(image, segments, mask, image.mean())
            perturbed_tensor = self._preprocess_image(perturbed)

            with torch.no_grad():
                output = self.model(perturbed_tensor)
                prob = F.softmax(output, dim=1)[0, target_class].item()

            # SHAP value = marginal contribution
            shap_values[seg_id] = prob - baseline_prob

        return {
            "shap_values": shap_values,
            "segments": segments,
            "target_class": target_class,
            "baseline_probability": baseline_prob,
        }

    def _perturb_image(self, image: np.ndarray, segments: np.ndarray,
                      mask: np.ndarray, fill_value: float) -> np.ndarray:
        """Perturb image based on segment mask."""
        perturbed = np.ones_like(image) * fill_value

        for i in range(len(mask)):
            if mask[i]:
                perturbed[segments == i] = image[segments == i]

        return perturbed

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        if image.ndim == 2:
            tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def get_explanation_mask(self, explanation: Dict) -> np.ndarray:
        """Generate SHAP heatmap."""
        segments = explanation["segments"]
        shap_values = explanation["shap_values"]

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).flatten()

        heatmap = np.zeros_like(segments, dtype=float)

        for seg_id in range(len(shap_values)):
            heatmap[segments == seg_id] = abs(shap_values[seg_id])

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


class ImageExplanationComparator:
    """Compare LIME and SHAP explanations for the same image."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.lime_explainer = ImageLIMEExplainer(model)
        self.shap_explainer = ImageSHAPExplainer(model)

    def compare(self, image: np.ndarray, target_class: Optional[int] = None) -> Dict:
        """
        Compare LIME and SHAP explanations.

        Returns:
            Dictionary with both explanations and comparison metrics
        """
        # Get LIME explanation
        lime_exp = self.lime_explainer.explain(image, target_class)
        lime_mask = self.lime_explainer.get_explanation_mask(lime_exp)

        # Get SHAP explanation
        shap_exp = self.shap_explainer.explain(image, target_class)
        shap_mask = self.shap_explainer.get_explanation_mask(shap_exp)

        # Compute similarity
        from skimage.metrics import structural_similarity as ssim

        similarity = ssim(lime_mask, shap_mask, data_range=1.0)

        # Correlation
        corr = np.corrcoef(lime_mask.flatten(), shap_mask.flatten())[0, 1]

        return {
            "lime_explanation": lime_exp,
            "shap_explanation": shap_exp,
            "lime_mask": lime_mask,
            "shap_mask": shap_mask,
            "ssim_similarity": similarity,
            "correlation": corr,
        }


# Integration with scaffolding attack for images
class ImageScaffoldingAttack:
    """
    Extend scaffolding attack to images.
    Adds decoy segments that fool LIME/SHAP.
    """

    def __init__(self, model: torch.nn.Module, n_decoy_segments: int = 5):
        self.model = model
        self.n_decoy_segments = n_decoy_segments
        self.segmenter = ImageSegmenter(method="slic", n_segments=50 + n_decoy_segments)

    def add_decoy_segments(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add decoy segments to image.

        Returns:
            (modified_image, segment_mask)
        """
        segments = self.segmenter.segment(image)

        # Add high-variance decoy segments at random locations
        decoy_segments = []
        for i in range(self.n_decoy_segments):
            seg_id = 50 + i  # IDs after original segments
            # Create random pattern
            mask = np.random.rand(*image.shape[:2]) > 0.5
            segments[mask] = seg_id
            decoy_segments.append(seg_id)

        return image, segments

    def fool_lime_image(self, image: np.ndarray, target_class: Optional[int] = None) -> Dict:
        """Demonstrate LIME fooling on image."""
        image_aug, segments = self.add_decoy_segments(image)

        explainer = ImageLIMEExplainer(self.model, 
                                       ImageSegmenter(method="slic", 
                                                    n_segments=50 + self.n_decoy_segments))

        explanation = explainer.explain(image_aug, target_class)

        # Check if decoy segments got high importance
        decoy_importance = []
        for seg_id in range(50, 50 + self.n_decoy_segments):
            if seg_id in explanation["segment_importance"]:
                decoy_importance.append(explanation["segment_importance"][seg_id])

        return {
            "explanation": explanation,
            "decoy_importance": decoy_importance,
            "fooling_success": len([d for d in decoy_importance if d > 0.5]) > 0
        }


if __name__ == "__main__":
    # Test with simple model
    from src.models.architectures import Conv4

    model = Conv4(num_classes=10)
    model.eval()

    # Create test image
    image = np.random.rand(28, 28)

    # LIME explanation
    lime = ImageLIMEExplainer(model)
    lime_exp = lime.explain(image, n_samples=100)
    print(f"LIME segments: {lime_exp['n_segments']}")

    # SHAP explanation
    shap = ImageSHAPExplainer(model)
    shap_exp = shap.explain(image, n_samples=50)
    print(f"SHAP segments: {len(shap_exp['shap_values'])}")
