"""
Calibration metrics for evaluating model uncertainty.

This module provides a comprehensive collection of binning-based calibration
metrics for classification models, centered around the General Calibration
Error (GCE) framework.
"""

import numpy as np
from scipy.special import softmax
from typing import Union, Literal, Tuple


def general_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    class_conditional: bool = False,
    adaptive_bins: bool = False,
    top_k_classes: Union[int, Literal['all']] = 1,
    norm: Union[int, Literal['inf']] = 1,
    thresholding: float = 0.0,
    logits: bool = False,
    direction: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Calculate General Calibration Error (GCE).
    
    The GCE is a flexible calibration metric that can be configured to produce
    many popular calibration metrics including ECE, MCE, RMSCE, ACE, and SCE.
    
    The class-conditional GCE with L^p norm is defined as:
    GCE = (Σ_k Σ_b (n_bk / NK) |acc(b,k) - conf(b,k)|^p)^(1/p)
    
    Where acc(b,k) and conf(b,k) are the accuracy and confidence of bin b for
    class label k; n_bk is the number of predictions in bin b for class k;
    N is the total number of data points; and K is the number of classes.
    
    When direction=True, overconfident and underconfident bins are aggregated
    separately (not netted). Returns (over, under), the weighted L1 means of
    max(conf - acc, 0) and max(acc - conf, 0) respectively.
    
    References:
        Kull et al. (2019). "Beyond temperature scaling: Obtaining well-calibrated
        multiclass probabilities with Dirichlet calibration." NeurIPS.
        
        Nixon et al. (2020). "Measuring Calibration in Deep Learning."
        CVPR Workshops.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        class_conditional: If True, compute class-conditional calibration.
            Default: False.
        adaptive_bins: If True, use adaptive binning based on data distribution.
            Default: False (uniform bins).
        top_k_classes: Number of top predicted classes to consider. Use 'all'
            to consider all classes. Default: 1 (top prediction only).
        norm: L^p norm to use. Can be 1, 2, or 'inf'. Default: 1.
        thresholding: Ignore probabilities below this threshold. Default: 0.0.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        direction: If True, return separate over- and under-confidence L1
            means as (over, under). Default: False.
    
    Returns:
        float: GCE value when direction=False (non-negative).
        tuple[float, float]: (over, under) when direction=True, both
            non-negative. over + under equals the non-directional L1 error
            for the same configuration.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> gce = general_calibration_error(probs, labels)
        >>> print(f"GCE: {gce:.4f}")
    """
    # Convert logits to probabilities if needed
    if logits:
        probabilities = softmax(probabilities, axis=1)
    
    # Validate inputs
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_classes)")
    
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array of shape (n_samples,)")
    
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Number of samples in probabilities and labels must match")
    
    n_samples = probabilities.shape[0]
    n_classes = probabilities.shape[1]
    
    # Apply thresholding
    if thresholding > 0:
        probabilities = probabilities.copy()
        probabilities[probabilities < thresholding] = 0.0
    
    # Get predictions and confidences
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    if not class_conditional:
        # Standard calibration (top-1 prediction only)
        return _compute_calibration_error(
            confidences, accuracies, n_bins, adaptive_bins, norm, direction
        )

    # Class-conditional calibration
    class_errors = []
    if top_k_classes == 'all':
        for k in range(n_classes):
            class_probs = probabilities[:, k]
            class_correct = (labels == k).astype(float)

            if np.sum(class_probs > 0) == 0:  # Skip if no predictions for this class
                continue

            error = _compute_calibration_error(
                class_probs, class_correct, n_bins, adaptive_bins, norm,
                direction
            )
            class_errors.append(error)
    else:
        # Compute for top-k classes
        top_k = min(top_k_classes, n_classes)
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:]

        for k in range(top_k):
            # Get k-th highest probability for each sample
            k_idx = top_k_indices[:, -(k+1)]
            k_probs = probabilities[np.arange(n_samples), k_idx]
            k_correct = (labels == k_idx).astype(float)

            error = _compute_calibration_error(
                k_probs, k_correct, n_bins, adaptive_bins, norm, direction
            )
            class_errors.append(error)

    if not class_errors:
        return (0.0, 0.0) if direction else 0.0

    if direction:
        overs, unders = zip(*class_errors)
        return float(np.mean(overs)), float(np.mean(unders))

    return float(np.mean(class_errors))


def _compute_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int,
    adaptive_bins: bool,
    norm: Union[int, Literal['inf']],
    direction: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Compute calibration error for a single set of confidences and accuracies.
    
    Args:
        confidences: Array of confidence values.
        accuracies: Array of binary accuracy values.
        n_bins: Number of bins.
        adaptive_bins: Whether to use adaptive binning.
        norm: L^p norm to use.
        direction: If True, return separate over- and under-confidence L1
            means as (over, under) instead of a single absolute error.
            Default: False.
    
    Returns:
        float: Calibration error when direction=False.
        tuple[float, float]: (over, under) when direction=True.
    """
    n_samples = len(confidences)
    
    if n_samples == 0:
        return (0.0, 0.0) if direction else 0.0
    
    # Compute bin boundaries
    if adaptive_bins:
        # Adaptive binning: equal number of samples per bin
        bin_n = max(1, n_samples // n_bins)
        sorted_indices = np.argsort(confidences)
        bin_boundaries = [0.0]
        
        for i in range(1, n_bins):
            idx = min(i * bin_n, n_samples - 1)
            bin_boundaries.append(confidences[sorted_indices[idx]])
        bin_boundaries.append(1.0)
        
        bin_lowers = np.array(bin_boundaries[:-1])
        bin_uppers = np.array(bin_boundaries[1:])
    else:
        # Uniform binning
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    
    # Compute calibration error for each bin
    bin_errors = []
    bin_weights = []
    over = 0.0
    under = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(
            confidences > bin_lower,
            confidences <= bin_upper
        )
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_confidence = np.mean(confidences[in_bin])
            bin_accuracy = np.mean(accuracies[in_bin])
            gap = bin_confidence - bin_accuracy
            weight = bin_size / n_samples

            if direction:
                if gap > 0:
                    over += weight * gap
                elif gap < 0:
                    under += weight * (-gap)
            else:
                bin_errors.append(np.abs(gap))
                bin_weights.append(weight)
    
    if direction:
        return float(over), float(under)

    if not bin_errors:
        return 0.0
    
    bin_errors = np.array(bin_errors)
    bin_weights = np.array(bin_weights)
    
    # Compute weighted norm
    if norm == 'inf':
        return float(np.max(bin_errors))
    elif norm == 1:
        return float(np.sum(bin_weights * bin_errors))
    elif norm == 2:
        return float(np.sqrt(np.sum(bin_weights * (bin_errors ** 2))))
    else:
        # General L^p norm
        return float((np.sum(bin_weights * (bin_errors ** norm))) ** (1.0 / norm))


# Wrapper functions for common metrics

def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between model confidence and accuracy
    across uniformly-spaced bins. It is defined as:
    ECE = Σ_b (n_b / N) |acc(b) - conf(b)|
    
    Reference:
        Naeini et al. (2015). "Obtaining Well Calibrated Probabilities 
        Using Bayesian Binning." AAAI.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: ECE value between 0 and 1 (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> ece = expected_calibration_error(probs, labels)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=False,
        adaptive_bins=False, top_k_classes=1, norm=1, logits=logits
    )


def maximum_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    MCE is the maximum calibration error across all bins:
    MCE = max_b |acc(b) - conf(b)|
    
    Reference:
        Naeini et al. (2015). "Obtaining Well Calibrated Probabilities 
        Using Bayesian Binning." AAAI.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: MCE value between 0 and 1 (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> mce = maximum_calibration_error(probs, labels)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=False,
        adaptive_bins=False, top_k_classes=1, norm='inf', logits=logits
    )


def root_mean_square_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Root Mean Square Calibration Error (RMSCE).
    
    RMSCE is the root mean square of calibration errors across bins:
    RMSCE = sqrt(Σ_b (n_b / N) (acc(b) - conf(b))^2)
    
    Reference:
        Hendrycks et al. (2019). "Deep Anomaly Detection with Outlier Exposure."
        ICLR.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: RMSCE value between 0 and 1 (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> rmsce = root_mean_square_calibration_error(probs, labels)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=False,
        adaptive_bins=False, top_k_classes=1, norm=2, logits=logits
    )


def static_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Static Calibration Error (SCE).
    
    SCE is the class-conditional calibration error with uniform binning,
    averaged across all classes:
    SCE = (1/K) Σ_k Σ_b (n_bk / N) |acc(b,k) - conf(b,k)|
    
    Reference:
        Nixon et al. (2020). "Measuring Calibration in Deep Learning."
        CVPR Workshops.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: SCE value (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> sce = static_calibration_error(probs, labels)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=True,
        adaptive_bins=False, top_k_classes='all', norm=1, logits=logits
    )


def adaptive_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False,
    direction: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Calculate Adaptive Calibration Error (ACE).
    
    By default, ACE is the class-conditional calibration error with adaptive
    binning (equal number of samples per bin), averaged across all classes.
    
    When direction=True, uses top-1 predictions only (not class-conditional)
    and returns (over, under): the weighted L1 means of max(conf - acc, 0)
    and max(acc - conf, 0) across bins. These are not netted; over + under
    equals top-1 adaptive absolute L1 error.
    
    Reference:
        Nixon et al. (2020). "Measuring Calibration in Deep Learning."
        CVPR Workshops.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        direction: If True, return top-1 (over, under) L1 means. Default: False.
    
    Returns:
        float: Class-conditional ACE when direction=False.
        tuple[float, float]: (over, under) when direction=True, both
            non-negative.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> ace = adaptive_calibration_error(probs, labels)
        >>> over, under = adaptive_calibration_error(probs, labels, direction=True)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins,
        class_conditional=not direction, adaptive_bins=True,
        top_k_classes='all', norm=1, logits=logits, direction=direction
    )


def top_k_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    k: int = 1,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Top-k Calibration Error.
    
    Computes calibration error for the top-k predicted classes,
    averaged across the k classes.
    
    Reference:
        Gupta et al. (2021). "Calibration of Neural Networks using Splines."
        ICLR.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        k: Number of top classes to consider. Default: 1.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: Top-k calibration error (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> top2_ce = top_k_calibration_error(probs, labels, k=2)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=True,
        adaptive_bins=False, top_k_classes=k, norm=1, logits=logits
    )


def thresholded_adaptive_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.01,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Thresholded Adaptive Calibration Error (TACE).
    
    TACE ignores predictions with confidence below a threshold before
    computing the adaptive calibration error.
    
    Reference:
        Nixon et al. (2020). "Measuring Calibration in Deep Learning."
        CVPR Workshops.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        threshold: Confidence threshold. Predictions below this are ignored.
            Default: 0.01.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: TACE value (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> tace = thresholded_adaptive_calibration_error(probs, labels, threshold=0.01)
    """
    return general_calibration_error(
        probabilities, labels, n_bins=n_bins, class_conditional=True,
        adaptive_bins=True, top_k_classes='all', norm=1,
        thresholding=threshold, logits=logits
    )


def overconfidence_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False
) -> float:
    """
    Calculate Overconfidence Error (OE).
    
    OE measures the degree of overconfidence, penalizing confident
    but incorrect predictions more heavily:
    OE = Σ_b (n_b / N) * conf(b) * max(conf(b) - acc(b), 0)
    
    Reference:
        Thulasidasan et al. (2019). "On Mixup Training: Improved Calibration
        and Predictive Uncertainty for Deep Neural Networks." NeurIPS.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
    
    Returns:
        float: OE value (lower is better).
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        >>> labels = np.array([0, 1, 0])
        >>> oe = overconfidence_error(probs, labels)
    """
    # Convert logits to probabilities if needed
    if logits:
        probabilities = softmax(probabilities, axis=1)
    
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    n_samples = len(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    oe = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            confidences > bin_lower,
            confidences <= bin_upper
        )
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_confidence = np.mean(confidences[in_bin])
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_weight = bin_size / n_samples
            
            # Overconfidence penalty
            overconf = max(bin_confidence - bin_accuracy, 0)
            oe += bin_weight * bin_confidence * overconf
    
    return float(oe)


# Convenient aliases
ECE = expected_calibration_error
MCE = maximum_calibration_error
RMSCE = root_mean_square_calibration_error
SCE = static_calibration_error
ACE = adaptive_calibration_error
TACE = thresholded_adaptive_calibration_error
OE = overconfidence_error
GCE = general_calibration_error
