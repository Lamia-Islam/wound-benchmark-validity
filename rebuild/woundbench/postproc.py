"""Shared post-processing applied identically to every method.

Manuscript Batch 2, section 3.7:

    All methods receive identical post-processing: morphological closing then
    opening with an elliptical kernel, and removal of connected components below
    0.05% of frame area. Learned methods additionally receive Gaussian smoothing
    of the probability field before thresholding, rather than morphological
    cleaning of a binary mask, because morphology on a noisy binary mask
    preferentially erases small lesions.

NOT RECORDED in the manuscript: the morphological kernel size, and the Gaussian
sigma for the learned probability field. Values below are this rebuild's choices
and are collected here so a sensitivity sweep can change them in one place.
"""

import cv2
import numpy as np

# Rebuild choices, not recovered from the manuscript.
MORPH_KERNEL = 5          # elliptical structuring element, side length in px
CLOSE_ITERS = 1
OPEN_ITERS = 1
MIN_COMPONENT_FRAC = 0.0005   # 0.05% of frame area  (this IS from the manuscript)
PROB_GAUSSIAN_SIGMA = 2.0     # learned methods only


def _kernel(size: int = MORPH_KERNEL) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def remove_small_components(mask: np.ndarray,
                            min_frac: float = MIN_COMPONENT_FRAC) -> np.ndarray:
    """Drop connected components smaller than `min_frac` of the frame."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = min_frac * mask.size
    out = np.zeros_like(mask)
    for i in range(1, n):                      # 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 1
    return out


def postprocess(mask: np.ndarray) -> np.ndarray:
    """Close, then open, then drop small components. Returns uint8 0/1."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    k = _kernel()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    return remove_small_components(mask)


def postprocess_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Learned-method path: smooth the probability field, then threshold.

    No morphology on the binary mask, per section 3.7.
    """
    smoothed = cv2.GaussianBlur(prob.astype(np.float32), (0, 0),
                                PROB_GAUSSIAN_SIGMA)
    mask = (smoothed >= threshold).astype(np.uint8)
    return remove_small_components(mask)
