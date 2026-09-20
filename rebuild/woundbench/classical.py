"""The 18 classical (non-learned) segmentation methods.

Manuscript Batch 2, sections 3.1-3.4. Every method takes a BGR uint8 image and
returns a binary mask; no method receives per-image tuning. Shared
post-processing (section 3.7) is applied by the registry in methods.py, not
here, so these functions return raw masks.

Free parameters NOT recorded in the manuscript are marked `REBUILD CHOICE`.
They are collected at the top of each function so a sensitivity sweep can find
them.
"""

import cv2
import numpy as np
from skimage.segmentation import chan_vese

CORNER_PATCH = 20   # "four 20 x 20 image corners", from the manuscript


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _corner_pixels(img: np.ndarray, patch: int = CORNER_PATCH) -> np.ndarray:
    """Stack the four corner patches as an (N, C) array."""
    h, w = img.shape[:2]
    p = min(patch, h // 2, w // 2)
    corners = [img[:p, :p], img[:p, w - p:], img[h - p:, :p], img[h - p:, w - p:]]
    flat = [c.reshape(-1, img.shape[2]) if img.ndim == 3 else c.reshape(-1, 1)
            for c in corners]
    return np.concatenate(flat, axis=0)


def _otsu(chan: np.ndarray) -> np.ndarray:
    """Otsu threshold on a single channel, rescaled to uint8 first."""
    c = chan.astype(np.float32)
    lo, hi = float(c.min()), float(c.max())
    if hi - lo < 1e-9:
        return np.zeros(c.shape, np.uint8)
    u8 = ((c - lo) / (hi - lo) * 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(np.uint8)


def _lab_a(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def _invert_if_mostly_foreground(mask: np.ndarray, limit: float = 0.5) -> np.ndarray:
    """Otsu has no notion of which side is the lesion.

    REBUILD CHOICE. The manuscript does not state how polarity was resolved for
    the Otsu-family methods. We take the convention that a mask claiming more
    than `limit` of the frame has its polarity flipped. This matters: on
    uniform-background synthetic images the lesion is a minority of the frame,
    and on clinical images it is a small minority, so the convention is
    consistent across benchmarks rather than tuned per benchmark.
    """
    if mask.mean() > limit:
        return 1 - mask
    return mask


# --------------------------------------------------------------------------
# 3.1 Background-referenced methods (5)
# --------------------------------------------------------------------------

def adaptive_gaussian(bgr):
    """Grayscale adaptive threshold, Gaussian variant.

    REBUILD CHOICE: block size 31, C = 6, taken from the Track A
    `abcde_analyzer.preprocess`, which is the ancestor of this method and is the
    only recorded parameterisation.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    return (cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 6)).astype(np.uint8)


def adaptive_mean(bgr):
    """Grayscale adaptive threshold, mean variant."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    return (cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 6)).astype(np.uint8)


def otsu_gray(bgr):
    """Global Otsu on grayscale [9]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _invert_if_mostly_foreground(_otsu(gray))


def bgr_corner_distance(bgr):
    """Euclidean distance from the median corner colour in BGR, Otsu'd."""
    ref = np.median(_corner_pixels(bgr), axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - ref, axis=2)
    return _otsu(dist)


def hsv_corner_distance(bgr):
    """Euclidean distance from the median corner colour in HSV, Otsu'd.

    This is the Track A 'replacement' segmentation, and doubles as a
    consistency check against the recovered 25,236 px^2 figure on healthy_01.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ref = np.median(_corner_pixels(hsv), axis=0)
    dist = np.linalg.norm(hsv - ref, axis=2)
    return _otsu(dist)


# --------------------------------------------------------------------------
# 3.2 Colour-referenced methods (4)
# --------------------------------------------------------------------------

def otsu_a(bgr):
    """Otsu on the CIELAB a* channel."""
    return _invert_if_mostly_foreground(_otsu(_lab_a(bgr)))


def pct92_a(bgr):
    """92nd percentile of a*: pixels above that value are lesion."""
    a = _lab_a(bgr)
    return (a >= np.percentile(a, 92)).astype(np.uint8)


def redness_ratio(bgr):
    """R / (G + B), Otsu'd.

    The ratio is unbounded: pixels where G+B is near zero send it to ~1e6, and a
    plain min-max rescale to uint8 then collapses the entire image into a
    handful of distinct values, leaving Otsu nothing to separate and returning
    an empty mask on every image. Clipping to the 1st-99th percentile before
    thresholding fixes it. This is a numerical-robustness fix, not a tuning
    choice: without it the method is degenerate rather than merely weak, and
    would occupy a rank it did not earn.
    """
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    ratio = r / (g + b + 1e-6)
    lo, hi = np.percentile(ratio, [1, 99])
    ratio = np.clip(ratio, lo, hi)
    return _invert_if_mostly_foreground(_otsu(ratio))


def otsu_saturation(bgr):
    """Otsu on HSV saturation."""
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return _invert_if_mostly_foreground(_otsu(s))


# --------------------------------------------------------------------------
# 3.3 Clustering methods (4)
# --------------------------------------------------------------------------

def _kmeans_labels(bgr, k, attempts=3, seed=0):
    """k-means in CIELAB. REBUILD CHOICE: 3 attempts, 20 iters, eps 1.0."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(flat, k, None, crit, attempts,
                                    cv2.KMEANS_PP_CENTERS)
    return labels.reshape(lab.shape[:2]), centers


def _kmeans_drop_corner_cluster(bgr, k):
    """Modal corner cluster is background; everything else is lesion."""
    labels, _ = _kmeans_labels(bgr, k)
    h, w = labels.shape
    p = min(CORNER_PATCH, h // 2, w // 2)
    corner_lbls = np.concatenate([
        labels[:p, :p].ravel(), labels[:p, w - p:].ravel(),
        labels[h - p:, :p].ravel(), labels[h - p:, w - p:].ravel()])
    bg = np.bincount(corner_lbls, minlength=k).argmax()
    return (labels != bg).astype(np.uint8)


def kmeans_lab_k3(bgr):
    return _kmeans_drop_corner_cluster(bgr, 3)


def kmeans_lab_k5(bgr):
    return _kmeans_drop_corner_cluster(bgr, 5)


def kmeans_lab_k4_higha(bgr):
    """k = 4, retain the cluster with the highest mean a*.

    Asks which cluster resembles wound tissue rather than which is not
    background. This is the method the manuscript singles out as conceptually
    different from the corner-referenced family.
    """
    labels, _ = _kmeans_labels(bgr, 4)
    a = _lab_a(bgr)
    means = [a[labels == i].mean() if (labels == i).any() else -np.inf
             for i in range(4)]
    return (labels == int(np.argmax(means))).astype(np.uint8)


def meanshift_otsu_a(bgr):
    """Pyramid mean-shift filtering [13] then Otsu on a*.

    REBUILD CHOICE: spatial radius 21, colour radius 51, max level 1.
    """
    filt = cv2.pyrMeanShiftFiltering(bgr, 21, 51, maxLevel=1)
    return _invert_if_mostly_foreground(_otsu(_lab_a(filt)))


# --------------------------------------------------------------------------
# 3.4 Region and contour methods (5)
# --------------------------------------------------------------------------

def watershed_a(bgr):
    """Marker-based watershed [11] seeded from a* extremes.

    REBUILD CHOICE: sure-foreground above the 95th percentile of a*,
    sure-background below the 50th.
    """
    a = _lab_a(bgr)
    fg = (a >= np.percentile(a, 95)).astype(np.uint8)
    bg = (a <= np.percentile(a, 50)).astype(np.uint8)
    markers = np.zeros(a.shape, np.int32)
    markers[bg > 0] = 1
    markers[fg > 0] = 2
    if (markers == 2).sum() == 0 or (markers == 1).sum() == 0:
        return np.zeros(a.shape, np.uint8)
    cv2.watershed(bgr.copy(), markers)
    return (markers == 2).astype(np.uint8)


def grabcut_center(bgr):
    """GrabCut [14] initialised from a central rectangle covering 70% of frame.

    REBUILD CHOICE: 3 iterations.
    """
    h, w = bgr.shape[:2]
    # 70% of frame AREA -> sqrt(0.7) of each side
    side = np.sqrt(0.70)
    rw, rh = int(w * side), int(h * side)
    rect = ((w - rw) // 2, (h - rh) // 2, rw, rh)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)


def floodfill_a(bgr):
    """Flood-fill region growing from the a* maximum.

    REBUILD CHOICE: tolerance of 8 a*-units either side of the seed value.
    """
    a = _lab_a(bgr)
    h, w = a.shape
    seed = np.unravel_index(int(np.argmax(a)), a.shape)
    a8 = np.clip(a, 0, 255).astype(np.uint8)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8)
    cv2.floodFill(a8.copy(), ff_mask, (int(seed[1]), int(seed[0])), 0,
                  loDiff=8, upDiff=8, flags=flags)
    return ff_mask[1:-1, 1:-1].astype(np.uint8)


def chanvese_a(bgr):
    """Two-phase Chan-Vese style region competition on a*, no edge term [12].

    REBUILD CHOICE: mu = 0.1, 120 iterations, dt = 0.5, on a half-scale image
    for tractability, then resized back.
    """
    a = _lab_a(bgr)
    small = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    lo, hi = float(small.min()), float(small.max())
    if hi - lo < 1e-9:
        return np.zeros(a.shape, np.uint8)
    norm = (small - lo) / (hi - lo)
    try:
        phi = chan_vese(norm, mu=0.1, lambda1=1, lambda2=1, tol=1e-3,
                        max_num_iter=120, dt=0.5, init_level_set="checkerboard",
                        extended_output=False)
    except Exception:
        return np.zeros(a.shape, np.uint8)
    seg = np.asarray(phi, dtype=np.uint8)
    # Chan-Vese polarity is arbitrary; keep the phase with the higher mean a*.
    if seg.sum() and (1 - seg).sum():
        if norm[seg == 1].mean() < norm[seg == 0].mean():
            seg = 1 - seg
    return cv2.resize(seg, (a.shape[1], a.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def canny_fill(bgr):
    """Canny [10] with morphological closure and contour filling.

    REBUILD CHOICE: hysteresis thresholds 50/150, closing kernel 5x5.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(gray.shape, np.uint8)
    if not contours:
        return out
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


CLASSICAL_METHODS = {
    # 3.1 background-referenced
    "adaptive_gaussian": adaptive_gaussian,
    "adaptive_mean": adaptive_mean,
    "otsu_gray": otsu_gray,
    "bgr_corner_distance": bgr_corner_distance,
    "hsv_corner_distance": hsv_corner_distance,
    # 3.2 colour-referenced
    "otsu_a": otsu_a,
    "pct92_a": pct92_a,
    "redness_ratio": redness_ratio,
    "otsu_saturation": otsu_saturation,
    # 3.3 clustering
    "kmeans_lab_k3": kmeans_lab_k3,
    "kmeans_lab_k5": kmeans_lab_k5,
    "kmeans_lab_k4_higha": kmeans_lab_k4_higha,
    "meanshift_otsu_a": meanshift_otsu_a,
    # 3.4 region and contour
    "watershed_a": watershed_a,
    "grabcut_center": grabcut_center,
    "floodfill_a": floodfill_a,
    "chanvese_a": chanvese_a,
    "canny_fill": canny_fill,
}

assert len(CLASSICAL_METHODS) == 18, len(CLASSICAL_METHODS)
