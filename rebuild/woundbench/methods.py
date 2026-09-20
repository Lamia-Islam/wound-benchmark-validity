"""The 22-method registry.

Manuscript Batch 2, section 3. Eighteen classical methods (3.1-3.4), three
learned per-pixel classifiers (3.5), one convolutional network (3.6). Shared
post-processing (3.7) is applied here so no individual method has to remember.

Usage:

    reg = MethodRegistry()
    reg.fit(train_images, train_masks)      # fits the 4 learned methods
    masks = reg.run_all(bgr)                # {name: binary mask}
"""

from collections import OrderedDict

import numpy as np

from .classical import CLASSICAL_METHODS
from . import learned as L
from .postproc import postprocess, postprocess_prob

LEARNED_NAMES = ["logreg_all", "gbm_all", "logreg_colour", "unet"]
METHOD_NAMES = list(CLASSICAL_METHODS.keys()) + LEARNED_NAMES

assert len(METHOD_NAMES) == 22, len(METHOD_NAMES)

# Manuscript 5.4: threshold selected by sweep on FUSeg training images 250-400,
# then applied unchanged to every benchmark.
DEFAULT_LEARNED_THRESHOLD = 0.9
DEFAULT_UNET_THRESHOLD = 0.5


class MethodRegistry:
    """Holds the fitted learned models and dispatches all 22 methods."""

    def __init__(self,
                 learned_threshold=DEFAULT_LEARNED_THRESHOLD,
                 unet_threshold=DEFAULT_UNET_THRESHOLD):
        self.learned_threshold = learned_threshold
        self.unet_threshold = unet_threshold
        self.logreg_all = None
        self.gbm_all = None
        self.logreg_colour = None
        self.unet = None

    # -- fitting ----------------------------------------------------------

    def fit_pixel_classifiers(self, images, masks, seed=0, log=print):
        X, y = L.sample_pixels(images, masks, seed=seed)
        if log:
            log(f"  pixel sample: {X.shape[0]:,} px x {X.shape[1]} features "
                f"({y.mean():.1%} lesion)")
        self.logreg_all = L.fit_logreg(X, y, colour_only=False, seed=seed)
        self.logreg_colour = L.fit_logreg(X, y, colour_only=True, seed=seed)
        self.gbm_all = L.fit_gbm(X, y, seed=seed)
        return self

    def fit_unet(self, images, masks, seed=0, log=print, **kw):
        self.unet = L.train_unet(images, masks, seed=seed, log=log, **kw)
        return self

    def fit(self, images, masks, seed=0, log=print):
        self.fit_pixel_classifiers(images, masks, seed=seed, log=log)
        self.fit_unet(images, masks, seed=seed, log=log)
        return self

    @property
    def fitted(self) -> bool:
        return all(m is not None for m in
                   (self.logreg_all, self.gbm_all, self.logreg_colour, self.unet))

    # -- inference --------------------------------------------------------

    def probability(self, name, bgr):
        """Probability field for a learned method, else None."""
        if name == "logreg_all":
            return L.predict_prob(self.logreg_all, bgr, colour_only=False)
        if name == "logreg_colour":
            return L.predict_prob(self.logreg_colour, bgr, colour_only=True)
        if name == "gbm_all":
            return L.predict_prob(self.gbm_all, bgr, colour_only=False)
        if name == "unet":
            return L.unet_prob(self.unet, bgr)
        return None

    def run(self, name, bgr) -> np.ndarray:
        """Run one method with its shared post-processing. Returns uint8 0/1."""
        if name in CLASSICAL_METHODS:
            return postprocess(CLASSICAL_METHODS[name](bgr))
        if name in LEARNED_NAMES:
            if not self.fitted:
                raise RuntimeError("learned methods not fitted; call fit() first")
            prob = self.probability(name, bgr)
            thr = (self.unet_threshold if name == "unet"
                   else self.learned_threshold)
            return postprocess_prob(prob, thr)
        raise KeyError(name)

    def run_all(self, bgr, names=None) -> "OrderedDict[str, np.ndarray]":
        names = names or METHOD_NAMES
        out = OrderedDict()
        for n in names:
            if n in LEARNED_NAMES and not self.fitted:
                continue
            try:
                out[n] = self.run(n, bgr)
            except Exception:
                # A method that crashes on an image yields an empty mask rather
                # than aborting the benchmark. Logged by the caller.
                out[n] = np.zeros(bgr.shape[:2], np.uint8)
        return out


# Grouping used for the results tables, per manuscript section headings.
METHOD_FAMILY = {
    **{n: "background-referenced" for n in
       ["adaptive_gaussian", "adaptive_mean", "otsu_gray",
        "bgr_corner_distance", "hsv_corner_distance"]},
    **{n: "colour-referenced" for n in
       ["otsu_a", "pct92_a", "redness_ratio", "otsu_saturation"]},
    **{n: "clustering" for n in
       ["kmeans_lab_k3", "kmeans_lab_k5", "kmeans_lab_k4_higha",
        "meanshift_otsu_a"]},
    **{n: "region/contour" for n in
       ["watershed_a", "grabcut_center", "floodfill_a", "chanvese_a",
        "canny_fill"]},
    **{n: "learned" for n in ["logreg_all", "gbm_all", "logreg_colour"]},
    "unet": "CNN",
}

assert set(METHOD_FAMILY) == set(METHOD_NAMES)
