"""Portable image statistics, so the dose-response axis is MEASURED not set.

The texture dose-response is only a specification if a third party can compute
where their own benchmark sits on it. `mottle_amplitude` is a field in this
generator's config and means nothing outside it. These statistics are computed
from pixels and apply to any image set, synthetic or clinical, with or without
masks.

Primary statistic
-----------------
`background_texture` : median over images of the median local standard deviation
of luminance within the background region, at a fixed window, after dividing out
low-frequency shading.

Two variants:
  - masked    : background = everything outside the ground-truth mask. Use when
                the benchmark ships masks (all of them do, that is what makes it
                a benchmark).
  - unmasked  : computed over the whole frame. Use to characterise an image set
                with no masks to hand. Reported alongside so the two can be
                related.

Design notes
------------
Local SD at a small window responds to fine texture and is insensitive to the
large-scale illumination gradient, which is a separate design property. We
divide out a heavily blurred copy first so that a bright image and a dark image
of the same material give the same number: the statistic is contrast-like, not
intensity-like.
"""

import cv2
import numpy as np

WINDOW = 9          # local SD window, matches the feature extractor's window
SHADING_SIGMA = 31  # low-frequency shading removed before measuring
CANONICAL_SIDE = 512  # every image is resampled to this before measuring

# RESOLUTION IS PART OF THE DEFINITION.
#
# A fixed-pixel window measures texture at a fixed pixel scale, so the same
# scene photographed at two resolutions gives two different answers. Measured
# natively, Medetec (224 px) reads 5.07 and FUSeg (512 px) reads 3.88.
# Resampled to a common 512, they read 2.69 and 3.88: the ORDERING REVERSES.
# Comparing image sets at their native resolutions is therefore invalid, and an
# earlier version of this analysis did exactly that.
#
# The statistic consequently resamples every image to CANONICAL_SIDE before
# measuring. Any reported value must state the canonical side and the window,
# because the number is meaningless without them.
#
# Measured at 512 with a 9 px window:
#     Medetec   2.69      FUSeg   3.88      generator A   2.10


def _luminance(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _detrended(bgr):
    """Luminance divided by its own low-frequency shading, scaled to ~[0,255]."""
    lum = _luminance(bgr)
    shading = cv2.GaussianBlur(lum, (0, 0), SHADING_SIGMA)
    shading = np.maximum(shading, 1.0)
    return (lum / shading) * 128.0


def local_sd(img_plane, window: int = WINDOW):
    c = img_plane.astype(np.float32)
    mean = cv2.blur(c, (window, window))
    mean_sq = cv2.blur(c * c, (window, window))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _to_canonical(bgr, mask=None, side: int = CANONICAL_SIDE):
    """Resample to the canonical side so measurements are comparable."""
    if side is None or bgr.shape[0] == side and bgr.shape[1] == side:
        return bgr, mask
    out_i = cv2.resize(bgr, (side, side), interpolation=cv2.INTER_AREA
                       if bgr.shape[0] > side else cv2.INTER_LINEAR)
    out_m = None
    if mask is not None:
        out_m = cv2.resize(np.asarray(mask).astype(np.uint8), (side, side),
                           interpolation=cv2.INTER_NEAREST)
    return out_i, out_m


def texture_of_image(bgr, mask=None, window: int = WINDOW,
                     side: int = CANONICAL_SIDE):
    """Median local SD of detrended luminance, over background or whole frame.

    The image is resampled to `side` first; see the note at the top of this
    module for why that is part of the definition rather than a convenience.
    """
    bgr, mask = _to_canonical(bgr, mask, side)
    sd = local_sd(_detrended(bgr), window)
    if mask is not None:
        bg = (np.asarray(mask) == 0)
        # exclude a margin around the lesion so the lesion edge does not
        # contribute its own step to the background texture measurement
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window * 3, window * 3))
        near = cv2.dilate((np.asarray(mask) > 0).astype(np.uint8), k) > 0
        bg = bg & (~near)
        if bg.sum() < 100:
            bg = (np.asarray(mask) == 0)
        vals = sd[bg]
    else:
        vals = sd.reshape(-1)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def texture_of_set(images, masks=None, window: int = WINDOW):
    """Returns (masked_median, unmasked_median) over an image set."""
    m_vals, u_vals = [], []
    for i, im in enumerate(images):
        u_vals.append(texture_of_image(im, None, window))
        if masks is not None:
            m_vals.append(texture_of_image(im, masks[i], window))
    masked = float(np.median(m_vals)) if m_vals else float("nan")
    return masked, float(np.median(u_vals))


def summarise(name, images, masks=None):
    m, u = texture_of_set(images, masks)
    return {"name": name, "n": len(images),
            "background_texture_masked": m,
            "background_texture_unmasked": u}
