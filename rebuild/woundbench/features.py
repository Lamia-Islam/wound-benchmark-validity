"""Per-pixel feature extraction for the learned segmentation methods.

Manuscript Batch 2, section 3.5:

    Three per-pixel classifiers over fourteen features: CIELAB, HSV and BGR
    channels; local mean and standard deviation of a* and of grayscale over a
    9 x 9 window; and a* and saturation centred on each image's own median. The
    last two express "redder and more saturated than this patient's skin" rather
    than an absolute colour, which is what permits transfer across skin tones
    and illumination. Pixel coordinates are deliberately excluded.

DISCREPANCY (flagged in the code inventory): the manuscript says FOURTEEN
features but the enumeration sums to FIFTEEN:

    BGR (3) + Lab (3) + HSV (3)                      =  9 colour channels
    local mean/SD of a*, local mean/SD of grayscale  =  4 texture
    a* - median(a*), S - median(S)                   =  2 self-referenced
                                                       ---
                                                        15

The "logistic regression over the nine colour features only" variant confirms
the 9-channel block, so the colour count is not in doubt. We implement all 15
and treat "fourteen" as a manuscript typo. If the original really used 14, the
most likely dropped feature is one of the four texture channels; FEATURE_NAMES
below makes it a one-line change to test that.
"""

import cv2
import numpy as np

WINDOW = 9   # local mean / SD window, from the manuscript

COLOUR_FEATURES = ["B", "G", "R", "L", "a", "b", "H", "S", "V"]
TEXTURE_FEATURES = ["a_localmean", "a_localsd", "gray_localmean", "gray_localsd"]
SELF_FEATURES = ["a_centred", "S_centred"]
FEATURE_NAMES = COLOUR_FEATURES + TEXTURE_FEATURES + SELF_FEATURES  # 15

# RESOLVED. The leave-one-out ablation (scripts/feature_ablation.py) identifies
# the missing 14th feature uniquely. Dropping `a_localsd` moves generator A
# performance from 0.936 to 0.122, against the manuscript's reported 0.166;
# every other single drop leaves it at ~0.935. Gap 0.044 versus ~0.77.
#
# The mechanism is the paper's own thesis restated at the level of one input
# feature. Generator A's background is near-uniform (measured background texture
# 2.10 against 3.97-5.07 clinically), so local standard deviation of a* is
# ~0 everywhere outside the lesion. "Is there local texture here?" therefore
# answers "is this the lesion?" exactly, and the classifier gets the answer for
# free. On clinical images the background has real texture and the feature
# stops being a free win.
#
# MANUSCRIPT_14 is that set: the 15 enumerated features minus a_localsd.
MANUSCRIPT_14 = [n for n in FEATURE_NAMES if n != "a_localsd"]
MANUSCRIPT_14_IDX = [FEATURE_NAMES.index(n) for n in MANUSCRIPT_14]
assert len(MANUSCRIPT_14) == 14


def _local_mean_sd(chan: np.ndarray, win: int = WINDOW):
    """Box-filter local mean and standard deviation."""
    c = chan.astype(np.float32)
    mean = cv2.blur(c, (win, win))
    mean_sq = cv2.blur(c * c, (win, win))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, np.sqrt(var)


def extract(bgr: np.ndarray) -> np.ndarray:
    """Return an (H, W, 15) float32 feature stack for one BGR image."""
    bgr = bgr.astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    a_mean, a_sd = _local_mean_sd(a)
    g_mean, g_sd = _local_mean_sd(gray)

    planes = [
        bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2],
        lab[:, :, 0], lab[:, :, 1], lab[:, :, 2],
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        a_mean, a_sd, g_mean, g_sd,
        a - np.median(a),      # redder than THIS patient's skin
        s - np.median(s),      # more saturated than THIS patient's skin
    ]
    return np.stack([p.astype(np.float32) for p in planes], axis=-1)


def extract_flat(bgr: np.ndarray) -> np.ndarray:
    """(H*W, 15) view, for classifier predict()."""
    f = extract(bgr)
    return f.reshape(-1, f.shape[-1])


def colour_only(features: np.ndarray) -> np.ndarray:
    """Slice the 9 colour channels out of a 15-channel stack."""
    return features[..., :len(COLOUR_FEATURES)]
