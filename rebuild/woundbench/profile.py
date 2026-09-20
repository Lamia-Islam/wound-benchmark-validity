"""Multi-scale, multi-channel texture profile.

The single-channel single-scale statistic failed external validation: three
benchmarks at matched luminance texture gave rank agreement 0.121, 0.434 and
0.189, and texture had no effect at all inside generator C. Measuring chroma
separately showed why: generator B's chroma texture sits at five times the
clinical value at every level and its texture knob never touched it, while
generator C has almost no chroma structure at all.

So the quantity is a PROFILE, not a number: local standard deviation of the
background, measured in three channels at four spatial scales, giving a
12-dimensional descriptor of how a benchmark's background varies.

The claim to be tested is then a matching claim: a benchmark whose profile is
close to the clinical profile should predict clinical method ranking well,
REGARDLESS of which generator produced it. That is the test the single statistic
failed.
"""

import cv2
import numpy as np

CANONICAL_SIDE = 512
SCALES = (9, 21, 45, 91)
CHANNELS = ("L", "a", "b")
PROFILE_KEYS = [f"{c}@{s}" for c in CHANNELS for s in SCALES]


def _canonical(bgr, mask=None, side=CANONICAL_SIDE):
    if bgr.shape[0] == side and bgr.shape[1] == side:
        return bgr, mask
    interp = cv2.INTER_AREA if bgr.shape[0] > side else cv2.INTER_LINEAR
    out_i = cv2.resize(bgr, (side, side), interpolation=interp)
    out_m = None
    if mask is not None:
        out_m = cv2.resize(np.asarray(mask).astype(np.uint8), (side, side),
                           interpolation=cv2.INTER_NEAREST)
    return out_i, out_m


def _local_sd(plane, win):
    c = plane.astype(np.float32)
    mean = cv2.blur(c, (win, win))
    msq = cv2.blur(c * c, (win, win))
    return np.sqrt(np.maximum(msq - mean * mean, 0.0))


def _background_mask(mask, win, shape):
    if mask is None:
        return np.ones(shape, bool)
    m = np.asarray(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win * 2 + 1, win * 2 + 1))
    near = cv2.dilate((m > 0).astype(np.uint8), k) > 0
    bg = (m == 0) & (~near)
    if bg.sum() < 200:
        bg = (m == 0)
    return bg


def profile_of_image(bgr, mask=None):
    """12-dim profile: local SD in L, a, b at four scales, over background."""
    bgr, mask = _canonical(bgr, mask)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # divide out low-frequency shading in L only, so the descriptor is
    # contrast-like rather than exposure-like. Chroma is already
    # exposure-insensitive.
    L = lab[..., 0]
    shade = np.maximum(cv2.GaussianBlur(L, (0, 0), 31), 1.0)
    planes = {"L": (L / shade) * 128.0, "a": lab[..., 1], "b": lab[..., 2]}

    out = {}
    for ch in CHANNELS:
        for s in SCALES:
            sd = _local_sd(planes[ch], s)
            bg = _background_mask(mask, s, sd.shape)
            out[f"{ch}@{s}"] = float(np.median(sd[bg])) if bg.any() else float("nan")
    return out


def profile_of_set(images, masks=None):
    rows = []
    for i, im in enumerate(images):
        rows.append(profile_of_image(im, None if masks is None else masks[i]))
    return {k: float(np.median([r[k] for r in rows])) for k in PROFILE_KEYS}


def to_vector(prof):
    return np.array([prof[k] for k in PROFILE_KEYS], float)


def distance(prof, reference, eps=1e-3):
    """Mean absolute log-ratio between two profiles.

    Scale-free: a benchmark with twice the clinical texture everywhere and one
    with half of it are equally wrong, which is the behaviour a matching
    criterion needs. Averaging over the 12 entries keeps every channel and scale
    equally weighted rather than letting the largest-magnitude entry dominate.
    """
    a = np.maximum(to_vector(prof), eps)
    b = np.maximum(to_vector(reference), eps)
    return float(np.mean(np.abs(np.log(a / b))))


def per_entry_log_ratio(prof, reference, eps=1e-3):
    a = np.maximum(to_vector(prof), eps)
    b = np.maximum(to_vector(reference), eps)
    return dict(zip(PROFILE_KEYS, np.log(a / b)))
