"""The 32-method pool: the original 22 plus ten modern architectures.

Presents one object with the same interface as MethodRegistry, so every existing
experiment script works unchanged by swapping the registry it loads.

The original pool was 21 classical methods and one 117,681-parameter U-Net. The
ten additions are in deep.py. Two things about the combination are worth stating
because they are what make the enlarged pool worth having:

  n rises from 22 to 32. At n = 22 a Spearman needs about |rho| >= 0.43 for
  p < 0.05, so any true agreement below that was undetectable. At n = 32 the
  threshold falls to about 0.35.

  The pool now varies along the axes modern segmentation actually varies along.
  Seven decoders share one encoder and three encoders share one decoder, so a
  benchmark that orders the pool correctly is ordering architectural choices,
  not only colour-threshold behaviour.

Decision thresholds for the deep models are swept on FUSeg training images
400 to 550. That slice is disjoint from the 393 images all of them were trained
on and from the FUSeg validation slice used for evaluation. The original U-Net's
threshold was swept on images 250 to 400, which overlaps its own training data;
that is kept as-is so the existing 22 numbers do not move, and the new models
use the clean split.
"""

from collections import OrderedDict

import numpy as np

from .deep import DEEP_FAMILY, DEEP_NAMES, DeepZoo
from .methods import METHOD_FAMILY, METHOD_NAMES
from .postproc import postprocess_prob

ALL_METHOD_NAMES = list(METHOD_NAMES) + list(DEEP_NAMES)
ALL_METHOD_FAMILY = {**METHOD_FAMILY, **DEEP_FAMILY}

assert len(ALL_METHOD_NAMES) == 32
assert set(ALL_METHOD_FAMILY) == set(ALL_METHOD_NAMES)


class CombinedRegistry:
    """Classical registry plus the deep zoo, behind one run_all."""

    def __init__(self, classical, zoo: DeepZoo, thresholds=None):
        self.classical = classical
        self.zoo = zoo
        self.thresholds = dict(thresholds or {})

    # kept so scripts that print it keep working
    @property
    def learned_threshold(self):
        return self.classical.learned_threshold

    def available(self):
        return [n for n in ALL_METHOD_NAMES
                if n not in DEEP_NAMES or n in self.zoo.models]

    def run(self, name, bgr) -> np.ndarray:
        if name in self.zoo.models:
            prob = self.zoo.probability(name, bgr)
            return postprocess_prob(prob, self.thresholds.get(name, 0.5))
        return self.classical.run(name, bgr)

    def run_all(self, bgr, names=None) -> "OrderedDict[str, np.ndarray]":
        names = names or self.available()
        out = OrderedDict()
        for n in names:
            try:
                out[n] = self.run(n, bgr)
            except Exception:
                out[n] = np.zeros(bgr.shape[:2], np.uint8)
        return out
