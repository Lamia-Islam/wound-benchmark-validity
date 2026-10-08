"""The 21 synthetic sets, regenerated from the generator code.

Generator B cells: texture amplitude 11, 28, 40 x seeds 2024, 7, 101.
Table 8 benchmarks: A (uniform drape), B with texture ablated, C without
noise, C with noise 3.4, each at seeds 2024, 7, 101. 45 images each.

fingerprint() hashes the raw arrays, so the GPU run can confirm it scores
exactly the images the local 32-method pool was scored on.
"""

import hashlib
from dataclasses import replace

import numpy as np

from woundbench.generator_a import build as build_a
from woundbench.generator_b import GenBConfig, GeneratorB
from woundbench.generator_c import build as build_c

SEEDS = [2024, 7, 101]
AMPS = [11.0, 28.0, 40.0]


def _b(seed, amp):
    r = GeneratorB(seed=seed, config=replace(GenBConfig(),
                   mottle_amplitude=amp)).generate(15)
    return [x[2] for x in r], [x[3] for x in r]


def synth_sets():
    for amp in AMPS:
        for s in SEEDS:
            yield f"genB_amp{int(amp)}_s{s}", (lambda s=s, a=amp: _b(s, a))
    for s in SEEDS:
        yield f"nrA_s{s}", (lambda s=s: build_a(seed=s, n_per_class=15)[:2])
        yield f"nrB0_s{s}", (lambda s=s: _b(s, 0.0))
        yield f"nrC0_s{s}", (lambda s=s: build_c(seed=s, n_per_class=15,
                                                 noise_amplitude=0.0)[:2])
        yield f"nrC34_s{s}", (lambda s=s: build_c(seed=s, n_per_class=15,
                                                  noise_amplitude=3.4)[:2])


def fingerprint(img, mask):
    h = hashlib.md5(np.ascontiguousarray(img).tobytes())
    h.update(np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8)).tobytes())
    return h.hexdigest()
