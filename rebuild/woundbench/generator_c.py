"""Generator C: an independently constructed generator, for external validation.

The texture specification is derived from generator B. If it is a property of
images rather than of that particular code, a generator built on a different
construction principle should land on the same texture-to-validity curve.

Generator C shares no code path with generator B and differs in every mechanism:

|                | generator B                      | generator C                        |
|----------------|----------------------------------|------------------------------------|
| background     | upsampled Gaussian grid mottling | 1/f^beta spectral-filtered noise   |
| illumination   | linear directional ramp          | radial vignette + smooth polynomial|
| nuisance       | hair strokes, gauze, instruments | speckle, film grain, blur gradient |
| lesion shape   | filled polygon, hard edge        | distance-field alpha blend, soft   |
| tissue colour  | HSV offset from skin tone        | Gaussian mixture in CIELAB         |
| grade encoding | blob patches of slough/necrosis  | spatial gradient of tissue mixture |

Nothing here was tuned against any segmentation method or against generator B's
results. The texture knob is `noise_amplitude` and the same measured statistic
(woundbench.imagestats) is applied to both generators, which is the point: the
comparison is on measured texture, not on matching knob values.
"""

from dataclasses import dataclass

import cv2
import numpy as np

IMG_SIZE = 512
SEED = 2024
GRADES = ("healthy", "concerning", "critical")

# Tissue endpoints in CIELAB (L, a, b), uint8 OpenCV convention.
LAB_GRANULATION = (120.0, 158.0, 145.0)
LAB_SLOUGH = (170.0, 138.0, 175.0)
LAB_NECROSIS = (45.0, 140.0, 135.0)

# Grade is the tissue mixture and the edge softness, never size.
GRADE_C = {
    "healthy":    {"mix": (1.00, 0.00, 0.00), "edge": 1.5, "rough": 0.06},
    "concerning": {"mix": (0.60, 0.40, 0.00), "edge": 4.0, "rough": 0.18},
    "critical":   {"mix": (0.35, 0.30, 0.35), "edge": 7.0, "rough": 0.40},
}


@dataclass
class GenCConfig:
    noise_amplitude: float = 14.0   # the texture knob
    spectral_beta: float = 1.6      # 1/f^beta; 1.0 pink, 2.0 brown
    speckle_prob: float = 0.0035
    vignette_strength: float = 0.28
    grain_sigma: float = 2.5
    lesion_frac_lo: float = 0.005
    lesion_frac_hi: float = 0.12
    placement_span: float = 0.60
    skin_L: tuple = (150.0, 205.0)  # CIELAB lightness range for skin


class GeneratorC:
    def __init__(self, seed: int = SEED, size: int = IMG_SIZE,
                 config: GenCConfig | None = None):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.cfg = config or GenCConfig()

    # -- background: spectral noise, a different model from B -------------

    def _spectral_noise(self, beta: float) -> np.ndarray:
        """Zero-mean field with a 1/f^beta radial power spectrum."""
        S = self.size
        white = self.rng.normal(0, 1, (S, S))
        F = np.fft.fftshift(np.fft.fft2(white))
        fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(S)),
                             np.fft.fftshift(np.fft.fftfreq(S)), indexing="ij")
        r = np.sqrt(fx ** 2 + fy ** 2)
        r[r == 0] = 1.0 / S
        F *= 1.0 / (r ** (beta / 2.0))
        out = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        sd = out.std()
        return out / sd if sd > 1e-9 else out

    def _background(self):
        S = self.size
        cfg = self.cfg

        # skin as a CIELAB base, lightness varied, chroma fixed to skin
        L = self.rng.uniform(*cfg.skin_L)
        lab = np.zeros((S, S, 3), np.float32)
        lab[..., 0] = L
        lab[..., 1] = 142.0 + self.rng.normal(0, 2.0)
        lab[..., 2] = 140.0 + self.rng.normal(0, 2.0)

        # texture enters as luminance-only spectral noise
        lab[..., 0] += self._spectral_noise(cfg.spectral_beta) * cfg.noise_amplitude
        # a faint chroma component at a different spectral slope
        lab[..., 1] += self._spectral_noise(cfg.spectral_beta + 0.6) * \
            cfg.noise_amplitude * 0.10

        # radial vignette, a different illumination model from B's ramp
        yy, xx = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S),
                             indexing="ij")
        cx, cy = self.rng.uniform(-0.4, 0.4, 2)
        rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / 1.6
        lab[..., 0] *= (1.0 - cfg.vignette_strength * rad)

        # film grain
        lab[..., 0] += self.rng.normal(0, cfg.grain_sigma, (S, S))

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # speckle: isolated bright/dark specks, a different nuisance from hair
        spk = self.rng.random((S, S)) < cfg.speckle_prob
        sign = np.where(self.rng.random((S, S)) < 0.5, -1, 1)
        img = np.clip(img.astype(np.int16)
                      + (spk * sign * self.rng.integers(25, 60, (S, S)))[..., None],
                      0, 255).astype(np.uint8)
        return img

    # -- lesion: distance-field alpha blend, not a filled polygon ---------

    def _lesion_alpha(self, grade):
        """Soft-edged lesion alpha map plus a hard mask for ground truth."""
        S = self.size
        cfg = self.cfg
        spec = GRADE_C[grade]

        frac = float(np.exp(self.rng.uniform(np.log(cfg.lesion_frac_lo),
                                             np.log(cfg.lesion_frac_hi))))
        r = np.sqrt(frac * S * S / np.pi)
        span = cfg.placement_span * S
        cx = float(np.clip(S / 2 + self.rng.uniform(-span / 2, span / 2),
                           r + 6, S - r - 6))
        cy = float(np.clip(S / 2 + self.rng.uniform(-span / 2, span / 2),
                           r + 6, S - r - 6))

        # implicit radial field perturbed by low-order harmonics
        yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        dx, dy = xx - cx, yy - cy
        theta = np.arctan2(dy, dx)
        rad = np.sqrt(dx ** 2 + dy ** 2)
        warp = np.ones_like(theta)
        for k in (2, 3, 5, 7):
            warp += spec["rough"] * self.rng.normal(0, 1) * np.cos(
                k * theta + self.rng.uniform(0, 2 * np.pi)) / k
        field = rad / np.maximum(r * warp, 1e-6)

        mask = (field <= 1.0).astype(np.uint8)
        # soft alpha from the same field, so the edge is a blend not a step
        alpha = np.clip((1.05 - field) / (spec["edge"] / r * 4 + 1e-6), 0, 1)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0),
                                 max(0.6, spec["edge"] * 0.4))
        return alpha, mask, (cx, cy, r)

    def _tissue_field(self, grade, centre):
        """Tissue colour as a spatial gradient of a CIELAB Gaussian mixture."""
        S = self.size
        cx, cy, r = centre
        gran, slo, nec = GRADE_C[grade]["mix"]
        yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1e-6)

        # necrosis concentrates centrally, slough mid, granulation peripheral
        w_nec = np.exp(-(d / 0.45) ** 2) * nec
        w_slo = np.exp(-((d - 0.55) / 0.35) ** 2) * slo
        w_gra = np.clip(1.0 - w_nec - w_slo, 0, None) * max(gran, 1e-6)
        tot = w_nec + w_slo + w_gra + 1e-6

        lab = np.zeros((S, S, 3), np.float32)
        for w, col in ((w_gra, LAB_GRANULATION), (w_slo, LAB_SLOUGH),
                       (w_nec, LAB_NECROSIS)):
            for c in range(3):
                lab[..., c] += (w / tot) * col[c]
        for c in range(3):
            lab[..., c] += self.rng.normal(0, 3.0, (S, S))
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2BGR)

    def one(self, grade):
        bg = self._background()
        alpha, mask, centre = self._lesion_alpha(grade)
        tissue = self._tissue_field(grade, centre)
        a = alpha[..., None]
        img = (a * tissue.astype(np.float32)
               + (1 - a) * bg.astype(np.float32)).astype(np.uint8)
        # specular sheen over the wet region
        sheen = (self.rng.random((self.size, self.size)) < 0.003) & (mask > 0)
        img[sheen] = np.clip(img[sheen].astype(np.int16) + 50, 0,
                             255).astype(np.uint8)
        return img, mask

    def generate(self, n_per_class: int = 15):
        out = []
        for g in GRADES:
            for i in range(1, n_per_class + 1):
                img, m = self.one(g)
                out.append((g, i, img, m))
        return out


def build(seed: int = SEED, n_per_class: int = 15,
          noise_amplitude: float | None = None):
    cfg = GenCConfig()
    if noise_amplitude is not None:
        cfg.noise_amplitude = noise_amplitude
    recs = GeneratorC(seed=seed, config=cfg).generate(n_per_class)
    return ([r[2] for r in recs], [r[3] for r in recs], [r[0] for r in recs])
