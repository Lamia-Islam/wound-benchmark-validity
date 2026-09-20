"""Generator B: lesion embedded in textured skin, plus the five ablations.

Manuscript Batch 2, section 4.2. The design-target table, recovered verbatim:

    | property     | generator B                                            |
    |--------------|--------------------------------------------------------|
    | lesion frac  | log-uniform, 0.005-0.12 of frame, sampled independently |
    |              | of grade                                                |
    | placement    | uniform over the central 60% of frame                   |
    | background   | textured skin: low-frequency mottling, fine noise,      |
    |              | directional lighting gradient, hair                     |
    | skin tone    | varied per image over a plausible BGR range             |
    | clutter      | gauze and instrument-like regions in a minority of      |
    |              | frames                                                  |
    | lesion shape | radially perturbed polygon                              |
    | specularity  | highlights over moist tissue                            |

    Grade is defined by tissue composition (granulation, slough, necrosis
    proportions) and boundary irregularity, never by size.

    Both generators emit exact ground-truth masks and are deterministic given a
    seed.

Section 4.2 also states the design intent, which constrains the implementation:

    Generator B was constructed independently, targeting measured properties of
    clinical wound photographs rather than the weaknesses of any segmentation
    method. Building a generator to defeat corner sampling would be as circular
    as building one that rewards it.

So nothing here is tuned against a method. Every parameter is either recovered
from the table above or a REBUILD CHOICE picked for plausibility, and the
choices are collected in GenBConfig so they can be swept.

Section 7.2 records what the texture is made of, which pins down more than the
table alone:

    our generator B uses low-frequency mottling, additive noise, a directional
    illumination gradient, and simple hair strokes.

ABLATIONS (section 5.3): each reverts one feature to generator A's
corresponding assumption, all other features held at generator B settings, seed
fixed.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

IMG_SIZE = 512
SEED = 2024
GRADES = ("healthy", "concerning", "critical")

# Tissue palette, kept consistent with generator A so that colour is not a
# confound between the two benchmarks.
GRANULATION = (100, 130, 210)    # well-perfused granulation tissue
SLOUGH = (80, 190, 220)          # fibrinous slough
NECROSIS = (30, 35, 55)          # necrotic / eschar

# Grade is tissue composition + boundary irregularity, never size.
# (granulation, slough, necrosis) proportions, and radial irregularity.
GRADE_SPEC = {
    "healthy":    {"mix": (1.00, 0.00, 0.00), "irregularity": 0.08},
    "concerning": {"mix": (0.65, 0.35, 0.00), "irregularity": 0.22},
    "critical":   {"mix": (0.40, 0.30, 0.30), "irregularity": 0.48},
}


@dataclass
class GenBConfig:
    """Every free parameter in one place. Ablations flip the booleans."""
    # --- recovered from the manuscript ---
    lesion_frac_lo: float = 0.005
    lesion_frac_hi: float = 0.12
    placement_span: float = 0.60        # central 60% of frame

    # --- feature switches, flipped by the ablations ---
    grade_independent_size: bool = True
    off_centre_placement: bool = True
    textured_background: bool = True
    clutter: bool = True
    irregular_shape: bool = True

    # --- REBUILD CHOICES (amplitudes were never recorded) ---
    mottle_scale: int = 16              # low-freq mottling: noise grid before upscale
    mottle_amplitude: float = 11.0      # luminance amplitude
    mottle_chroma_ratio: float = 0.18   # chroma as a fraction of luminance
    fine_noise_sigma: float = 4.0
    fine_noise_chroma_ratio: float = 0.12
    gradient_strength: float = 0.22     # directional illumination
    n_hairs: tuple = (8, 26)
    hair_len_frac: tuple = (0.05, 0.18)  # of frame side
    hair_darkness: int = 45
    clutter_prob: float = 0.30          # "a minority of frames"
    n_polygon_pts: int = 28
    specular_prob: float = 0.004
    specular_boost: int = 55
    skin_tone_lo: tuple = (105, 128, 158)   # deeper skin, B < G < R
    skin_tone_hi: tuple = (180, 198, 222)   # lighter skin, same ordering
    skin_warmth_jitter: float = 7.0

    # Lesion-to-perilesional colour contrast, CALIBRATED AGAINST CLINICAL DATA.
    #
    # Measured over Medetec (n=152) and FUSeg validation (n=114), comparing
    # lesion pixels against a 41px perilesional ring:
    #
    #     dataset   HSV saturation delta   CIELAB a* delta
    #     Medetec          +29.6                 +4.7
    #     FUSeg            +36.8                +10.3
    #     ---------------------------------------------
    #     target           ~+33                  ~+8
    #
    # For comparison, generator A sits at +55.8 / +16.5, and an earlier version
    # of this generator sat at +72.6 / +18.5, i.e. roughly twice the clinical
    # contrast. At that contrast a simple saturation threshold separates lesion
    # from skin trivially, which clinical images do not permit.
    #
    # This is a measured property of clinical photographs, per section 4.2's
    # design principle, not a parameter tuned to defeat any method.
    lesion_sat_delta: float = 33.0
    lesion_sat_jitter: float = 9.0
    lesion_val_delta: float = -12.0     # wound beds read slightly darker


# Generator A's corresponding assumptions, used by the ablations.
GEN_A_GRADE_SIZE = {"healthy": 0.030, "concerning": 0.042, "critical": 0.075}


class GeneratorB:
    def __init__(self, seed: int = SEED, size: int = IMG_SIZE,
                 config: GenBConfig | None = None):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.cfg = config or GenBConfig()

    # -- background -------------------------------------------------------

    def _skin_tone(self):
        """Vary skin DEPTH, not hue.

        Sampling B, G and R independently lets blue exceed red and produces
        green or blue skin. Human skin tone varies mainly in lightness and
        slightly in warmth, and always keeps R > G > B. So we draw one lightness
        scalar, apply a fixed skin chromaticity, and jitter the warmth a little.
        """
        lo = np.array(self.cfg.skin_tone_lo, float)
        hi = np.array(self.cfg.skin_tone_hi, float)
        t = self.rng.random()                       # one depth parameter
        tone = lo + t * (hi - lo)                   # stays ordered B < G < R
        warmth = self.rng.normal(0, self.cfg.skin_warmth_jitter)
        tone = tone + np.array([-warmth, 0.0, warmth])
        return np.clip(tone, 40, 245)

    def _tissue_palette(self, tone):
        """Derive the tissue colours FROM the sampled skin tone.

        A fixed absolute palette gives whatever contrast happens to fall out of
        the skin tone draw, and in practice gave twice the clinical value. Wound
        tissue is perilesional skin shifted toward red and saturation by a
        measured amount, so the palette is built from the tone rather than
        beside it. Necrosis stays genuinely dark: that is a real tissue
        property, not a contrast artefact.
        """
        cfg = self.cfg
        bgr = np.clip(tone, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)

        def to_bgr(hh, ss, vv):
            arr = np.array([[[hh % 180, np.clip(ss, 0, 255),
                              np.clip(vv, 0, 255)]]], np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0, 0].astype(float)

        d = cfg.lesion_sat_delta + self.rng.normal(0, cfg.lesion_sat_jitter)
        # granulation: perilesional hue pulled toward red, saturation raised
        gran = to_bgr(h * 0.45, s + d, v + cfg.lesion_val_delta)
        # slough: yellower, similar saturation lift
        slough = to_bgr(22 + self.rng.normal(0, 3), s + d * 0.85, v + 6)
        # necrosis: dark, genuinely low value
        necro = to_bgr(h * 0.6, s * 0.7 + 20, max(28.0, v * 0.22))
        return gran, slough, necro

    def _background(self):
        """Textured skin, or generator A's near-uniform drape when ablated."""
        S = self.size
        tone = self._skin_tone()
        self._last_tone = tone
        self._palette = self._tissue_palette(tone)
        img = np.full((S, S, 3), tone, dtype=np.float64)

        if not self.cfg.textured_background:
            # ABLATION: generator A's assumption. Near-uniform with only the
            # mild radial illumination gradient and fine sensor noise.
            ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
            dist = np.sqrt((ii - S // 2) ** 2 + (jj - S // 2) ** 2)
            factor = np.clip(1.0 - dist / (S * 0.9), 0.85, 1.0)[..., None]
            img *= factor
            img += self.rng.normal(0, 2.0, img.shape)
            return np.clip(img, 0, 255).astype(np.uint8)

        # Low-frequency mottling.
        #
        # This must be predominantly LUMINANCE variation. Drawing independent
        # noise per colour channel decorrelates the chroma and produces rainbow
        # pastel blotches, which is not skin: it makes the background more
        # saturated than the lesion and scrambles the colour-referenced methods
        # in a way no clinical image does. Skin mottles in brightness, with only
        # slight chroma drift, so one shared luminance field carries most of the
        # amplitude and a weak per-channel field supplies the rest.
        sc = self.cfg.mottle_scale
        lum = cv2.resize(self.rng.normal(0, 1, (sc, sc)), (S, S),
                         interpolation=cv2.INTER_CUBIC)
        chroma = cv2.resize(self.rng.normal(0, 1, (sc, sc, 3)), (S, S),
                            interpolation=cv2.INTER_CUBIC)
        img += lum[..., None] * self.cfg.mottle_amplitude
        img += chroma * self.cfg.mottle_amplitude * self.cfg.mottle_chroma_ratio

        # directional lighting gradient, random direction
        th = self.rng.random() * 2 * np.pi
        ii, jj = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S),
                             indexing="ij")
        ramp = (np.cos(th) * jj + np.sin(th) * ii)
        img *= (1.0 + self.cfg.gradient_strength * ramp)[..., None]

        # fine sensor noise
        # Fine sensor noise.
        #
        # Drawn independently per channel this is CHROMA noise, and the profile
        # descriptor showed it put generator B's a* and b* background texture at
        # roughly six times the clinical value (log ratio +1.75 and +1.81 at the
        # 9 px scale) while its luminance texture matched clinical almost
        # exactly. Real sensor noise in skin photography is predominantly
        # luminance. Most of the amplitude therefore goes into a shared
        # luminance term, with a small independent chroma residual.
        S = img.shape[0]
        lum_noise = self.rng.normal(0, self.cfg.fine_noise_sigma, (S, S))
        img += lum_noise[..., None]
        img += self.rng.normal(0, self.cfg.fine_noise_sigma
                               * self.cfg.fine_noise_chroma_ratio, img.shape)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # hair strokes
        n = int(self.rng.integers(*self.cfg.n_hairs))
        for _ in range(n):
            x0, y0 = self.rng.integers(0, S, 2)
            ang = self.rng.random() * 2 * np.pi
            ln = int(self.rng.uniform(*self.cfg.hair_len_frac) * S)
            curve = self.rng.normal(0, 0.25)
            pts = []
            for t in np.linspace(0, 1, 14):
                a = ang + curve * t
                pts.append([int(x0 + ln * t * np.cos(a)),
                            int(y0 + ln * t * np.sin(a))])
            pts = np.array(pts, np.int32)
            shade = int(max(0, self.rng.integers(0, self.cfg.hair_darkness)))
            cv2.polylines(img, [pts], False, (shade, shade, shade),
                          int(self.rng.integers(1, 3)), cv2.LINE_AA)
        return img

    def _add_clutter(self, img):
        """Gauze and instrument-like regions, in a minority of frames.

        Drawn BEFORE the lesion and never written into the ground-truth mask,
        so clutter is background that a segmenter must reject.
        """
        S = self.size
        if self.rng.random() > self.cfg.clutter_prob:
            return img
        for _ in range(int(self.rng.integers(1, 3))):
            if self.rng.random() < 0.5:
                # gauze: pale soft-edged quadrilateral
                cx, cy = self.rng.integers(0, S, 2)
                w, h = self.rng.integers(S // 8, S // 3, 2)
                box = np.array([[cx - w // 2, cy - h // 2], [cx + w // 2, cy - h // 2],
                                [cx + w // 2, cy + h // 2], [cx - w // 2, cy + h // 2]])
                box = box + self.rng.integers(-15, 15, box.shape)
                layer = np.zeros((S, S), np.uint8)
                cv2.fillPoly(layer, [box.astype(np.int32)], 255)
                layer = cv2.GaussianBlur(layer, (0, 0), 3)
                a = (layer.astype(np.float32) / 255.0)[..., None]
                gauze = np.full_like(img, (225, 230, 235))
                img = (a * gauze + (1 - a) * img).astype(np.uint8)
            else:
                # instrument: bright metallic bar
                p1 = self.rng.integers(0, S, 2)
                p2 = p1 + self.rng.integers(-S // 2, S // 2, 2)
                cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2),
                         (200, 200, 205), int(self.rng.integers(6, 16)),
                         cv2.LINE_AA)
        return img

    # -- lesion -----------------------------------------------------------

    def _lesion_mask(self, grade):
        """Shape, size and placement. Size is independent of grade by design."""
        S = self.size
        cfg = self.cfg

        if cfg.grade_independent_size:
            # log-uniform over the measured clinical range
            lo, hi = np.log(cfg.lesion_frac_lo), np.log(cfg.lesion_frac_hi)
            frac = float(np.exp(self.rng.uniform(lo, hi)))
        else:
            # ABLATION: generator A's assumption, size carries the grade label
            frac = GEN_A_GRADE_SIZE[grade]

        area = frac * S * S
        r = np.sqrt(area / np.pi)

        if cfg.off_centre_placement:
            span = cfg.placement_span * S
            cx = S / 2 + self.rng.uniform(-span / 2, span / 2)
            cy = S / 2 + self.rng.uniform(-span / 2, span / 2)
        else:
            cx = cy = S / 2   # ABLATION: generator A's assumption
        cx = float(np.clip(cx, r + 4, S - r - 4))
        cy = float(np.clip(cy, r + 4, S - r - 4))

        mask = np.zeros((S, S), np.uint8)
        if cfg.irregular_shape:
            irr = GRADE_SPEC[grade]["irregularity"]
            n = cfg.n_polygon_pts
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radii = r * (1 + self.rng.uniform(-irr, irr, n))
            # smooth the radial perturbation so the outline is a wound, not a star
            k = np.array([0.25, 0.5, 0.25])
            radii = np.convolve(np.r_[radii[-1], radii, radii[0]], k, "valid")
            radii *= r / radii.mean()
            pts = np.stack([cx + radii * np.cos(ang),
                            cy + radii * np.sin(ang)], axis=1).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            # ABLATION: generator A's assumption, an ellipse
            cv2.ellipse(mask, (int(cx), int(cy)), (int(r), int(r * 0.94)),
                        float(self.rng.uniform(0, 180)), 0, 360, 255, -1)
        return (mask > 0).astype(np.uint8)

    def _paint_lesion(self, img, mask, grade):
        """Tissue composition carries the grade."""
        gran, slough, necro = GRADE_SPEC[grade]["mix"]
        S = self.size
        # Palette derived from this image's own skin tone, so the lesion sits at
        # the measured clinical contrast above its own surround.
        c_gran, c_slough, c_necro = getattr(
            self, "_palette", (GRANULATION, SLOUGH, NECROSIS))

        # base granulation
        base = np.full_like(img, c_gran, dtype=np.float64)
        base += self.rng.normal(0, 14, base.shape)
        out = img.copy()
        out[mask > 0] = np.clip(base, 0, 255).astype(np.uint8)[mask > 0]

        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return out
        cy, cx = ys.mean(), xs.mean()
        r_eq = np.sqrt(mask.sum() / np.pi)

        def blob_patch(target_frac, colour, spread, noise):
            """Paint sub-regions covering roughly target_frac of the lesion."""
            if target_frac <= 0:
                return
            painted = np.zeros((S, S), np.uint8)
            tries = 0
            want = target_frac * mask.sum()
            while painted.sum() < want and tries < 40:
                tries += 1
                bx = cx + self.rng.normal(0, r_eq * spread)
                by = cy + self.rng.normal(0, r_eq * spread)
                br = max(3, int(r_eq * self.rng.uniform(0.18, 0.45)))
                cv2.ellipse(painted, (int(bx), int(by)),
                            (br, int(br * self.rng.uniform(0.6, 1.4))),
                            float(self.rng.uniform(0, 180)), 0, 360, 255, -1)
                painted = cv2.bitwise_and(painted, mask * 255)
            col = np.full_like(img, colour, dtype=np.float64)
            col += self.rng.normal(0, noise, col.shape)
            out[painted > 0] = np.clip(col, 0, 255).astype(np.uint8)[painted > 0]

        blob_patch(slough, c_slough, 0.45, 12)
        blob_patch(necro, c_necro, 0.30, 8)

        # specular highlights over moist tissue
        glint = (self.rng.random((S, S)) < self.cfg.specular_prob) & (mask > 0)
        out[glint] = np.clip(out[glint].astype(np.int16)
                             + self.cfg.specular_boost, 0, 255).astype(np.uint8)
        return out

    # -- assembly ---------------------------------------------------------

    def one(self, grade):
        img = self._background()
        if self.cfg.clutter:
            img = self._add_clutter(img)
        mask = self._lesion_mask(grade)
        img = self._paint_lesion(img, mask, grade)
        return img, mask

    def generate(self, n_per_class: int = 15):
        out = []
        for grade in GRADES:
            for i in range(1, n_per_class + 1):
                img, mask = self.one(grade)
                out.append((grade, i, img, mask))
        return out


# --------------------------------------------------------------------------
# ablations (manuscript 5.3)
# --------------------------------------------------------------------------

ABLATIONS = {
    "none": {},
    "grade_independent_size": {"grade_independent_size": False},
    "off_centre_placement": {"off_centre_placement": False},
    "clutter": {"clutter": False},
    "irregular_shape": {"irregular_shape": False},
    "textured_background": {"textured_background": False},
}


def build(seed: int = SEED, n_per_class: int = 15, ablation: str = "none"):
    """Returns (images, masks, grades) for one generator B variant.

    `ablation` names the feature REVERTED to generator A's assumption.
    """
    if ablation not in ABLATIONS:
        raise KeyError(f"unknown ablation {ablation!r}")
    cfg = replace(GenBConfig(), **ABLATIONS[ablation])
    recs = GeneratorB(seed=seed, config=cfg).generate(n_per_class)
    return ([r[2] for r in recs], [r[3] for r in recs], [r[0] for r in recs])
