"""Generator A: lesion composited onto uniform background.

Manuscript Batch 2, section 4.1:

    Generator A synthesises 512 x 512 images in three severity grades. A centred
    ellipse (semi-axes 90 x 85 px for the healthy grade) is composited onto a
    near-uniform surgical drape with a mild radial illumination gradient.
    Sutures are drawn for the two less severe grades. Grade is encoded through
    lesion size, tissue colour composition, and boundary irregularity.

This is a port of `generate_wounds.py` from the Track A repository
(github.com/Lamia-Islam/Surgical-Vision), verified against the manuscript
description: IMG_SIZE 512, seed 2024, healthy semi-axes rx=90 ry=85, radial
illumination gradient, sutures, three grades, 15 per class.

THE ONE CHANGE: the original paints from internal masks but writes only JPEGs.
The manuscript states both generators emit exact ground-truth masks. This port
returns the mask alongside the image. The drawing calls, the palette, the RNG
and its draw order are untouched, so the images are byte-identical to the
originals.

The ground-truth mask is the wound body mask (the region passed to
paint_tissue for the main lesion), NOT the erythema ring or the satellite
dehiscence spots in the critical grade. Those are painted outside the lesion and
are background for segmentation purposes.
"""

import cv2
import numpy as np

IMG_SIZE = 512
CENTER = (IMG_SIZE // 2, IMG_SIZE // 2)
N_PER_CLASS = 15
SEED = 2024

# Tissue colour palette (BGR), verbatim from the Track A source.
SKIN_BASE = (140, 160, 190)
HEALTHY_PINK = (100, 130, 210)
PALE_TISSUE = (160, 170, 200)
NECROTIC_DARK = (30, 35, 55)
BLOOD_RED = (40, 50, 160)
FIBRIN_YELLOW = (80, 190, 220)
SUTURE_COLOR = (20, 20, 20)

GRADES = ("healthy", "concerning", "critical")


class GeneratorA:
    """Deterministic given the seed. One instance = one dataset."""

    def __init__(self, seed: int = SEED, size: int = IMG_SIZE):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.center = (size // 2, size // 2)

    # -- canvas and primitives (ported verbatim) --------------------------

    def make_canvas(self):
        S = self.size
        img = np.full((S, S, 3), SKIN_BASE, dtype=np.uint8)
        # Vectorised form of the original's pixel loop. Same arithmetic.
        ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        dist = np.sqrt((ii - S // 2) ** 2 + (jj - S // 2) ** 2)
        factor = np.clip(1.0 - dist / (S * 0.9), 0.85, 1.0)[..., None]
        img = np.clip(img.astype(np.float64) * factor, 0, 255).astype(np.uint8)
        noise = self.rng.integers(-6, 6, img.shape, dtype=np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def ellipse_mask(self, rx, ry, angle=0):
        m = np.zeros((self.size, self.size), np.uint8)
        cv2.ellipse(m, self.center, (rx, ry), angle, 0, 360, 255, -1)
        return m

    def irregular_mask(self, mean_radius, irregularity=0.3, n_pts=24):
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        radii = mean_radius * (1 + self.rng.uniform(-irregularity,
                                                    irregularity, n_pts))
        radii = np.clip(radii, 20, self.size // 2 - 20)
        pts = np.stack([self.center[0] + radii * np.cos(angles),
                        self.center[1] + radii * np.sin(angles)],
                       axis=1).astype(np.int32)
        m = np.zeros((self.size, self.size), np.uint8)
        cv2.fillPoly(m, [pts], 255)
        return m, pts

    def paint_tissue(self, img, mask, base_color, noise_range=18):
        col = np.full_like(img, base_color)
        noise = self.rng.integers(-noise_range, noise_range, img.shape,
                                  dtype=np.int16)
        col = np.clip(col.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out = img.copy()
        out[mask > 0] = col[mask > 0]
        return out

    def add_specular_glint(self, img, mask):
        glint = self.rng.random(img.shape[:2]) > 0.995
        g = (glint & (mask > 0))
        out = img.copy()
        out[g] = np.clip(out[g].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        return out

    def add_sutures(self, img, cx, cy, radius, n=6):
        out = img.copy()
        for k in range(n):
            th = 2 * np.pi * k / n
            x = int(cx + radius * np.cos(th))
            y = int(cy + radius * np.sin(th))
            dx = int(10 * np.cos(th + np.pi / 2))
            dy = int(10 * np.sin(th + np.pi / 2))
            cv2.line(out, (x - dx, y - dy), (x + dx, y + dy),
                     SUTURE_COLOR, 2, cv2.LINE_AA)
        return out

    # -- the three grades -------------------------------------------------

    def healthy(self):
        img = self.make_canvas()
        rx, ry = 90, 85
        mask = self.ellipse_mask(rx, ry)
        img = self.paint_tissue(img, mask, HEALTHY_PINK, noise_range=14)
        img = self.add_specular_glint(img, mask)
        cv2.ellipse(img, self.center, (rx, ry), 0, 0, 360, (60, 80, 140), 2,
                    cv2.LINE_AA)
        img = self.add_sutures(img, *self.center, rx + 8, n=8)
        return img, (mask > 0).astype(np.uint8)

    def concerning(self):
        img = self.make_canvas()
        mask, pts = self.irregular_mask(mean_radius=100, irregularity=0.22)
        img = self.paint_tissue(img, mask, HEALTHY_PINK, noise_range=20)
        for _ in range(self.rng.integers(1, 3)):
            px = self.center[0] + self.rng.integers(-50, 50)
            py = self.center[1] + self.rng.integers(-50, 50)
            pm = np.zeros((self.size, self.size), np.uint8)
            cv2.ellipse(pm, (int(px), int(py)),
                        (int(self.rng.integers(18, 35)),
                         int(self.rng.integers(18, 35))),
                        int(self.rng.integers(0, 180)), 0, 360, 255, -1)
            pm = cv2.bitwise_and(pm, mask)
            img = self.paint_tissue(img, pm, PALE_TISSUE, noise_range=10)
        img = self.add_specular_glint(img, mask)
        border = np.zeros_like(mask)
        cv2.polylines(border, [pts], True, 255, 4)
        blur = cv2.GaussianBlur(border, (11, 11), 4)
        alpha = blur.astype(np.float32) / 255.0
        dark = np.full_like(img, (50, 65, 110))
        for c in range(3):
            img[:, :, c] = (alpha * dark[:, :, c]
                            + (1 - alpha) * img[:, :, c]).astype(np.uint8)
        img = self.add_sutures(img, *self.center, 108, n=8)
        return img, (mask > 0).astype(np.uint8)

    def critical(self):
        img = self.make_canvas()
        mask, pts = self.irregular_mask(mean_radius=130, irregularity=0.48)
        img = self.paint_tissue(img, mask, HEALTHY_PINK, noise_range=16)
        nm = np.zeros((self.size, self.size), np.uint8)
        cv2.ellipse(nm, self.center,
                    (int(self.rng.integers(35, 55)),
                     int(self.rng.integers(35, 55))),
                    int(self.rng.integers(0, 180)), 0, 360, 255, -1)
        nm = cv2.bitwise_and(nm, mask)
        img = self.paint_tissue(img, nm, NECROTIC_DARK, noise_range=8)
        for _ in range(self.rng.integers(2, 5)):
            fx = self.center[0] + self.rng.integers(-70, 70)
            fy = self.center[1] + self.rng.integers(-70, 70)
            fm = np.zeros((self.size, self.size), np.uint8)
            cv2.ellipse(fm, (int(fx), int(fy)),
                        (int(self.rng.integers(12, 28)),
                         int(self.rng.integers(12, 28))),
                        int(self.rng.integers(0, 180)), 0, 360, 255, -1)
            fm = cv2.bitwise_and(fm, mask)
            img = self.paint_tissue(img, fm, FIBRIN_YELLOW, noise_range=12)
        ring = np.zeros((self.size, self.size), np.uint8)
        cv2.ellipse(ring, self.center, (140, 140), 0, 0, 360, 255, -1)
        cv2.ellipse(ring, self.center, (130, 130), 0, 0, 360, 0, -1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
        img = self.paint_tissue(img, ring, BLOOD_RED, noise_range=20)
        for _ in range(self.rng.integers(3, 6)):
            sx = self.center[0] + self.rng.integers(-160, 160)
            sy = self.center[1] + self.rng.integers(-160, 160)
            sr = int(self.rng.integers(8, 20))
            sm = np.zeros((self.size, self.size), np.uint8)
            cv2.circle(sm, (int(sx), int(sy)), sr, 255, -1)
            col = [NECROTIC_DARK, PALE_TISSUE,
                   FIBRIN_YELLOW][int(self.rng.integers(0, 3))]
            img = self.paint_tissue(img, sm, col, noise_range=10)
        img = self.add_specular_glint(img, mask)
        cv2.polylines(img, [pts], True, (15, 20, 40), 3, cv2.LINE_AA)
        return img, (mask > 0).astype(np.uint8)

    # -- dataset ----------------------------------------------------------

    def generate(self, n_per_class: int = N_PER_CLASS):
        """Yield (grade, index, image, mask) in the original draw order.

        Draw order matters: the original calls all healthy, then all concerning,
        then all critical, from one shared RNG stream.
        """
        out = []
        for fn, grade in ((self.healthy, "healthy"),
                          (self.concerning, "concerning"),
                          (self.critical, "critical")):
            for i in range(1, n_per_class + 1):
                img, mask = fn()
                out.append((grade, i, img, mask))
        return out


def build(seed: int = SEED, n_per_class: int = N_PER_CLASS):
    """Convenience: returns (images, masks, grades)."""
    recs = GeneratorA(seed=seed).generate(n_per_class)
    return ([r[2] for r in recs], [r[3] for r in recs], [r[0] for r in recs])
