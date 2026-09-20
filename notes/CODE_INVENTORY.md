# Code inventory: what was lost and what has to be rebuilt

Derived from the recovered manuscript (Batches 1, 1RevA, 2, 3, 4). Everything
below lived in the August cloud container at `/home/claude/build` and no longer
exists. Nothing in the three GitHub repos covers any of it.

Rebuild difficulty is graded:

- **A** — fully specified in the manuscript, rebuild is transcription
- **B** — specified but with free parameters the manuscript does not pin down
- **C** — underspecified, will need judgement calls that may change the numbers

---

## 1. Synthetic generators

### Generator A — difficulty A
512 × 512, three severity grades. Centred ellipse (semi-axes 90 × 85 px for the
healthy grade) composited onto near-uniform surgical drape with mild radial
illumination gradient. Sutures drawn for the two less severe grades. Grade
encoded through lesion size, tissue colour composition, boundary irregularity.

**Partial recovery available.** `generate_wounds.py` from the Track A repo
(`github.com/Lamia-Islam/Surgical-Vision`) is the ancestor of Generator A and is
deterministic under `np.random.default_rng(2024)`. It produces 45 images
(15 per class). Whether Generator A as used in the benchmark is identical to
this script or an extension of it is NOT recorded. **Check first.**

### Generator B — difficulty B
Lesion embedded in textured skin. Design targets recovered in full (Batch 2,
§4.2 table): log-uniform lesion fraction 0.005–0.12 sampled independently of
grade; uniform placement over central 60% of frame; textured skin background
with low-frequency mottling, fine noise, directional lighting gradient and hair;
per-image skin tone variation over a plausible BGR range; gauze and
instrument-like clutter in a minority of frames; radially perturbed polygon
lesion shape; specular highlights over moist tissue. Grade defined by tissue
composition (granulation, slough, necrosis proportions) and boundary
irregularity, never by size.

**What is missing:** exact amplitudes, frequencies and kernel sizes for the
mottling, noise, gradient and hair strokes. Distribution parameters for skin
tone. Clutter frequency ("a minority of frames"). Number of images generated.
The seed.

### Ablation variants of Generator B — difficulty B
Five, each reverting one feature to Generator A's assumption with everything
else held at B settings and the seed fixed:
1. grade-independent lesion size → grade-dependent
2. off-centre placement → centred
3. clutter → none
4. irregular lesion shape → ellipse
5. **textured background → uniform** (the load-bearing one)

---

## 2. The 22 segmentation methods

All take BGR in, return a binary mask, no per-image tuning.

### Background-referenced (5) — difficulty A
Background model from four 20 × 20 image corners.
1. grayscale adaptive threshold, Gaussian variant
2. grayscale adaptive threshold, mean variant
3. global Otsu on grayscale
4. Euclidean distance from median corner colour, BGR
5. Euclidean distance from median corner colour, HSV

### Colour-referenced (4) — difficulty A
6. Otsu on CIELAB a\*
7. 92nd percentile of a\*
8. redness ratio R/(G+B)
9. Otsu on HSV saturation

### Clustering (4) — difficulty A
10. k-means in CIELAB, k = 3, modal corner cluster as background
11. k-means in CIELAB, k = 5, modal corner cluster as background
12. k-means, k = 4, retain cluster with highest mean a\*
13. pyramid mean-shift filtering then Otsu on a\*

### Region and contour (5) — difficulty B
14. marker-based watershed seeded from a\* extremes
15. GrabCut initialised from central rectangle covering 70% of frame
16. flood-fill region growing from the a\* maximum
17. two-phase Chan–Vese style region competition on a\*, no edge term
18. Canny edge detection with morphological closure and contour filling

Free parameters not recorded: watershed marker percentiles, GrabCut iteration
count, flood-fill tolerance, Chan–Vese iteration count and lambda, Canny
hysteresis thresholds.

### Learned per-pixel classifiers (3) — difficulty A for features, B for fit
Fourteen features: CIELAB, HSV and BGR channels (9); local mean and SD of a\*
and of grayscale over a 9 × 9 window (4); a\* and saturation centred on each
image's own median (2). NOTE: that sums to 15, not 14. **The manuscript's
feature count and its enumeration disagree. Resolve on rebuild.**
Pixel coordinates deliberately excluded.

19. logistic regression, all 14 features
20. gradient boosting, all 14 features
21. logistic regression, 9 colour features only

Training: 264,292 pixels sampled from 250 FUSeg training images, balanced
lesion/non-lesion, ~1 s on one CPU core. Decision threshold 0.9, selected by
sweep on FUSeg training images 250–400.

Free parameters not recorded: gradient boosting hyperparameters, logistic
regression regularisation, sampling RNG seed.

### Convolutional network (1) — difficulty A
22. Three-level U-Net, 16/32/64 channels, **117,681 parameters**. Trained from
random init on 393 FUSeg training images at 128 × 128, 30 epochs, combined
Dice + BCE loss, horizontal flip augmentation, Adam at lr 3 × 10⁻³. 470 s on one
CPU core.

The parameter count is exact enough to verify the architecture rebuild: if the
reconstructed model does not have 117,681 parameters, the architecture is wrong.

### Shared post-processing — difficulty B
Morphological closing then opening, elliptical kernel; removal of connected
components below 0.05% of frame area. Learned methods additionally get Gaussian
smoothing of the probability field before thresholding, instead of morphology on
the binary mask.

Kernel size and Gaussian sigma not recorded.

---

## 3. Evaluation harness — difficulty A

- Dice per image, averaged. Empty prediction + empty reference = 1.
- Spearman rank correlation between benchmarks.
- Permutation test, 20,000 permutations, two-sided.
- Bootstrap 95% CI on mean Dice, 5,000 resamples.
- Output file was `results/rank_validity_n22.json` — needed for Table 4 and
  Supplementary Table S1, both currently incomplete in the draft.

---

## 4. Competence signals — difficulty C

Seventeen label-free signals across four families. **Only four are named in the
recovered manuscript** (Table 7, the best of each family):

- model confidence: fraction of decisive pixels
- ensemble disagreement: mean IoU to learned mask
- geometric plausibility: predicted area fraction
- TTA instability: mask area coefficient of variation

The other thirteen are not recorded anywhere in the recovered text. This is the
single largest gap. Section 6.5 and Table 8 (cross-source transfer, the
ρ = 0.948 vs 0.316 result) depend on all seventeen.

Also needed: the learned combination model over all seventeen, five-fold
within-Medetec CV, and cross-source train/test in both directions.

---

## 5. Gate analysis — difficulty B

- ABCDE scorer: asymmetry (contour solidity), border (Laplacian variance on
  boundary strip), colour (tissue-zone ratios), diameter (min enclosing circle),
  evolution (inter-frame area change). Each 0/1/2. **Partially recoverable from
  the Track A repo's `abcde_analyzer.py`.**
- Verdict mapping: 0–1 PASS/full/1.00, 2 CAUTION/reduced/0.50, ≥3 FAIL/hold/0.00
- Unsafe-optimistic rate: full autonomy granted where Dice < 0.5
- Conservative rate: reduced or hold where Dice ≥ 0.5
- Oracle gate: full autonomy iff Dice ≥ 0.5
- Threshold recalibration on held-out synthetic seeds, cross-seed mean over
  eight unseen seeds
- End-to-end latency measurement (reported 40.4 ms at 512 × 512, single core;
  gate evaluation 1.18 ms)

---

## 6. Data — difficulty A, no rebuild needed

Both clinical datasets are public at `github.com/uwm-bigdata/wound-segmentation`.

| | FUSeg | Medetec |
|---|---|---|
| images used | 1,010 (400 train / 150 val / 120 test) | 152 |
| resolution | 512 × 512 | 224 × 224 |
| median lesion fraction | 0.0068 | 0.098 |
| mask encoding | 0/255 | 0/1 |

Binarise on any non-zero value. A fixed mid-range threshold silently empties the
Medetec masks and yields Dice exactly zero for every method. This bug was hit
and fixed once already.

---

## Rebuild order

1. **Verify Generator A against the Track A repo** before writing anything. If
   it matches, one generator is free.
2. Download both clinical datasets, confirm the counts and median lesion
   fractions above. These are cheap checks that validate the recovered numbers.
3. Build the 22 methods. Difficulty A ones first (16 of 22).
4. Build the evaluation harness and regenerate Table 3. **This is the
   go/no-go point:** if generator A vs generator B does not come out near
   ρ = −0.526 and Medetec vs FUSeg near ρ = 0.560, something is wrong and the
   paper's central claim does not stand as written.
5. Generator B and the five ablations. Regenerate Table 5.
6. Gate analysis. Regenerate Table 6.
7. Competence signals. This is the hardest and least specified; consider
   whether Section 6.5 must be in this paper at all, or whether it is a second
   paper.

---

## Honest assessment

Steps 1–5 are a real rebuild but a tractable one: the manuscript specifies most
of it well enough, and the Kaggle free tier (≈30 GPU-hours/week on a P100) is
far more than a 117k-parameter U-Net needs. Most of this runs on CPU anyway,
which is how it ran the first time.

Step 7 is the problem. Thirteen of seventeen competence signals are simply not
recorded. Any reconstruction will be a different set, so the ρ = 0.948 vs 0.316
numbers will not reproduce exactly. That result is contribution 4 of 5 in the
Introduction and occupies Section 6.5, Table 7, Table 8 and Figure 4.

Two options, to decide before starting:
- Reconstruct a seventeen-signal set from the four named plus standard choices,
  accept that the numbers will differ, and report what the rebuild actually
  produces.
- Cut the competence-signal section, drop to four contributions, and keep the
  paper focused on benchmark design. Shorter, tighter, and every number in it
  regenerable.

The second is probably the better paper regardless of the recovery situation.
