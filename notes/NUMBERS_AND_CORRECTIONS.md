# Reconciled numbers and recovered corrections

## Part 1 — Which numbers are final

Three different sets of rank correlations appear across the August conversation.
They are successive stages of the same work, not contradictions.

### Stage 1: six methods (superseded, underpowered)

| comparison | ρ |
|---|---|
| synthetic vs Medetec | −0.086 |
| synthetic vs FUSeg | 0.029 |
| Medetec vs FUSeg | 0.771 |

Six methods: HSV colour distance, k-means (Lab), BGR background subtraction,
adaptive threshold, learned, grayscale Otsu.

**Do not use.** Spearman at n = 6 requires |ρ| ≥ 0.83 for p < 0.05, so none of
these were significant. The manuscript itself says so (Batch 2 §3.8). This stage
is only worth mentioning as the motivation for expanding the pool.

### Stage 2: twenty-one methods (superseded)

An early abstract draft reported "twenty-one segmentation methods", Spearman
−0.481 (p = 0.028) between synthetic benchmarks and +0.557 (p = 0.0098) between
clinical datasets.

**Superseded.** The method pool grew to 22 and the numbers moved.

### Stage 3: twenty-two methods — THESE ARE THE FINAL NUMBERS

Consistent across Batch 1 (abstract and contributions), Batch 2 (notes for
Batch 3) and Batch 3 (Table 3). Permutation test, 20,000 permutations.

| comparison | ρ | p |
|---|---|---|
| generator A vs Medetec | −0.121 | 0.588 |
| generator A vs FUSeg | −0.466 | 0.033 |
| generator B vs Medetec | 0.439 | 0.040 |
| generator B vs FUSeg | 0.560 | 0.007 |
| **generator A vs generator B** | **−0.526** | **0.014** |
| **Medetec vs FUSeg** | **0.560** | **0.008** |

**One thing to check on rebuild:** generator B vs FUSeg and Medetec vs FUSeg are
both reported as ρ = 0.560 with different p values (0.007 and 0.008). Possible
with a permutation test on different rankings, but it is the kind of coincidence
a careful reviewer notices. Confirm both when regenerating Table 3.

### Other headline numbers (Stage 3, treat as final)

- Ablation: removing textured background costs 0.532 mean ρ and reverses sign
  (0.070 Medetec, −0.268 FUSeg). Next largest single effect 0.079.
- U-Net: 0.080 generator A, 0.680 generator B, 0.446 Medetec, 0.670 FUSeg.
  117,681 parameters, 470 s CPU training.
- Learned baseline at threshold 0.9: 0.452 FUSeg validation, 0.451 Medetec.
  Converge to within 0.001.
- Gate unsafe-optimistic: 0.61 → 0.62 FUSeg (recalibration does not help),
  0.34 → 0.27 Medetec.
- Synthetic grade accuracy 60.0% → 100.0% under recalibration; cross-seed mean
  99.7% (SD 0.8) over eight unseen seeds.
- Competence: within-Medetec 5-fold ρ = 0.948; cross-source ρ = 0.316.
- Latency: 40.4 ms end-to-end at 512 × 512 single core; gate eval 1.18 ms.

---

## Part 2 — Corrections logged in the lost FINDINGS.md

Recovered verbatim. These were deliberately kept in the repo's FINDINGS.md so
that a reviewer or collaborator could see what was corrected and why. They must
survive into the rebuild.

- **Dice 0.559 on Medetec** was presented as the learned segmenter's headline.
  It is inflated: Medetec 224×224 images are pre-cropped around the wound, with
  a median wound fraction of 0.098 against 0.0068 for uncropped FUSeg images, a
  14-fold difference. The corrected cross-source figure is 0.45.
- **k-means beating the HSV method on synthetic data** came from a scratch
  harness using ratio-based diameter scoring. Under the package they tie at
  60.0%.
- **A malformed `<name>` element in `package.xml`** was a display artifact. The
  original file was correct.
- **Dice of exactly 0.000 for all methods on Medetec** in a first run was a bug:
  Medetec masks are encoded 0/1 and FUSeg 0/255, and a fixed threshold of 127
  emptied the Medetec masks. Fixed by binarising on any nonzero value.
- The claim that excluding the evolution criterion explains the gap between the
  manuscript's 6.7%/64.4% and this repository's 2.2%/60.0% is an inference and
  has not been directly verified.

### Additional corrections named in the conversation but not in the FINDINGS list

- Three wrong venue recommendations were made and withdrawn: SIViP, MELBA, and
  an initial overstatement of JIIM's standing. MELBA was ultimately rejected in
  favour of JIIM because MELBA lacks Scopus indexing, which matters for lecturer
  and PhD applications.
- Underpowered n = 6 correlations were at one point presented as significant.
  Corrected by expanding to n = 22 and stating the power requirement explicitly.

---

## Part 3 — Confidence-as-competence detail (Section 6.5 supporting data)

Recovered from the lost FINDINGS.md. Useful because it partially fills the
competence-signal gap.

Per-image confidence versus per-image Dice, three statistics:

| statistic | Medetec Spearman | FUSeg Spearman |
|---|---|---|
| mean certainty | 0.337 | 0.423 |
| negated mean entropy | 0.261 | 0.378 |
| fraction decisive | 0.435 | 0.411 |

Expected calibration error: 0.125 FUSeg validation, 0.141 Medetec.

On out-of-distribution input the model becomes **more** confident: mean certainty
0.942 (SD 0.010) on synthetic surgical images against 0.902 Medetec and 0.910
FUSeg.

Gating on the least-confident quartile, threshold chosen on FUSeg validation and
applied to Medetec, catches 41–45% of failing images. Mean Dice of retained
images rises from 0.559 to 0.597.

That recovers **three** of the seventeen competence signals by name (mean
certainty, negated mean entropy, fraction decisive), plus the three named in
Table 7 from other families. Roughly six of seventeen are now identified. The
remaining eleven are still unrecorded.

---

## Part 4 — Generator A verification (done)

`generate_wounds.py` in `github.com/Lamia-Islam/Surgical-Vision`
(`surgical-vision-codes/surgical_vision/`) matches the manuscript's Generator A
description exactly:

- 512 × 512 (`IMG_SIZE = 512`)
- seed 2024 (`np.random.default_rng(2024)`)
- healthy grade ellipse semi-axes rx = 90, ry = 85
- radial illumination gradient on a near-uniform canvas
- sutures drawn for the less severe grades
- three grades, `N_PER_CLASS = 15`

**Generator A does not need rebuilding from scratch.** One modification is
required: the script paints from internal masks (`ellipse_mask`,
`irregular_mask`) but only writes JPEGs. The manuscript states both generators
emit exact ground-truth masks, so mask export must be added.

Open question: whether the benchmark used the 45-image version or a larger
regeneration. Not recorded.
