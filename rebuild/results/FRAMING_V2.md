# The framing result, after the three objections

Supersedes FRAMING_LAW.md. That version was measured over 21 thresholding
algorithms and one small U-Net, on one seed, under one metric. All three have
now been fixed. The effect survived. Two things about how it was reported did
not.

---

## 1. What changed

| | old | new |
|---|---|---|
| method pool | 22, of which 21 classical | 32, including ten modern architectures |
| seeds | 1 | 4 |
| metric | Dice | Dice and normalised surface Dice at 2 px |
| statistic | Spearman of agreement against contrast | OLS slope of agreement against contrast |

The statistic had to change. A Spearman over seven contrast levels saturates at
+/-1 as soon as the underlying relationship is monotone, and at n = 32 it became
monotone almost everywhere: five of the eight crop levels reported exactly
+1.000 under both metrics, which is not a measurement, it is a ceiling. The
agreement values behind that +1.000 run from -0.33 to +0.36, so there was plenty
of signal, and the rank statistic was discarding it.

The slope of agreement on measured contrast, per 10 saturation units, is
continuous and does not saturate. Everything below uses it.

## 2. The curve, under both metrics

Slope of rank agreement against contrast. Positive means the reference rewards
high lesion-to-surround contrast.

| lesion fraction | Dice | sd over seeds | NSD | sd over seeds |
|---|---|---|---|---|
| 0.0077 | +0.103 | 0.003 | +0.107 | 0.005 |
| 0.0101 | +0.103 | 0.003 | +0.107 | 0.005 |
| 0.0200 | +0.098 | 0.003 | +0.105 | 0.006 |
| 0.0350 | +0.100 | 0.003 | +0.106 | 0.005 |
| 0.0549 | +0.089 | 0.003 | +0.093 | 0.003 |
| 0.0798 | +0.047 | 0.006 | +0.056 | 0.004 |
| 0.1099 | +0.001 | 0.008 | +0.018 | 0.007 |
| 0.1597 | -0.051 | 0.007 | -0.039 | 0.006 |

Monotonicity against lesion fraction, per seed:

| metric | seed 2024 | seed 7 | seed 101 | seed 555 |
|---|---|---|---|---|
| Dice | -0.952 (p 0.0010) | -0.952 (p 0.0010) | -0.976 (p 0.0002) | -0.952 (p 0.0010) |
| NSD | -0.881 (p 0.0067) | -1.000 (p <0.0001) | -1.000 (p <0.0001) | -0.952 (p 0.0010) |

Eight of eight seed-metric combinations are monotonic decreasing at p < 0.01.
The seed standard deviation never exceeds 0.008 on a range of 0.154. The sign
change sits between lesion fraction 0.110 and 0.160 under both metrics.

**All three objections are answered.** The effect is not an artifact of ranking
thresholding algorithms, because it holds over a pool where ten modern
architectures are the top performers. It is not a single-seed accident, because
four seeds agree to within 0.008. And it is not the Dice region-size bias, which
was the most dangerous objection, because a boundary-sensitive metric gives the
same curve with the same sign change.

## 3. The held-out placements, honestly

Curve built entirely from cropped FUSeg. Medetec and AZH took no part in it.

| metric | dataset | lesion fraction | predicted | measured | residual | as % of curve range |
|---|---|---|---|---|---|---|
| Dice | FUSeg | 0.0109 | +0.103 | +0.107 | +0.004 | 2.4% |
| Dice | AZH | 0.0285 | +0.099 | +0.118 | +0.019 | 12.4% |
| Dice | Medetec | 0.1024 | +0.012 | +0.038 | +0.026 | 16.6% |
| NSD | FUSeg | 0.0109 | +0.107 | +0.121 | +0.015 | 10.0% |
| NSD | AZH | 0.0285 | +0.105 | +0.114 | +0.008 | 5.5% |
| NSD | Medetec | 0.1024 | +0.027 | +0.056 | +0.028 | 19.4% |

FUSeg is not a check, the curve is made of its crops.

Medetec and AZH both land with the right sign and the right magnitude, with
residuals of 5% to 19% of the curve's range. That is a real out-of-sample
placement and it is worth reporting. It is not the 0.027-on-a-1.93-scale
bullseye the previous version claimed, and that earlier number should be treated
as an artifact of the saturating statistic rather than as a result.

Both held-out datasets sit slightly **above** the curve under both metrics. A
systematic offset in one direction across two independent datasets is more
likely to be a property of the cropping construction than noise, and it is not
explained here.

## 4. What the result is now

The verdict a clinical reference gives about synthetic benchmark design is a
monotonic decreasing function of how much of the frame its lesions occupy. The
relationship holds across four generator seeds, two metrics and a method pool
that spans classical algorithms and modern architectures. References with small
lesions reward high lesion-to-surround contrast; references with large lesions
reward low contrast; the crossover sits somewhere above lesion fraction 0.11.

What it is not:

- It is not a portable constant. The crossover moved from the 0.055 to 0.080
  interval to the 0.110 to 0.160 interval when the method pool and the statistic
  changed. The location is a property of the pool as well as the references, so
  it should never be quoted as a threshold value for other people to apply.
- It is not a prediction rule. A 12% to 19% residual on a held-out dataset
  supports the relationship, not a calibrated forecast.

The defensible sentence is that benchmark-design conclusions drawn against a
single clinical reference are statements about that reference's framing, and
that the direction of the bias is predictable from a property of the reference
that anyone can measure in one line of code.

## 5. Caveats

- The crops use the ground-truth mask centroid and area, which is oracle
  localisation. These are instruments for varying one property, not
  preprocessing pipelines anyone could run.
- Eight crop levels from 120 FUSeg frames. The three highest levels discard most
  of the surrounding limb, so the top of the curve is further from clinical
  practice than the bottom.
- The contrast sweep has seven levels. The slope is fitted on seven points.
- Both held-out datasets sit above the curve, unexplained.
- Two of the three wound references are from the same clinic, and all of this is
  chronic wound photography. The cross-modality work shows the verdict statistic
  is not interpretable for endoscopy references, so this result is scoped to
  within-modality use.
