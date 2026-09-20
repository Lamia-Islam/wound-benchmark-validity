# The profile descriptor, and what it bought

Two results. The descriptor works where the single statistic failed, but only as
far as diagnosis. Using it as a design target does not work. It did find a real
defect, and fixing that defect measurably improved the benchmark.

---

## 1. The profile descriptor recovers across-family prediction

Local standard deviation of the background in three channels (L, a, b) at four
scales (9, 21, 45, 91 px at canonical 512), giving a 12-entry descriptor.
Distance to clinical is the mean absolute log-ratio, so it is scale-free and
weights every channel and scale equally.

Correlating distance-to-clinical against already-measured rank agreement, over
15 benchmarks spanning all three generator families:

| descriptor | Spearman vs agreement | p |
|---|---|---|
| single luminance statistic | +0.244 (wrong sign) | 0.401 |
| **12-entry profile distance** | **-0.626** (correct sign) | **0.020** |

Negative is the predicted direction: closer profile, higher agreement. The
single statistic could not order the three families at all.

No new method scoring was done for this test. Every agreement value was read
from results already on disk, so the descriptor was fitted to nothing.

## 2. It diagnosed a real defect in generator B

Per-entry deviation from clinical, log ratio, for each family's closest
configuration:

| entry | generator A | generator B | generator C |
|---|---|---|---|
| L@9 | -0.39 | **+0.05** | +0.50 |
| L@21 | -1.05 | **+0.03** | -0.07 |
| L@45 | -1.65 | **+0.08** | -0.64 |
| L@91 | -2.08 | **-0.09** | -1.05 |
| a@9 | +1.60 | **+1.75** | +0.06 |
| b@9 | +1.66 | **+1.81** | -6.08 |

Generator B matched clinical luminance texture at all four scales almost
exactly, and carried roughly six times the clinical chroma texture at every
level. Its texture knob moved luminance only, so no amount of sweeping could
have found this. The single-channel statistic could not see it by construction.

Cause: fine sensor noise was drawn independently per colour channel, which is
chroma noise. Real sensor noise in skin photography is predominantly luminance.

Fix: most of the amplitude now goes into a shared luminance term with a small
independent chroma residual. Chroma log-ratios fell from +1.75 and +1.81 to
+0.02 and +0.10. Profile distance at the default level fell from 1.04 to 0.22.

## 3. Fixing it helps only at clinical-range texture, and not overall

CORRECTED 2026-09-20. The first version of this section used three seeds for the
corrected generator against five for the uncorrected one and reported a clean
win rising to 99% of ceiling. Extending the corrected generator to the same five
seeds removes most of that. The three-seed result was favourable sampling, which
is the second time seed count has changed a conclusion in this project.

Five seeds per level, same clinical references, same seeds on both sides.

| texture level | profile distance | corrected | uncorrected | change | % of ceiling |
|---|---|---|---|---|---|
| 0 | 0.556 | 0.394 | 0.431 | -0.037 | 69% |
| 11 | 0.221 | 0.424 | 0.439 | -0.015 | 75% |
| 28 | 0.446 | **0.554** | 0.523 | **+0.031** | **98%** |
| 40 | 0.706 | 0.551 | 0.536 | +0.015 | 97% |

Seed-matched pairwise, all twenty cells:

| texture level | mean delta | seeds improved |
|---|---|---|
| 0 | -0.037 | 1/5 |
| 11 | -0.015 | 2/5 |
| 28 | +0.031 | 3/5 |
| 40 | +0.015 | 4/5 |
| **overall** | **-0.001** | **10/20** |
| **clinical range only (28, 40)** | **+0.023** | **7/10** |

So the correction is **not** a net improvement to the generator. It is worth
about +0.023 at clinical-range texture, where it matters, and it costs about
-0.026 at low texture, where the benchmark is dominated by the uniform
background failure anyway. Averaged over the whole sweep the two cancel.

The best configuration reaches 0.554 against a ceiling of 0.567, which is 98%.
The earlier 99% figure was three-seed noise and should not be used.

## 4. What the descriptor cannot do

Within the corrected generator, profile distance does not predict agreement at
all: over five seeds per level the Spearman is exactly 0.000, p = 1.000. The
minimum-distance configuration (level 11, distance 0.221) scores 0.424, while
the best-scoring one (level 28, distance 0.446) is twice as far from clinical.

So the descriptor is a **diagnostic**, not a design target. It answers "which
part of my benchmark is unlike clinical images" and it answered that correctly
and usefully here. It does not answer "how good will my benchmark be", and
minimising it is not a valid objective.

## 5. Honest summary of the arc

- A single luminance statistic looked like a portable specification and was not:
  three benchmarks at matched value scored 0.121, 0.434 and 0.189.
- The 12-entry profile restores across-family ordering (-0.626, p = 0.020) and
  localises the discrepancy to specific channels and scales.
- Used as a diagnostic it found a six-fold chroma error in generator B that had
  gone unnoticed through every previous experiment.
- Correcting that error is worth +0.023 at clinical-range texture over five
  matched seeds, and nothing on average across the whole sweep. Best benchmark
  moves from 94% to 98% of the clinical-to-clinical ceiling.
- Used as an objective it fails outright: within the corrected generator the
  distance-agreement correlation is 0.000.

The defensible claim is therefore about diagnosis and about a demonstrated
improvement, not about a specification. That is narrower than the claim this
project started with and it is supported by measurements that survived the test
designed to break them.

## 6. The ablation improved too

Re-running the design-property ablation on the corrected generator sharpens the
paper's central comparison.

| reverted property | corrected | uncorrected | manuscript |
|---|---|---|---|
| grade-independent lesion size | -0.043 | -0.154 | -0.073 |
| off-centre placement | -0.057 | -0.130 | -0.079 |
| clutter | -0.006 | -0.091 | -0.064 |
| irregular lesion shape | +0.006 | -0.045 | +0.020 |
| **textured background** | **-0.235** | -0.316 | -0.532 |

Texture now leads the next-largest property by **4.1x**, against 2.1x on the
uncorrected generator. The manuscript reported 6.7x. Every non-texture property
also moved closer to its reported value, and irregular shape recovered the
small positive sign the manuscript reported.

Removing the chroma defect did not weaken the finding, it cleaned it up: the
nuisance properties now contribute almost nothing and texture carries the
effect, which is the structure the paper claims.

## 7. Caveats

- The comparison is now seed-matched at five seeds per level on both sides, and
  that is what demoted the headline from a clean win to a texture-dependent one.
  Section 3 carries the corrected numbers; anything quoting 99% of ceiling or a
  uniform improvement predates this and is wrong.
- The ablation in Section 6 was run single-seed on the corrected generator and
  has not had the same treatment. Given that seed count has now overturned two
  conclusions in this project, the 4.1x ratio should be assumed provisional
  until it is repeated across seeds.
- The profile's channels, scales and distance metric are choices. Nothing was
  tuned against the agreement values, but alternatives were not explored.
- Generator C remains an outlier the descriptor orders correctly but does not
  explain; its b-channel deviation of -6.08 is extreme and reflects a genuinely
  near-constant chroma background rather than a subtle mismatch.
- All results still rest on two clinical datasets of one pathology.
- Every earlier result that used generator B was produced with the uncorrected
  chroma. The ablation table and the original dose-response should be re-run on
  the corrected generator before anything is written up.
