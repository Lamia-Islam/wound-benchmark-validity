# Pre-registration: label-free per-image competence score

Written and committed before any score was computed. Branch claude/novelty-competence. The JIIM paper in manuscript_v6/ is not changed.

## Score
- **Features** (per image, computed on the image as the methods see it, using the perimage32 loaders):
  - colour statistics in RGB and HSV for each channel: mean, standard deviation, and the 5th, 50th and 95th percentiles;
  - share of near-black pixels (every channel at most 8, as in the paper);
  - local texture: `woundbench.imagestats.texture_of_image`, whole image;
  - image height, width and aspect ratio;
  - edge density: share of Canny edge pixels.
  
  An ImageNet ResNet-50 embedding is used as a secondary feature set only if the torchvision weights download in under 2 minutes. If they do not, it is skipped and that is reported.
- **Fit:** only on FUSeg training images 0 to 392, the models' training images (`rebuild/results/v7/fuseg_train_index.json`), without the training images removed as duplicates of evaluation images. Features are standardised with a scaler fitted on these images.
- **Classes:** terciles of the lesion fraction of the training mask (3 classes). Each class is a Gaussian with diagonal covariance; each variance is shrunk towards the pooled variance (shrinkage weight 0.1) and floored at 1e-3.
- **Score:** the log of the sum over classes of (1/3) times the class density (log-sum-exp, uniform prior).
- **Threshold:** the 5th percentile of the scores of FUSeg training images 400 to 549 (held out from the fit, duplicates removed). No threshold is ever chosen on evaluation images.

## Evaluation data
- **Sets:** medetec, fuseg (validation reference), azh (test) and azh_train_slice (control).
- **CO2Wounds-V2 is dropped:** its images are not in the public repository and are not on disk.
- Image order follows `rebuild/results/perimage32/<set>.json`, with `rebuild/results/exclusions_v6.json` applied.
- **Method scores:** `rebuild/scripts/perimage_v7.py` (groups U, F, K, M19, and seed averages).
- **Units for bootstrap:** as in perimage_v7 (photographs for Medetec and the AZH sets, images otherwise).

## H1: failure prediction (primary)
- **Outcome:** per image with a non-empty reference, y is the share of the individual pretrained seed models (all trainings of the 18 architectures that exist) with Dice below 0.1. A "majority miss" is y of at least 0.5.
- **Statistics:**
  - Spearman(−score, y). A low score means low competence, so the sign is flipped so that positive means "low score, more misses".
  - AUROC of −score for majority miss.
  - Both pooled over the four sets and per set.
- **Intervals:** 95% percentile, B = 2000, numpy default_rng(2026). Units are resampled within each set (pooled: within every set independently).
- **Baselines:** black-pixel share (higher means more misses) and lesion fraction (lower means more misses), with the same statistics.
- **Partial Spearman** of −score with y controlling for black share: the Pearson correlation of the rank residuals after regressing ranks on the rank of black share. Pooled, with the same bootstrap.
- **Support:** pooled AUROC lower bound above 0.6 AND the pooled partial Spearman interval excludes 0.
- **Caution:** if no set has at least 5 majority misses, the AUROC for that set is not reported. If the pooled count is below 10, H1 is reported as not testable.

## H2: competence-filtered agreement (primary)
- **Statistic:** M19 wound-pair agreement, the mean Spearman correlation of per-method mean Dice over the pairs of the available wound datasets (medetec, fuseg, azh; 3 pairs, because CO2Wounds-V2 is dropped). Computed with only images whose score is at least the threshold, and with all kept images.
- **Change** = filtered minus all.
- **Interval:** method bootstrap, stats_v5 design. Methods are resampled with replacement within M19 (`np.random.default_rng(0)`, B = 2000, the same draws for both conditions), and the paired change is taken per draw.
- **Random-removal control:** in each of 1000 draws (default_rng(2027)), remove from each dataset the same number of images as the filter removes, chosen at random; recompute the change. Report the 95th percentile of the control changes.
- **Support:** change interval lower bound above 0 AND observed change above the 95th percentile of the control.
- If the filter removes no image from any wound dataset, H2 is not testable.

## H3: benchmark shift (exploratory, report only)
- **Synthetic sets:** generator B at amplitudes 11, 28 and 40, and the four matched-texture benchmarks, each at seeds 2024, 7 and 101. They are regenerated with the repository's generator code exactly as `rebuild/scripts/kaggle_export.py` builds them.
- **Per synthetic set:** the mean competence score, and the Fréchet distance of standardised features (diagonal-covariance form, same scaler) to each wound set.
- **Statistics:** Spearman, across synthetic sets, of (a) the mean score and (b) the mean Fréchet distance to the wound sets with benchmark-to-wound agreement (`bench_v7.json`, rel/<set>/<amp>/mean|bench for the generator-B sets; matched benchmarks from `matched` where present). Separately for M19 and U+M19.
- No decision rule.

## Reporting
- All numbers go to `novelty/results/*.json` by script.
- `novelty/RESULTS_P1.md` is generated from the JSON (no hand-typed numbers, no em dashes).
- **Decision:** GO to a method paper only if H1 or H2 is supported by the rules above. Otherwise submit the JIIM paper unchanged.
