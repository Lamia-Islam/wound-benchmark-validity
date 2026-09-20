# Synthetic benchmark validity in medical image segmentation

Code and results for a study of whether a synthetic benchmark orders
segmentation methods the way clinical data does.

22 segmentation methods across 6 families, scored by Dice on synthetic
benchmarks from three independently constructed generator families and on five
clinical references.

## Clinical references

| reference | n scored | modality | median lesion fraction |
|---|---|---|---|
| Medetec | 152 | wound photography | 0.102 |
| FUSeg (validation) | 120 | wound photography | 0.011 |
| AZH | 150 | wound photography | 0.029 |
| Kvasir-SEG | 120 | colonoscopy | 0.106 |
| Kvasir-SEG sessile | 120 | colonoscopy | 0.043 |

Datasets are not redistributed here. `rebuild/data/` is gitignored; see
`rebuild/results/DATA_SOURCES.md` for where each one comes from.

## Layout

- `rebuild/woundbench/` library: methods, metrics, generators, image statistics
- `rebuild/scripts/` one script per experiment, each resumable
- `rebuild/results/` measured outputs, figures, and the findings write-ups
- `manuscript/` recovered v1 text
- `manuscript_v2/` current draft, being rewritten around the non-reducibility
  and framing results

## Main findings

See, in order:

1. `rebuild/results/CROSS_MODALITY.md` rank agreement falls in three steps and
   vanishes across modality
2. `rebuild/results/FRAMING_LAW.md` a clinical reference's verdict is a
   monotonic function of its lesion fraction, with held-out datasets on the curve
3. `rebuild/results/EXTERNAL_VALIDATION.md` benchmark validity is not reducible
   to a single measured image statistic
4. `rebuild/results/PROFILE_AND_CHROMA.md` a multi-scale background descriptor
   as a diagnostic
5. `rebuild/results/GAP_AND_SCOPE.md` where this sits in the literature
