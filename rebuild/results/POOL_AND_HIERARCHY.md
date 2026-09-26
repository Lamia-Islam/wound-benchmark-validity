# The 32-method pool, and what it did to the hierarchy

The biggest objection to this study was that 21 of its 22 methods were classical
thresholding algorithms, so any rank-agreement result risked being a statement
about colour statistics rather than about benchmark validity. Ten modern
architectures have been added. The result is a stronger paper and one wrong
conclusion caught.

---

## 1. What was added

Ten architectures from segmentation_models_pytorch, all randomly initialised, no
pretrained weights anywhere:

| axis | methods |
|---|---|
| seven decoders on one encoder (resnet18) | Unet, UnetPlusPlus, DeepLabV3Plus, FPN, PSPNet, Linknet, MAnet |
| three alternative encoders | Unet/mobilenet_v2, Unet/efficientnet-b0, Segformer/mit_b0 |

Every one trained on the same 393 FUSeg training images at 128 px with the same
Dice+BCE loss, Adam, horizontal flips and 30 epochs as the original U-Net.
Decision thresholds were swept on FUSeg training images 400 to 550, disjoint
from training and from every evaluation slice. Held-out Dice ranged 0.507 to
0.641, so the pool has real spread rather than ten copies of one model.

Pool size goes from 22 to 32. The detection threshold for a Spearman at
p < 0.05 falls from |rho| 0.428 to 0.352.

## 2. The hierarchy, rebuilt

| pair | rho | p | 95% CI | relationship |
|---|---|---|---|---|
| Kvasir vs Kvasir-sessile | +0.952 | <0.0001 | [+0.871, +0.982] | same source |
| AZH test vs AZH train | +0.944 | <0.0001 | [+0.835, +0.984] | same dataset |
| FUSeg vs AZH | +0.660 | 0.0001 | [+0.430, +0.804] | same modality |
| Medetec vs AZH | +0.568 | 0.0012 | [+0.256, +0.781] | same modality |
| Medetec vs FUSeg | +0.461 | 0.0081 | [+0.085, +0.758] | same modality |
| Medetec vs Kvasir | +0.019 | 0.9172 | [-0.318, +0.360] | across |
| Medetec vs Kvasir-sessile | -0.042 | 0.8185 | [-0.397, +0.338] | across |
| AZH vs Kvasir | -0.191 | 0.2991 | [-0.506, +0.164] | across |
| AZH vs Kvasir-sessile | -0.277 | 0.1289 | [-0.579, +0.081] | across |
| FUSeg vs Kvasir | -0.477 | 0.0065 | [-0.706, -0.152] | across |
| FUSeg vs Kvasir-sessile | -0.552 | 0.0011 | [-0.741, -0.224] | across |

All three same-modality pairs are now significant with confidence intervals
excluding zero, which was not true at n = 22 where the weakest pair sat at
p = 0.0385 with a CI touching zero. The two controls are tighter. The structure
survived the pool change, and that is the main thing: it was not an artifact of
ranking thresholding algorithms.

## 3. The result that did not survive, and what replaced it

The pooled across-modality mean is -0.253, and two pairs are significantly
negative. The obvious reading is that rank agreement inverts across modality:
methods good at wounds are worse than chance at endoscopy.

That reading is wrong. Splitting the pool:

| subset | same modality | across modality |
|---|---|---|
| all 32 | +0.563 | **-0.253** |
| classical 22 only | +0.511 | **+0.108** |
| deep 10 only | +0.321 | **+0.236** |

The negative correlation exists only in the pooled comparison. Within either
subgroup it is weakly positive and not significant. This is a composition
effect, and the cause is visible in the mean Dice:

| reference | deep 10 | classical 22 | gap |
|---|---|---|---|
| Medetec | 0.374 | 0.295 | +0.079 |
| FUSeg | 0.581 | 0.132 | +0.449 |
| AZH | 0.396 | 0.330 | +0.066 |
| Kvasir | 0.062 | 0.200 | **-0.138** |
| Kvasir-sessile | 0.024 | 0.114 | **-0.091** |

The deep models were trained on wound photography. On wound references they beat
the untrained classical methods. On endoscopy they collapse to 0.062 and 0.024,
well below the classical methods, which never trained on anything and therefore
degrade gracefully. Two groups swapping places between modalities produces a
negative pooled correlation without any pair of methods actually inverting.

So the correct claims are:

- **Within a homogeneous method pool there is no detectable cross-modality rank
  agreement.** +0.108 for the classical subset, +0.236 for the deep subset,
  neither significant. This is the n = 22 conclusion, and it survives.
- **In a mixed pool, cross-modality rank comparison is dominated by which
  methods had domain-matched training.** The sign of the correlation is set by
  pool composition, not by the references. Anyone comparing benchmarks across
  domains with a pool containing both trained and untrained methods will measure
  this artifact and can easily mistake it for a property of the data.

The second is new, it is a direct consequence of having built the enlarged pool,
and it is a concrete methodological warning rather than a null result.

## 4. A problem this exposed in our own design

The deep models beat classical methods by +0.449 on FUSeg and by only +0.079 and
+0.066 on Medetec and AZH. FUSeg is where they were trained. The images are
disjoint, but the distribution is not.

FUSeg is therefore no longer a neutral clinical reference for this pool. The
original design already had this issue, since the pixel classifiers and the
U-Net were also trained on FUSeg training images, but with one trained method in
22 it was negligible. With eleven in 32 it is not.

Consequences:

- Medetec and AZH are the neutral references for this pool and headline numbers
  should be reported against them.
- The two significantly negative cross-modality pairs are both FUSeg pairs,
  which is consistent with this being the same training-domain artifact rather
  than anything about endoscopy.
- The clinical ceiling should be quoted as Medetec vs AZH, +0.568, rather than
  Medetec vs FUSeg.

## 5. Caveats

- The deep models are trained from random initialisation on 393 images at
  128 px. They are real architectures under a real protocol, but they are not
  the fully trained, augmented, pretrained models a practitioner would deploy.
  The claim is that the pool now spans modern architectural variation, not that
  it reproduces state-of-the-art performance.
- No nnU-Net and no SAM or MedSAM. Both need either a training budget or
  pretrained weights that this environment cannot reach.
- Pairwise agreement values moved when the pool changed. Medetec vs FUSeg fell
  from +0.567 to +0.461 and Medetec vs AZH rose from +0.451 to +0.568. Any
  single agreement number is a property of the reference pair and the method
  pool together, and should never be quoted as a property of the datasets alone.
- The same-modality against across-modality group test is Mann-Whitney on 3
  pairs against 6, U = 18, p = 0.0238, which is the smallest p that design can
  produce. The individual pair significances are what carry the result now.
- One endoscopy source, and Kvasir-sessile is a subset of it rather than an
  independent dataset.
