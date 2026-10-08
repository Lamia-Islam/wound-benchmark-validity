# What halves synthetic benchmark validity: pool or reference?

Generator B, three texture amplitudes, three seeds each. Validity is the mean of the benchmark's Spearman agreement with the two clinical references, as a percentage of the agreement between those two references measured over the same method pool.

| method pool | references | ceiling | best validity | % of ceiling |
|---|---|---|---|---|
| original 22 | Medetec + FUSeg | +0.567 (p 0.0071) | +0.562 (sd 0.047) | 99% |
| original 22 | Medetec + AZH | +0.451 (p 0.0385) | +0.522 (sd 0.048) | 116% |
| original 22 | Medetec + CO2Wounds | +0.529 (p 0.0135) | +0.712 (sd 0.009) | 135% |
| 18 untrained | Medetec + FUSeg | +0.740 (p 0.0007) | +0.511 (sd 0.028) | 69% |
| 18 untrained | Medetec + AZH | +0.486 (p 0.0443) | +0.480 (sd 0.003) | 99% |
| 18 untrained | Medetec + CO2Wounds | +0.540 (p 0.0232) | +0.698 (sd 0.006) | 129% |
| all 32 | Medetec + FUSeg | +0.461 (p 0.0081) | +0.276 (sd 0.007) | 60% |
| all 32 | Medetec + AZH | +0.568 (p 0.0012) | +0.272 (sd 0.006) | 48% |
| all 32 | Medetec + CO2Wounds | +0.502 (p 0.0036) | +0.484 (sd 0.025) | 96% |

Per texture amplitude, mean over three seeds:

| method pool | references | amp 11 | amp 28 | amp 40 |
|---|---|---|---|---|
| original 22 | Medetec + FUSeg | +0.427 (75%) | +0.556 (98%) | +0.562 (99%) |
| original 22 | Medetec + AZH | +0.475 (105%) | +0.506 (112%) | +0.522 (116%) |
| original 22 | Medetec + CO2Wounds | +0.508 (96%) | +0.693 (131%) | +0.712 (135%) |
| 18 untrained | Medetec + FUSeg | +0.417 (56%) | +0.511 (69%) | +0.494 (67%) |
| 18 untrained | Medetec + AZH | +0.480 (99%) | +0.443 (91%) | +0.453 (93%) |
| 18 untrained | Medetec + CO2Wounds | +0.482 (89%) | +0.671 (124%) | +0.698 (129%) |
| all 32 | Medetec + FUSeg | +0.085 (18%) | +0.238 (52%) | +0.276 (60%) |
| all 32 | Medetec + AZH | +0.133 (23%) | +0.236 (42%) | +0.272 (48%) |
| all 32 | Medetec + CO2Wounds | +0.260 (52%) | +0.452 (90%) | +0.484 (96%) |
