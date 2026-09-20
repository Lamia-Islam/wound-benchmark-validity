# Recovered manuscript: reading order

All five files below were reconstructed from conversation
f7ee7590-a160-40ce-bc6c-e90dd81e1c63 (August 2026). The originals lived in a
Claude cloud container that has since been wiped.

Read and assemble in this order:

| # | file | covers |
|---|---|---|
| 1 | `BATCH1_title_abstract_intro_relatedwork.md` | Title, abstract, Section 1 Introduction. Its Section 2 is a PLACEHOLDER. |
| 2 | `BATCH1_REVA_related_work.md` | Section 2 Related Work, verified citations. REPLACES the Section 2 in file 1. |
| 3 | `BATCH2_methods_data_protocol.md` | Sections 3 Methods, 4 Data, 5 Evaluation protocol. Contains three revisions issued later in the same turn, marked inline. |
| 4 | `BATCH3_results_limitations.md` | Part A: the Batch 2 replacements (already noted inline in file 3). Part B: Section 6 Results. Part C: Section 8 Limitations. |
| 5 | `BATCH4_discussion_conclusions_figures.md` | Section 7 Discussion, Section 9 Conclusions, Section 10 Declarations, figure specs, pre-submission checklist. |

Section numbering in the final paper:
1 Introduction, 2 Related Work, 3 Methods, 4 Data, 5 Evaluation Protocol,
6 Results, 7 Discussion, 8 Limitations, 9 Conclusions, 10 Declarations.

See `../notes/NUMBERS_AND_CORRECTIONS.md` for which figures are final and which
are superseded, and `../notes/CODE_INVENTORY.md` for what has to be rebuilt.

## Blocking issue

Section 10 Data Availability currently claims that all code, both generators,
all 22 methods, trained models and result files are available at
`github.com/Lamia-Islam/surgical-vision-ros2`. That is false. The repo contains
only the Track A (SoftwareX) code. The paper cannot be submitted until this is
either true or rewritten.
