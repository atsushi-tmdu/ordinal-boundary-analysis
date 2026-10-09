# Manuscript method-to-code map

| Paper question or sensitivity | Study adapter | Main model and scope |
|---|---|---|
| Q1; Table 2 | `StudyRunner.execute(..., 'q1', outcome)` | GCS 8/9 pair-only sample; categorical GCS, prespecified covariates, cluster-robust covariance, marginally standardized risk difference and odds ratio |
| Q1 pooled estimate | `combine_q1(first, second)` | Independent-era inverse-variance fixed-effect pooling; pooled OR on log scale; between-era log-OR contrast |
| Q2; Table S16 | `StudyRunner.execute(..., 'q2', outcome)` | Categorical GCS 3–15; E = (p8-p9) - [(p7-p8)+(p9-p10)]/2, with joint standardized-probability covariance |
| Q3 local | `StudyRunner.execute(..., 'q3_local', outcome)` | GCS 6–10, two-slope hinge reference, one additional lower-score step at 8.5 |
| Q3 full; Figure 4 and candidate-boundary tables | `StudyRunner.execute(..., 'q3_full', outcome)` | GCS 3–15, quartic graded reference, twelve lower-score step boundaries at 3.5–14.5 |
| GCS 5–12 spline; Table S18 | `StudyRunner.execute(..., 's18', outcome)` | Natural cubic spline with boundary knots 5/12 and interior knots 7/9/11; seven boundaries 5.5–11.5 |

The principal eras are 2008–2014 and 2019–2023; supported outcomes are severe head injury, death, urgent cranial surgery, and recorded post-arrival intubation. Covariate definitions, year centering, expected design ranks, and support restrictions are in the two JSON configuration files and the adapter. These specifications are intended to correspond to the paper's described statistical questions, but this newly organized package has not been validated as a byte-identical historical analysis pipeline.

Data ingestion from original JTDB extracts, source linkage, clinical adjudication, eligibility screening, and other supplementary analyses are not distributed. The code accepts an already harmonized and eligible input table. Actual patient-level study results cannot be reproduced from this repository alone. The original restricted datasets, original internal audit records, and detailed fitted outputs are not supplied.

No assertion is made that synthetic-data test results establish equality to original study estimates. Source records and complete numerical equivalence remain subject to separate authorized verification.
