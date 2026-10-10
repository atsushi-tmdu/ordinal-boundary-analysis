# Ordinal-boundary statistical methods

MIT-licensed software implementing the principal statistical model specifications for the study's Q1, Q2, Q3, and supplementary S18 analyses.

This package provides reusable components for clustered binary-outcome regression, marginal standardization, local contrasts, and nested boundary comparisons. The `ordinal_boundary.studies.primary` adapter and `configs/` specify the three principal analytical questions and the GCS 5–12 spline sensitivity described in *Reproducible Intubation Contrast Without Evidence of a Unique Threshold at Glasgow Coma Scale 8*.

## Scope

- Q1: Pair-specific GCS 8 versus 9, standardized risks, risk difference, and odds ratio; independent-era inverse-variance pooling and a between-era contrast.
- Q2: Categorical-score local excess at GCS 8/9 with joint covariance.
- Q3: Local two-slope reference with an 8/9 step; full-range quartic reference with 12 candidate boundaries.
- S18: GCS 5–12 natural cubic-spline reference, interior knots at 7, 9, 11, with seven candidate boundaries.

The two primary eras and four outcomes are represented in the configuration files. Analysis requires an **already eligible and harmonized** cohort. The software does not extract raw JTDB files, reconstruct historical registry cohorts, reproduce every supplementary analysis, or distribute restricted research data. The refactored adapter has not been established to reproduce every original patient-level fit. See `docs/METHODS_MAP.md`.

## Setup

Python 3.11 or later with the dependencies specified in `pyproject.toml`:

```bash
python3 -m pip install -e '.[test]'
pytest -q
```

Inspect a configuration without loading data or fitting a model:

```bash
python3 scripts/study.py --era 2008-2014 --task q2 --outcome intubation
```

Run a synthetic example:

```bash
python3 scripts/study.py --era 2008-2014 --task q1 --synthetic
python3 scripts/study.py --era 2008-2014 --task q2 --synthetic
python3 scripts/study.py --era 2008-2014 --task q3_local --synthetic
python3 scripts/study.py --era 2008-2014 --task q3_full --synthetic
python3 scripts/study.py --era 2008-2014 --task s18 --synthetic
```

With separately authorized, locally prepared eligible input, specify both the private input and a new, empty private output directory outside this repository:

```bash
python3 scripts/study.py --era 2008-2014 --outcome intubation --task q1 --execute --input /private/eligible.csv --private-output /private/results_q1
```

Never upload participant-level data or generated private output to this repository. The command is an interface example; the input path is a placeholder.

## Inputs

One row per eligible participant with columns `sid`, `hospital`, `gcs`, `year`, `age`, `male`, `blunt`, `alcohol_yes`, `alcohol_unknown`, `sbp`, `hr`, `rr`, and binary outcomes `severe_head`, `death`, `head_surgery`, `intubation`. A given analysis requires its selected outcome and covariates, not all four outcomes. Cohort selection and original clinical variable definitions are governed by the manuscript and supplementary methods. The input schema here is a normalized computational interface, not a substitute for those eligibility rules.

## Verification

Automated tests exercise numerical primitives, contract enforcement, synthetic examples, model-comparison identities, and both primary-era configurations. Passing tests does not constitute a reproduction of study estimates from restricted JTDB records.

## Distribution

The software and documentation are distributed under the MIT License (see LICENSE). This repository provides selected statistical model implementations and synthetic verification examples, not participant-level data, original cohort extraction procedures, restricted research outputs, or a complete numerical reproduction of the study estimates. See the tagged release and repository visibility on GitHub for current availability.
