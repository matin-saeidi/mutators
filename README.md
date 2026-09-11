# Dynamics of mutators of arbitrary dominance in humans

Code and data for:

> Saeidi M, Sella G, Przeworski M, Milligan WR. Dynamics of mutators of arbitrary dominance in humans. Submitted to *GENETICS*.

We develop a population genetic model of germline mutation rate modifiers ("mutators") with arbitrary dominance, and ask which kinds of mutators trio-based studies are most likely to discover.

## Repository layout

```
mutators/
├── data/     input tables (see data/README.md)
└── src/      analysis code
```

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`, and `matplotlib`.

## Quick start

```bash
python src/effect_sizes.py
```

This estimates the effect size of each known mutator from trio de novo mutation counts and reproduces Table S3 (and the effect sizes in Table 2).

## Where each result comes from

| Analysis | In the paper | Code |
|---|---|---|
| Effect sizes of known mutators | Table 2, Table S3; Supplementary Section S4.1 | `src/effect_sizes.py` |
| Stationary distribution and increase in mean mutation rate | Figure 1; Eqs. 7–11 | *(to be added)* |
| Mutator, single-site, and compound-heterozygote simulations | Figures 1–2, S1, S3, S5, S9, S10; Section S3 | *(to be added)* |
| Likelihoods and tests of neutrality | Figures 2–3, Table S6; Section S5 | *(to be added)* |
| Proband discovery in trios | Figure 4, Figures S6, S11, S12; Section S6 | *(to be added)* |

## Data

`data/` contains the small tables the analyses start from: gnomAD v4.1.1 allele counts of the known mutators by genetic ancestry group, Roulette mutation rates to each mutator allele, the *XPC* loss-of-function mutation rate, and the trio DNM counts behind the effect-size estimates. Column definitions, sources, and processing notes are in [`data/README.md`](data/README.md).

## Contact

Matin Saeidi (ss6917@columbia.edu).
