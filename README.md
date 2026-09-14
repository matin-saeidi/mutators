# Dynamics of mutators of arbitrary dominance in humans

Code and data for:

> Saeidi M, Sella G, Przeworski M, Milligan WR. Dynamics of mutators of arbitrary dominance in humans. Submitted to *GENETICS*.

We develop a population genetic model of germline mutation rate modifiers ("mutators") with arbitrary dominance, and ask which kinds of mutators trio-based studies are most likely to discover.

## Repository layout

```
mutators/
├── data/       input tables (see data/README.md)
├── src/        analysis code and the forward simulators
├── workflow/   Snakemake workflows that run the simulations
├── figures/    one script per figure
├── tools/      helper scripts
└── docs/       how to run the simulations on a cluster
```

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`, and `matplotlib`. The simulations also
need `snakemake` and, to build the simulators, a C++11 compiler and the Boost
headers. `environment.yml` pins the versions used.

```bash
conda env create -f environment.yml && conda activate mutators
```

## Quick start

```bash
python src/effect_sizes.py
```

This estimates the effect size of each known mutator from trio de novo mutation counts and reproduces Table S3 (and the effect sizes in Table 2).

To run the simulations and redraw the figures:

```bash
make sims                                     # build the simulators -> src/simulations/bin

# edit workflow/profiles/slurm/config.yaml for your cluster, then e.g.
snakemake --snakefile workflow/const_phiG.smk \
          --profile workflow/profiles/slurm --config pop=sas

make figures                                  # results/ -> figures/output/
```

Simulation output goes to `./results` and figures to `./figures/output`. Both can
be moved: pass `--config results_dir=...` to snakemake and set
`MUTATORS_RESULTS` to the same path when drawing figures.

## Where each result comes from

| Analysis | In the paper | Code |
|---|---|---|
| Effect sizes of known mutators | Table 2, Table S3; Supplementary Section S4.1 | `src/effect_sizes.py` |
| Stationary distribution and increase in mean mutation rate | Figure 1; Eqs. 7–11 | *(to be added)* |
| Mutator, single-site, and compound-heterozygote simulations | Figures 1–2, S1, S3, S5, S9, S10; Section S3 | `workflow/`, `src/simulations/`, `src/summaries/` |
| Likelihoods and tests of neutrality | Figures 2–3, Table S6; Section S5 | `figures/fig_sas_combined.py`, `figures/fig_dominance_sweep.py` |
| Proband discovery in trios | Figure 4, Figures S6, S11, S12; Section S6 | *(to be added)* |

Each figure is drawn by one script in `figures/`, which reads the simulation
output named in the last column:

| Figure | Script | Workflow |
|---|---|---|
| Figure 2 | `fig_sas_combined.py` | `const_phiG.smk` (pop=sas), `varying_selection.smk` |
| Figure 3 | `fig_dominance_sweep.py` | `varying_dominance.smk` |
| Figure S1 | `fig_model_validation.py` | `model_validation.smk` |
| Figure S4 | `fig_demographic_histories.py` | — (reads `data/demographic_models/`) |
| Figure S5 | `fig_xpc_varying_mu.py` | `XPC_varying_gene_mu.smk` |
| Figure S7 | `fig_scenario_comparison.py --scenario shet` | `const_phiG.smk`, `const_phiG_shet_1e-4.smk` |
| Figure S8 | `fig_three_histories.py` | `const_phiG.smk` at pop=eur, sas and afr |
| Figure S9 | `fig_scenario_comparison.py --scenario backmut` | `const_phiG.smk`, `const_phiG_no_back_mut.smk` |
| Figure S10 | `fig_allele_ages.py` | `mutator_ages.smk` |

```bash
make figures                                        # all of them
python figures/fig_sas_combined.py                  # just one
```

## Simulations

The forward simulators are in `src/simulations/` (see the README there for how
they are invoked and where they came from). `workflow/` drives them:

| workflow | produces |
|---|---|
| `const_phiG.smk` | frequency distributions at each mutator's estimated effect size, for `--config pop=sas`, `eur` or `afr` |
| `varying_selection.smk` | the same over a 42-point grid in *s* |
| `varying_dominance.smk` | grids in *h* for MPG and XPC, at three values of *s* |
| `const_phiG_shet_1e-4.smk` | the baseline re-run at one fifth the deleterious cost |
| `const_phiG_no_back_mut.smk` | the baseline re-run with back mutation off |
| `mutator_ages.smk` | allele-age distributions, with neutral controls |
| `XPC_varying_gene_mu.smk` | *XPC* focal-variant frequency against the gene-wide LoF rate |
| `model_validation.smk` | trajectories and *s* against *q*, checking the model |

See [`docs/running_the_simulations.md`](docs/running_the_simulations.md) for how
to run these on a cluster, what they write, and how to read it.

Simulation output is not distributed: the figure inputs come to about 1 GB of
`.npz` and the raw simulator output behind them to several hundred GB. The
workflows above regenerate it. The selection-coefficient grid is the expensive
one, at roughly 4,700 CPU-hours.

### Selection coefficients

A mutator raises the genome-wide mutation rate by `phi` per generation, and its
own selection coefficient in a homozygote is

```
s = 2 * phi_hom * f * G * s_het
```

with `s_het = 5e-4` the heterozygous fitness cost of one induced deleterious
mutation, `G = 3e9` bp, and `f = 0.08` the fraction of the genome under
selection. `phi_hom` is the homozygous effect size; where the estimate is for a
heterozygous carrier (POLE, POLD1) it is doubled. This is the same relation
`src/effect_sizes.py` applies, and the workflows use the `phi` values that script
produces.

The dominance of the mutator allele itself is separate, and is 0 for the
recessive mutators and 0.5 for the semi-dominant ones.

### What the saved distributions contain

Every simulated replicate reaches the saved `.npz`, and a replicate in which the
mutator was lost enters at a frequency of exactly 0, so the arrays have one entry
per run and MUTYH's three variant arrays are aligned run-for-run. Conditioning is
applied where the analysis is:

```python
z = np.load("results/sas/simulations/MPG/MPG_sas_dem_const_phiG.npz")
q = z["0.0102"]

q                    # every replicate, zeros included
q[q > 0]             # conditional on the mutator segregating
```

The published figures use every replicate. Allele ages are the exception: a
replicate in which the variant never arose has no age.

### Reproducibility

The C++ simulators seed from the wall clock, so re-running one with identical
arguments gives different replicates. The frequency and age distributions
reproduce statistically, to within Monte Carlo error at 1e5–4e6 replicates per
point, rather than replicate for replicate. `model_validation.smk` is pure Python
and is deterministic, as are the summarisers and the figure scripts.

## Data

`data/` contains the small tables the analyses start from: gnomAD v4.1.1 allele counts of the known mutators by genetic ancestry group, Roulette mutation rates to each mutator allele, the *XPC* loss-of-function mutation rate, the trio DNM counts behind the effect-size estimates, and the demographic histories. Column definitions, sources, and processing notes are in [`data/README.md`](data/README.md).

## Contact

Matin Saeidi (ss6917@columbia.edu).
