# Mutator simulations

The simulations behind Figures 1, S2 and S3 (Supplementary Section S3.1).
`workflow/mutator_simulation.smk` runs them on a cluster.

| file | what it does |
| --- | --- |
| `population.py` | the model |
| `simulate.py` | runs one replicate |
| `summarize.py` | pools the replicates at each phi_G |

## Usage

```bash
python3 src/mutator_simulation/simulate.py --phi-G 100 --seed 1 --out rep_1.npz
python3 src/mutator_simulation/summarize.py rep_*.npz --out summary.json
```

The defaults are the parameters of Figure 1: N = 2000, M = 1000, G = 3e8,
f = 0.08, s_het = 0.005, u0 = mu = 1.25e-7, 400,000 generations, frequencies
saved every 16,000 generations after a burn-in of 20,000. Add `-h` to see all
options. One replicate takes 3-6 hours on one core.

The same `--seed` gives the same output. If a run is stopped (for example at the
wall-clock limit), run the same command again and it continues from where it
stopped.

## Output

`simulate.py`, one file per replicate:

| key | contents |
| --- | --- |
| `freqs` | mutator frequency at each modifier site, one row per saved generation |
| `generations` | the saved generations |
| `N`, `M`, `phi_G`, `G`, `f`, `s_het`, `u0`, `mu`, `burn_in`, `sample_every`, `seed`, `n_generations` | the parameters |

`summarize.py`, one JSON keyed by phi_G: the mean of q and of q^2, the variance
of q, their standard errors, the number of values pooled, and the parameters.

## Provenance

Adapted from the mutator simulations of Milligan et al. (2022).
