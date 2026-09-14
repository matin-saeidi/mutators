# Running the simulations

How to build the simulators, run a workflow on a cluster, and read what comes
out. The model and the parameter values are in the top-level
[`README.md`](../README.md); this is the operational side.

## Before you start

Build the simulators once:

```bash
make sims          # -> src/simulations/bin/
```

Then set your account and partition in
[`workflow/profiles/slurm/config.yaml`](../workflow/profiles/slurm/config.yaml).
The profile ships with `CHANGE_ME` as the account so a misconfigured run fails
immediately rather than submitting somewhere unexpected. It uses snakemake's
`cluster-generic` executor and an `sbatch` command built from each rule's
`resources:`; adapt that line for a different scheduler.

## Running a workflow

```bash
snakemake --snakefile workflow/const_phiG.smk \
          --profile workflow/profiles/slurm \
          --config pop=sas
```

Run it from a **tmux session on a submit node**. The snakemake process has to
outlive every job it submits, and an interactive allocation that ends takes the
controller with it, leaving the cluster jobs running with nothing to merge or
summarise them.

```bash
tmux new -s mutators
snakemake --snakefile workflow/varying_selection.smk --profile workflow/profiles/slurm
# ctrl-b d to detach, tmux attach -t mutators to return
```

**Stopping:** Ctrl-C once, in the tmux window. The profile sets
`cluster-generic-cancel-cmd: scancel`, so the in-flight jobs are cancelled too.

**Resuming:** run the same command again. `rerun-incomplete: True` picks up from
whatever is already on disk and redoes any chunk that was mid-flight.

**Configuration:**

| option | effect |
|---|---|
| `--config pop=sas` | demographic history: `sas`, `eur` (NFE) or `afr` |
| `--config results_dir=/scratch/you/mutators` | write output somewhere other than `./results` |

Set `MUTATORS_RESULTS` to the same path when drawing figures.

## What a workflow produces

Output is laid out per population and per workflow:

```
results/<pop>/<workflow output dir>/
    <GENE>/s_<s>.out              raw simulator output
    <GENE>/s_<s>.params.txt       every argument the simulator was given
    <GENE>/<GENE>_<pop>_*.npz     the summarised distributions
    <GENE>/<GENE>_<pop>_*.json    summary statistics and metadata
    tmp_chunks/                   per-chunk output, deleted once merged
```

| workflow | output directory | final product |
|---|---|---|
| `const_phiG.smk` | `<pop>/simulations/` | `<GENE>_<pop>_dem_const_phiG.npz` |
| `varying_selection.smk` | `<pop>/simulations_varying_s/` | `<GENE>_<pop>_varying_s.npz` |
| `varying_dominance.smk` | `<pop>/simulations_varying_h/` | `<GENE>_<pop>_varying_h_<scale>_s.npz` |
| `const_phiG_shet_1e-4.smk` | `<pop>/simulations_shet_1e-4/` | `<GENE>_<pop>_dem_const_phiG_shet_1e-4.npz` |
| `const_phiG_no_back_mut.smk` | `<pop>/simulations_no_back_mut/` | `<GENE>_<pop>_dem_const_phiG_no_back_mut.npz` |
| `mutator_ages.smk` | `<pop>/simulations_ages/` | `all_mutators_<pop>_ages.npz`, `.json`, `.tsv` |
| `XPC_varying_gene_mu.smk` | `sas/simulations_varying_mu/` | `XPC_sas_varying_mu.npz`, `XPC_const1M_varying_mu.npz` |
| `model_validation.smk` | `model_validation/` | `trajectories/*.npz`, `s_vs_q/*.npz` |

Workflows that sweep a parameter summarise each point on its own into `per_s/`
or `per_mu/` and then stitch those into the single final `.npz`. This lets each
large raw `.out` be released as soon as its own summary exists, instead of the
whole sweep having to sit on disk at once. Set `KEEP_RAW_OUT = True` at the top
of a Snakefile to retain them; budget the disk first.

## npz keys

Keys differ by gene, because MUTYH's three variants share one simulation and a
bare key could not say which variant it refers to.

| workflow | gene | key | example |
|---|---|---|---|
| `const_phiG*` | MPG, POLD1, POLE | `<s>` | `0.0102` |
| `const_phiG*` | XPC | `XPC_freq_s=<s>`, `XPC_age_s=<s>` | `XPC_freq_s=0.0187` |
| `const_phiG*` | MUTYH | `<variant>_freq_s=<s>`, `<variant>_age_s=<s>` | `Y179C_freq_s=0.0020` |
| `varying_selection` | MPG, POLD1, POLE, XPC | `<s>` | `0.00037402` |
| `varying_selection` | MUTYH | `<variant>_freq_s=<s>` | `G368D_freq_s=0.00003996` |
| `varying_dominance` | MPG, XPC | `<h>` | `0.025` |
| `XPC_varying_gene_mu` | XPC | `<mu>`, `age_s=<mu>` | `1.06135934e-06` |
| `mutator_ages` | all | `<label>__<variant>_maxage_s=<s>` | `MPG__MPG_maxage_s=0.0102` |

Where there is a single focal variant, unprefixed aliases (`freq_s=<s>`,
`age_s=<s>`) are written as well.

`s` is formatted with the precision the run used — 4 decimals for the
constant-effect-size runs, 8 across the grids — so read the key out of
`z.files` rather than reconstructing it.

**Frequency arrays are per run; age arrays are per lineage.** The two are not
aligned with each other. For a per-run age use the `mutator_ages` output, whose
`maxage`, `freq` and `nlin` arrays *are* aligned run-for-run.

## The JSON summaries

Each `.npz` has a `.json` beside it with the same statistics, plus the counts
behind them and every parameter passed with `--meta`.

| key | over which runs |
|---|---|
| `mean`, `median`, `variance`, `ci_*` | every replicate, i.e. the array as saved |
| `segregating` | that mutator's own `q > 0` subset |
| `all_segregating` | replicates in which every variant of a compound-het gene segregates |
| `n_runs_total`, `n_runs_segregating`, `p_segregating` | the counts behind the above |
| `conditioned_on_segregating` | `false`; the figure scripts check this |

## Conditioning

Every replicate reaches the saved `.npz`, and a replicate in which the mutator
was lost enters at a frequency of exactly 0. The arrays therefore have one entry
per run, and for MUTYH the three variant arrays are aligned run-for-run: index
`i` is the same replicate in all three.

Conditioning is applied downstream, not at summarise time:

```python
import numpy as np
z = np.load("results/sas/simulations/MPG/MPG_sas_dem_const_phiG.npz")
q = z["0.0102"]

q                    # every replicate, zeros included
q[q > 0]             # conditional on the mutator segregating
```

```python
z = np.load("results/sas/simulations/MUTYH/MUTYH_sas_dem_const_phiG.npz")
y, v, g = (z[f"{n}_freq_s=0.0020"] for n in ("Y179C", "V234M", "G368D"))

y[y > 0]                            # Y179C conditional on itself segregating
keep = (y > 0) & (v > 0) & (g > 0)  # replicates where all three segregate
y[keep], v[keep], g[keep]
```

Allele ages are the exception: a replicate in which the variant never arose has
no age, so the age distributions are conditioned on the variant being present.

## Scale

Run counts are sized by the most demanding analysis the output has to support
rather than by what survives summarising. For MUTYH that is the estimator
conditioning on all three variants segregating at once, whose size tracks the
present-day `Ne`:

| history | present-day Ne | all three segregating | runs |
|---|---|---|---|
| NFE | 31,498,556 | ~83% | 1e5 |
| SAS | 4,837,454 | 12.4% | 7.5e5 |
| AFR | 2,763,103 | 4.5% | 2e6 |

The other genes use 1e5 runs in the constant-effect-size workflows.

Costs, approximately:

| workflow | jobs | CPU-hours |
|---|---|---|
| `const_phiG.smk` (sas) | 39 | 7 |
| `varying_selection.smk` | ~4,500 | ~4,700 |
| `model_validation.smk` | 8 | 2 |

Walltimes are set to `11:59:00`, just under the 12-hour cap of the partition
these runs used. Size them generously: contention on shared nodes made the same
job run several times slower on a busy node than on a quiet one. Memory is not
worth tuning — measured peak usage is well under 1 GB per job.

Chunk counts are per gene, in the `GENES` dict at the top of each Snakefile.
Raise `chunks` to spread a gene over more, shorter jobs.

## Re-summarising without re-simulating

Each summarise rule declares its script as an `input:`, so editing a summariser
marks the affected summaries out of date and the next run rebuilds them from the
existing `.out` files. This only works where the raw output was kept.

## Another demographic history

`--config pop=eur` or `pop=afr` runs the same job graph under the NFE or African
history. `const_phiG.smk` carries a `RUN_PLAN` table with per-history MUTYH run
counts; the other workflows would want their run counts re-checked, since the
fraction of replicates in which a mutator segregates varies with the
present-day `Ne`.
