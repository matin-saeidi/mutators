# ============================================================================
# Mutator simulations for Figures 1, S2 and S3 (Supplementary Section S3.1).
#
# 50 replicates at each of 21 values of phi_G from 1e-2 to 1e3, for each value
# of 2 N s_het in --config two_N_shet (default 10,20,40):
#   Figures 1 and S3   2 N s_het = 20
#   Figure S2          2 N s_het = 10, 20, 40
#
# Parameters: N = 2000, M = 1000, G = 3e8, f = 0.08, u0 = mu = 1.25e-7 and
# s_het = (2 N s_het) / 2N. Each replicate runs 400,000 generations and saves
# the frequencies every 16,000 generations after a burn-in of 20,000.
#
# Run from the repository root:
#   snakemake --snakefile workflow/mutator_simulation.smk --profile workflow/profiles/slurm
#   snakemake --snakefile workflow/mutator_simulation.smk --profile workflow/profiles/slurm \
#             --config two_N_shet=20                 # Figures 1 and S3 only
#
# Output, under results/mutator_simulation/:
#   2Ns_het_<x>/phi_G_<phi_G>/rep_<r>.npz    one replicate
#   2Ns_het_<x>/summary.json                 statistics at each phi_G
#
# 1,050 jobs of 3-6 h for each value of 2 N s_het. A job stopped at the wall
# clock carries on from where it stopped when it is resubmitted.
# ============================================================================

from pathlib import Path

import numpy as np

# — Paths ————————————————————————————————————————————————————
REPO    = Path(workflow.basedir).parent.resolve()
RESULTS = Path(config.get("results_dir", REPO / "results")).resolve()
SCRIPTS = REPO / "src" / "mutator_simulation"
OUTROOT = f"{RESULTS}/mutator_simulation"

# — Parameters ———————————————————————————————————————————————
TWO_N_SHET = [x.strip() for x in str(config.get("two_N_shet", "10,20,40")).split(",")]

N, M  = 2000, 1000
G, F  = 3e8, 0.08
U0    = 1.25e-7
MU    = 1.25e-7
REPLICATES   = 50
GENERATIONS  = 200 * N
BURN_IN      = 10 * N
SAMPLE_EVERY = 8 * N

PHI_G = [f"{x:.4e}" for x in np.logspace(-2, 3, 21)]


def s_het(two_n_shet):
    return float(two_n_shet) / (2 * N)


def seed(wc):
    """Distinct for every (2 N s_het, phi_G, replicate)."""
    return int(float(wc.x)) * 100_000 + PHI_G.index(wc.phi) * 1_000 + int(wc.rep)


wildcard_constraints:
    x   = r"[0-9.]+",
    phi = r"[0-9.]+e[+-][0-9]+",
    rep = r"\d+",

localrules: all


rule all:
    input:
        [f"{OUTROOT}/2Ns_het_{x}/summary.json" for x in TWO_N_SHET],


rule simulate:
    output:
        npz = f"{OUTROOT}/2Ns_het_{{x}}/phi_G_{{phi}}/rep_{{rep}}.npz",
    params:
        script = SCRIPTS / "simulate.py",
        s_het  = lambda wc: s_het(wc.x),
        seed   = seed,
    resources:
        partition     = "short",
        time          = "'11:59:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "1000M",
    shell:
        r"""
        python3 {params.script} \
          --phi-G {wildcards.phi} --s-het {params.s_het} \
          --N {N} --M {M} --G {G} --f {F} --u0 {U0} --mu {MU} \
          --generations {GENERATIONS} --burn-in {BURN_IN} --sample-every {SAMPLE_EVERY} \
          --seed {params.seed} \
          --out {output.npz}
        """


rule summarize:
    input:
        script = SCRIPTS / "summarize.py",
        reps   = lambda wc: expand(f"{OUTROOT}/2Ns_het_{wc.x}/phi_G_{{phi}}/rep_{{rep}}.npz",
                                   phi=PHI_G, rep=range(1, REPLICATES + 1)),
    output:
        json = f"{OUTROOT}/2Ns_het_{{x}}/summary.json",
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "4000M",
    shell:
        r"""
        python3 {input.script} {input.reps} --out {output.json}
        """
