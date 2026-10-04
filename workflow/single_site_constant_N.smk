# ============================================================================
# Single-site simulations at constant N = 2000 for Figure S3.
#
# Recessive mutator (h = 0) with selection coefficient s* = 0.0008 phi_G at the
# 21 values of phi_G of workflow/mutator_simulation.smk, mu = 1.25e-7,
# 100,000 replicates per value, 250,000 generations each.
#
# Run from the repository root:
#   snakemake --snakefile workflow/single_site_constant_N.smk --profile workflow/profiles/slurm
#
# Output, under results/single_site_constant_N/:
#   s_<s>.out, s_<s>.params.txt                raw simulator output and its arguments
#   single_site_recessive_N_2000.npz (+ .json)  present-day frequency of every
#                                               replicate, one array per s, keyed by s
#
# 21 jobs of about 15 minutes.
# ============================================================================

from pathlib import Path

import numpy as np

# — Paths ————————————————————————————————————————————————————
REPO    = Path(workflow.basedir).parent.resolve()
RESULTS = Path(config.get("results_dir", REPO / "results")).resolve()
SUMMARIZE = REPO / "src" / "summaries" / "summarize_single_site.py"
BINARY    = REPO / "src" / "simulations" / "bin" / "single_site_simulator"
OUTROOT   = f"{RESULTS}/single_site_constant_N"

# — Parameters ———————————————————————————————————————————————
N        = 2000
MU       = 1.25e-7
H        = 0
RUNS     = 100000
N_GENS   = 250000
F, S_HET = 0.08, 0.005

PHI_G = np.logspace(-2, 3, 21)
S     = [f"{2 * F * S_HET * x:.6e}" for x in PHI_G]

wildcard_constraints:
    s = r"[0-9.]+e[+-][0-9]+",

localrules: all


rule all:
    input:
        f"{OUTROOT}/single_site_recessive_N_{N}.npz",


# Arguments: mutU sel DOM RUNS mut_uncert dem_uncert dem_model Ne seed num_gens
# (src/simulations/README.md)
rule simulate:
    output:
        out         = f"{OUTROOT}/s_{{s}}.out",
        params_file = f"{OUTROOT}/s_{{s}}.params.txt",
    params:
        seed = lambda wc: 1 + S.index(wc.s),
    resources:
        partition     = "short",
        time          = "'11:59:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "500M",
    shell:
        r"""
        mkdir -p $(dirname {output.out})
        {BINARY} {MU} {wildcards.s} {H} {RUNS} 0 0 0 {N} {params.seed} {N_GENS} > {output.out}

        {{
          echo "simulator={BINARY}"
          echo "mut_rate={MU}"
          echo "s={wildcards.s}"
          echo "h_mutator={H}"
          echo "runs={RUNS}"
          echo "Ne={N}"
          echo "num_generations={N_GENS}"
          echo "random_seed={params.seed}"
        }} > {output.params_file}
        """


rule summarize:
    input:
        script = SUMMARIZE,
        outs   = [f"{OUTROOT}/s_{s}.out" for s in S],
    output:
        npz  = f"{OUTROOT}/single_site_recessive_N_{N}.npz",
        json = f"{OUTROOT}/single_site_recessive_N_{N}.json",
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "4000M",
    shell:
        r"""
        python3 {input.script} {input.outs} \
          --out-json {output.json} \
          --out-npz  {output.npz} \
          --meta Ne={N} --meta mut_rate={MU} --meta h_mutator={H} \
          --meta f={F} --meta s_het={S_HET}
        """
