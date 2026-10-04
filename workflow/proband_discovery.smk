# ============================================================================
# Proband discovery in trio studies for Figures 4, S6 and S12 (Supplementary
# Section S6).
#
# 1. Mutator frequencies: single-site simulations under the NFE history of a
#    recessive (h = 0) and a semi-dominant (h = 0.5) mutator at 100 values of
#    phi_G from 10 to 1000, s* = 8e-5 phi_G, mu = 1.25e-8, 2,000,000
#    replicates per value.
# 2. Trio studies: 250,000 simulated cohorts of 22,000 trios for each
#    frequency distribution (src/proband/simulate_trios.py), for
#        age_sex     M = 10, 100     Figure 4
#        no_age_sex  M = 1           Figure S12
#    Other values of M: --config age_sex_M=1,10,100,1000 no_age_sex_M=1
#
# Run from the repository root:
#   snakemake --snakefile workflow/proband_discovery.smk --profile workflow/profiles/slurm
#
# Output, under results/proband_discovery/:
#   frequencies/<dominance>/s_<s>.npz (+ .json)       present-day frequency of
#       every replicate, keyed by s
#   trios/n_22000/<arm>/M_<M>/<dominance>/s_<s>.npz (+ .json)
#
# Step 1 is 800 jobs of about 1 h; its raw output (~35 GB) is deleted once
# summarised, unless KEEP_RAW_OUT = True. Step 2 is 600 jobs of under 1 h.
# ============================================================================

from pathlib import Path

import numpy as np

# — Paths ————————————————————————————————————————————————————
REPO      = Path(workflow.basedir).parent.resolve()
RESULTS   = Path(config.get("results_dir", REPO / "results")).resolve()
BINARY    = REPO / "src" / "simulations" / "bin" / "single_site_simulator"
SUMMARIZE = REPO / "src" / "summaries" / "summarize_single_site.py"
TRIOS     = REPO / "src" / "proband" / "simulate_trios.py"
OUTROOT   = f"{RESULTS}/proband_discovery"
TMPROOT   = f"{OUTROOT}/tmp_chunks"

KEEP_RAW_OUT = False
maybe_temp = (lambda p: p) if KEEP_RAW_OUT else temp

# — Frequency distributions ——————————————————————————————————
DOMINANCE   = {"recessive": "0", "semi_dominant": "0.5"}
F, S_HET    = 0.08, 5e-4
MU          = 1.25e-8
POP         = "eur"           # NFE history (src/simulations/README.md)
FREQ_RUNS   = 2_000_000
FREQ_CHUNKS = 4

PHI_G = np.logspace(1, 3, 100)
S     = [f"{2 * F * S_HET * x:.6e}" for x in PHI_G]
PHI_G_OF = dict(zip(S, PHI_G))


def freq_seed(dom, s, chunk):
    return 1 + (list(DOMINANCE).index(dom) * len(S) + S.index(s)) * FREQ_CHUNKS + int(chunk)

# — Trio studies ——————————————————————————————————————————————
N_TRIOS   = int(config.get("n_trios", 22000))
TRIO_REPS = int(config.get("trio_reps", 250000))


def _ints(value):
    return [int(v) for v in str(value).split(",") if v.strip()]


ARMS = {"age_sex":    _ints(config.get("age_sex_M", "10,100")),
        "no_age_sex": _ints(config.get("no_age_sex_M", "1"))}
TRIO_OUT = f"{OUTROOT}/trios/n_{N_TRIOS}"

wildcard_constraints:
    dom   = "|".join(DOMINANCE),
    s     = r"[0-9.]+e[+-][0-9]+",
    chunk = r"\d+",
    arm   = "age_sex|no_age_sex",
    M     = r"\d+",

localrules: all


rule all:
    input:
        [f"{TRIO_OUT}/{arm}/M_{M}/{dom}/s_{s}.npz"
         for arm, Ms in ARMS.items() for M in Ms for dom in DOMINANCE for s in S],


# — 1. Frequency distributions ——————————————————————————————————
# Arguments: mutU sel DOM RUNS mut_uncert dem_uncert dem_model Ne seed num_gens pop
# (src/simulations/README.md)
rule simulate_frequencies_chunk:
    output:
        out = temp(f"{TMPROOT}/{{dom}}/s_{{s}}_chunk{{chunk}}.out"),
    params:
        h    = lambda wc: DOMINANCE[wc.dom],
        runs = FREQ_RUNS // FREQ_CHUNKS,
        seed = lambda wc: freq_seed(wc.dom, wc.s, wc.chunk),
    resources:
        partition     = "short",
        time          = "'11:59:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "500M",
    shell:
        r"""
        mkdir -p $(dirname {output.out})
        {BINARY} {MU} {wildcards.s} {params.h} {params.runs} 0 0 1 1e6 {params.seed} 2.5e5 {POP} \
          > {output.out}
        """


rule merge_frequency_chunks:
    input:
        chunks = expand(f"{TMPROOT}/{{{{dom}}}}/s_{{{{s}}}}_chunk{{chunk}}.out",
                        chunk=range(FREQ_CHUNKS)),
    output:
        out         = maybe_temp(f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.out"),
        params_file = f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.params.txt",
    params:
        h     = lambda wc: DOMINANCE[wc.dom],
        seeds = lambda wc: ",".join(str(freq_seed(wc.dom, wc.s, c)) for c in range(FREQ_CHUNKS)),
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "1000M",
    shell:
        r"""
        cat {input.chunks} > {output.out}
        {{
          echo "simulator={BINARY}"
          echo "population={POP}"
          echo "mut_rate={MU}"
          echo "s={wildcards.s}"
          echo "h_mutator={params.h}"
          echo "runs={FREQ_RUNS}"
          echo "n_chunks={FREQ_CHUNKS}"
          echo "chunk_seeds={params.seeds}"
        }} > {output.params_file}
        """


rule summarize_frequencies:
    input:
        script = SUMMARIZE,
        out    = f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.out",
    output:
        npz  = f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.npz",
        json = f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.json",
    params:
        h = lambda wc: DOMINANCE[wc.dom],
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "4000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --out-json {output.json} \
          --out-npz  {output.npz} \
          --population {POP} \
          --meta h_mutator={params.h} --meta mut_rate={MU}
        """


# — 2. Trio studies ——————————————————————————————————————————————
rule simulate_trios:
    input:
        script = TRIOS,
        freqs  = f"{OUTROOT}/frequencies/{{dom}}/s_{{s}}.npz",
    output:
        npz  = f"{TRIO_OUT}/{{arm}}/M_{{M}}/{{dom}}/s_{{s}}.npz",
        json = f"{TRIO_OUT}/{{arm}}/M_{{M}}/{{dom}}/s_{{s}}.json",
    params:
        h     = lambda wc: DOMINANCE[wc.dom],
        phi_G = lambda wc: repr(float(PHI_G_OF[wc.s])),
        flag  = lambda wc: "--age-sex" if wc.arm == "age_sex" else "",
        seed  = lambda wc: (100_000_000 * list(ARMS).index(wc.arm) + 10_000 * int(wc.M)
                            + 1_000 * list(DOMINANCE).index(wc.dom) + S.index(wc.s)),
    resources:
        partition     = "short",
        time          = "'11:59:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "1000M",
    shell:
        r"""
        python3 {input.script} \
          --freqs {input.freqs} --key {wildcards.s} \
          --h {params.h} --phi-G {params.phi_G} --M {wildcards.M} \
          --n-trios {N_TRIOS} --reps {TRIO_REPS} {params.flag} \
          --seed {params.seed} --out {output.npz}
        """
