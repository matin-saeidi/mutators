# ============================================================================
# Varying XPC gene-wide LoF mutation rate.
#
# Question: how does the gene-wide LoF mutation rate of XPC change the
# frequency distribution of the FOCAL XPC LoF variant?
#
# The compound-het simulator tracks every LoF lineage in the gene, so what is
# simulated is the gene-wide LoF allele at rate mu_gene. The summariser then
# assigns each surviving lineage to the focal variant with probability
# mu_variant / mu_gene. Both halves therefore have to move together: the same
# mu is handed to the simulator AND to `summarize_comp_het.py --gene-mu`.
# That coupling is the whole point of this Snakefile.
#
# Grid: 21 log-spaced values of mu_gene, from the focal variant's own rate
# (7.922075e-08, where the variant IS the whole gene and every lineage is kept)
# up to twice the estimated XPC gene LoF rate (2 x 4.4355e-06 = 8.8710e-06).
#
# Two demographic settings, run through the same graph:
#   sas      : South Asian history      (comp_het_lineage_tracker), 1e5 runs/mu
#   const1M  : constant Ne = 1e6        (comp_het_lineage_tracker_constN),       5e3 runs/mu
#
# s is held FIXED across the grid at the value implied by XPC's effect size
# (phi = 7.7921e-08 -> s = 0.018701), so mu is the only thing that varies.
#
# Final product: two npz + two json per setting, keyed by the mu string:
#   simulations_varying_mu/XPC/XPC_sas_varying_mu.npz      -> "7.92207500e-08", ... "8.87100000e-06"
#   simulations_varying_mu/XPC/XPC_const1M_varying_mu.npz
#
# Run from the repository root, in a
# tmux session so it survives a disconnect:
#   snakemake --snakefile workflow/XPC_varying_gene_mu.smk --profile .
# or to inspect first, add -np.
# ============================================================================

import math
import numpy as np
import os

# — 1. Paths ————————————————————————————————————————————————
# Every path is derived from where this file sits, so a clone runs anywhere.
# Send results elsewhere with:  --config results_dir=/scratch/you/mutators
from pathlib import Path

REPO    = Path(workflow.basedir).parent.resolve()
RESULTS = Path(config.get("results_dir", REPO / "results")).resolve()
SCRIPTS = str(REPO / "src" / "summaries")
SIMBIN  = REPO / "src" / "simulations" / "bin"

OUTROOT  = f"{RESULTS}/sas/simulations_varying_mu"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
DEM_BIN     = str(SIMBIN / "comp_het_lineage_tracker")
CONSTN_BIN  = str(SIMBIN / "comp_het_lineage_tracker_constN")

GENE = "XPC"

# — 1b. Disk policy —————————————————————————————————————————
# Measured output is ~17.5 kB/run at the central mu and scales roughly linearly
# with mu, so the SAS half of the grid is ~17 GB of raw .out if every merged
# file is kept at once (the const-Ne half, at 5e3 runs, is ~1 GB). Each mu is
# summarised on its own and its merged .out released immediately, which caps
# disk at whatever is in flight. Flip to True to keep them -- e.g. if you later
# want to re-summarise for allele ages without re-simulating.
KEEP_RAW_OUT = False
maybe_temp = (lambda p: p) if KEEP_RAW_OUT else temp

# — 2. Selection: the mutator's own fitness effect, held fixed ——————
# h/s below describe the deleterious mutations XPC's loss causes, not XPC itself.
H_DEL   = 0.5        # dominance of an induced deleterious mutation
S_DEL   = 0.001      # selection coefficient of an induced deleterious mutation
G_SIZE  = 3e9        # genome size (bp)
F_SEL   = 0.08       # fraction of the genome under selection

XPC_PHI       = 7.7921e-08   # homozygous effect size, same value as the const-phi run
XPC_PHI_SCOPE = "hom"
H_MUT         = "0"          # XPC mutator allele is recessive

_phi_hom = 2.0 * XPC_PHI if XPC_PHI_SCOPE == "het" else XPC_PHI
S_VALUE  = "{:.6f}".format(2.0 * _phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL)   # 0.018701

# — 3. The mutation-rate grid ————————————————————————————————
# MU_VARIANT is the focal single-variant rate and is the numerator of the
# lineage->variant probability at EVERY grid point; it never varies.
# MU_GENE is the current estimate of the gene-wide XPC LoF rate, used only to
# set the top of the grid (and matching the const-phi/varying-h Snakefiles).
MU_VARIANT = "7.922075e-08"
MU_GENE    = 4.4355e-06
N_MU       = 21

MU_VALUES = ["{:.8e}".format(mu) for mu in np.logspace(
    np.log10(float(MU_VARIANT)), np.log10(2.0 * MU_GENE), N_MU)]

if len(set(MU_VALUES)) != len(MU_VALUES):
    raise ValueError(f"mu values collide after formatting: {MU_VALUES}")

MU_INDEX = {mu: i for i, mu in enumerate(MU_VALUES)}

# — 4. The two settings ——————————————————————————————————————
# cost = measured seconds/run at the bottom, middle and top of the grid, timed
# on 2026-09-01 with 3 runs (const1M) / 30 runs (sas) of the real binaries at
# s = 0.0187, h = 0. Cost is very nearly linear in mu, since it is dominated by
# per-lineage bookkeeping and the grid spans a 112-fold range of mu, so chunks
# are sized per-mu rather than at the worst corner.
COST_MU_POINTS = [float(MU_VARIANT), MU_GENE, 2.0 * MU_GENE]

SETTINGS = {
    "sas": dict(
        binary=DEM_BIN,
        runs=100000,
        cost=[0.0343, 0.329, 0.595],
        population="sas",
        walltime="'11:59:00'", mem="1000M",
        summary_mem="16000M",
    ),
    "const1M": dict(
        binary=CONSTN_BIN,
        runs=5000,
        cost=[0.300, 10.40, 19.50],
        population="const_Ne_1e6",
        walltime="'11:59:00'", mem="1000M",
        summary_mem="8000M",
    ),
}
ALL_SETTINGS = list(SETTINGS)

# — 5. Simulator arguments ————————————————————————————————————
UFACTOR         = "1.0"
ALLOW_BACK_MUT  = "1"

# sas: comp_het_lineage_tracker
BURN_IN_COMPHET = "2.5e5"
BURNIN_NE       = "20000"   # Ne held during the burn-in, matching the other SAS runs

# const1M: comp_het_lineage_tracker_constN
NE_CONST        = "1e6"
BURN_IN_CONSTN  = "2e5"     # 2e5 generations; s = 0.019 equilibrates in ~1/s gens

BASE_SEED     = 10
SEED_OFFSET   = {st: 2000000 * (i + 1) for i, st in enumerate(ALL_SETTINGS)}

# Chunks are sized so each is ~CHUNK_TARGET_H hours nominal, which leaves room
# for the ~10x slowdown seen on contended nodes inside the 11:59:00 limit.
CHUNK_TARGET_H = 1.0
MAX_CHUNKS     = 60


def cost_per_run(setting, mu):
    """Interpolate the measured seconds/run at this mu."""
    return float(np.interp(float(mu), COST_MU_POINTS, SETTINGS[setting]["cost"]))


def n_chunks(setting, mu):
    total_s = SETTINGS[setting]["runs"] * cost_per_run(setting, mu)
    return max(1, min(MAX_CHUNKS, math.ceil(total_s / (CHUNK_TARGET_H * 3600))))


def chunk_list(setting, mu):
    return list(range(n_chunks(setting, mu)))


def chunk_runs(setting, mu, chunk):
    """Runs in one chunk; the chunks sum to exactly runs."""
    n = n_chunks(setting, mu)
    base, rem = divmod(SETTINGS[setting]["runs"], n)
    return base + (1 if int(chunk) < rem else 0)


def seed_for(setting, mu, chunk):
    return BASE_SEED + SEED_OFFSET[setting] + 1000 * MU_INDEX[mu] + int(chunk)


def per_mu_npz(setting, mu):
    return f"{OUTROOT}/{GENE}/{setting}/per_mu/{GENE}_{setting}_mu_{mu}.npz"


def per_mu_json(setting, mu):
    return f"{OUTROOT}/{GENE}/{setting}/per_mu/{GENE}_{setting}_mu_{mu}.json"


wildcard_constraints:
    setting = "|".join(ALL_SETTINGS),
    mu      = r"[0-9.]+e[+-][0-9]+",
    chunk   = r"\d+",

localrules: all


# — 6. Targets ——————————————————————————————————————————————
rule all:
    input:
        [f"{OUTROOT}/{GENE}/{GENE}_{st}_varying_mu.npz" for st in ALL_SETTINGS],
        [f"{OUTROOT}/{GENE}/{GENE}_{st}_varying_mu.json" for st in ALL_SETTINGS],


# — 7. Simulation, chunked ————————————————————————————————————
# Usage: comp_het_lineage_tracker RUNS sel DOM mutU ufactor seed
#                                     allow_back_mut burn_in pop burnin_Ne
rule simulate_sas_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{GENE}/sas/mu_{{mu}}_chunk{{chunk}}.out"),
    params:
        binary    = SETTINGS["sas"]["binary"],
        s         = S_VALUE,
        h_mut     = H_MUT,
        runs      = lambda wc: chunk_runs("sas", wc.mu, wc.chunk),
        ufactor   = UFACTOR,
        seed      = lambda wc: seed_for("sas", wc.mu, wc.chunk),
        back_mut  = ALLOW_BACK_MUT,
        burn_in   = BURN_IN_COMPHET,
        pop_label = "sas",
        burnin_Ne = BURNIN_NE,
    resources:
        partition     = "short",
        time          = SETTINGS["sas"]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = SETTINGS["sas"]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        # stdbuf -o 4M: the simulator writes through std::cout, whose default
        # stdio buffer is 8 KiB, so a multi-GB .out becomes ~10^6 tiny write()
        # syscalls per job. A 4 MB buffer makes them ~500x fewer and larger,
        # which is what Lustre wants. Output bytes are byte-for-byte identical.
        stdbuf -o 4M {params.binary} \
          {params.runs} {params.s} {params.h_mut} {wildcards.mu} \
          {params.ufactor} {params.seed} {params.back_mut} {params.burn_in} \
          {params.pop_label} {params.burnin_Ne} \
          > {output.chunk_out}
        """


# Usage: comp_het_lineage_tracker_constN RUNS N G sel DOM mutU ufactor seed allow_back_mut
rule simulate_const1M_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{GENE}/const1M/mu_{{mu}}_chunk{{chunk}}.out"),
    params:
        binary   = SETTINGS["const1M"]["binary"],
        Ne       = NE_CONST,
        burn_in  = BURN_IN_CONSTN,
        s        = S_VALUE,
        h_mut    = H_MUT,
        runs     = lambda wc: chunk_runs("const1M", wc.mu, wc.chunk),
        ufactor  = UFACTOR,
        seed     = lambda wc: seed_for("const1M", wc.mu, wc.chunk),
        back_mut = ALLOW_BACK_MUT,
    resources:
        partition     = "short",
        time          = SETTINGS["const1M"]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = SETTINGS["const1M"]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        stdbuf -o 4M {params.binary} \
          {params.runs} {params.Ne} {params.burn_in} {params.s} \
          {params.h_mut} {wildcards.mu} {params.ufactor} {params.seed} \
          {params.back_mut} \
          > {output.chunk_out}
        """


# — 8. Merge the chunks of one (setting, mu) ——————————————————
rule merge_chunks:
    input:
        chunks = lambda wc: expand(
            f"{TMPROOT}/{GENE}/{{setting}}/mu_{{mu}}_chunk{{chunk}}.out",
            setting=wc.setting, mu=wc.mu, chunk=chunk_list(wc.setting, wc.mu),
        ),
    output:
        out         = maybe_temp(f"{OUTROOT}/{GENE}/{{setting}}/mu_{{mu}}.out"),
        params_file = f"{OUTROOT}/{GENE}/{{setting}}/mu_{{mu}}.params.txt",
    params:
        binary      = lambda wc: SETTINGS[wc.setting]["binary"],
        population  = lambda wc: SETTINGS[wc.setting]["population"],
        runs        = lambda wc: SETTINGS[wc.setting]["runs"],
        n_chunks    = lambda wc: n_chunks(wc.setting, wc.mu),
        seed        = lambda wc: seed_for(wc.setting, wc.mu, 0),
        s           = S_VALUE,
        h_mut       = H_MUT,
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "1000M",
    shell:
        r"""
        mkdir -p $(dirname {output.out})
        cat {input.chunks} > {output.out}

        {{
          echo "gene={GENE}"
          echo "kind=compound_het"
          echo "setting={wildcards.setting}"
          echo "population={params.population}"
          echo "simulator={params.binary}"
          echo "gene_mut_rate={wildcards.mu}"
          echo "variant_mut_rate={MU_VARIANT}"
          echo "mu_gene_estimate={MU_GENE}"
          echo "s={params.s}"
          echo "h_mutator={params.h_mut}"
          echo "phi={XPC_PHI}"
          echo "phi_scope={XPC_PHI_SCOPE}"
          echo "h_deleterious={H_DEL}"
          echo "s_deleterious={S_DEL}"
          echo "genome_size={G_SIZE}"
          echo "fraction_selected={F_SEL}"
          echo "runs={params.runs}"
          echo "n_chunks={params.n_chunks}"
          echo "base_random_seed={params.seed}"
          echo "ufactor={UFACTOR}"
          echo "allow_back_mut={ALLOW_BACK_MUT}"
        }} > {output.params_file}
        """


# — 9. Summarise one (setting, mu), keyed by mu ————————————————
# --gene-mu IS the wildcard: the same rate that was simulated is the
# denominator of the lineage->variant assignment probability.
rule summarize_one_mu:
    input:
        script = f"{SCRIPTS}/summarize_comp_het.py",
        out = f"{OUTROOT}/{GENE}/{{setting}}/mu_{{mu}}.out",
    output:
        npz  = f"{OUTROOT}/{GENE}/{{setting}}/per_mu/{GENE}_{{setting}}_mu_{{mu}}.npz",
        json = f"{OUTROOT}/{GENE}/{{setting}}/per_mu/{GENE}_{{setting}}_mu_{{mu}}.json",
    params:
        population = lambda wc: SETTINGS[wc.setting]["population"],
        runs       = lambda wc: SETTINGS[wc.setting]["runs"],
        s          = S_VALUE,
        h_mut      = H_MUT,
    resources:
        partition     = "short",
        time          = "'02:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: SETTINGS[wc.setting]["summary_mem"],
    shell:
        r"""
        python3 {input.script} {input.out} \
          --gene-mu {wildcards.mu} \
          --variant {GENE}:{MU_VARIANT} \
          --npz-key-style bare \
          --key {wildcards.mu} \
          --out-json {output.json} --out-npz {output.npz} \
          --gene {GENE} --population {params.population} \
          --meta setting={wildcards.setting} \
          --meta gene_mut_rate={wildcards.mu} \
          --meta variant_mut_rate={MU_VARIANT} \
          --meta mu_gene_estimate={MU_GENE} \
          --meta s={params.s} \
          --meta h_mutator={params.h_mut} \
          --meta phi={XPC_PHI} \
          --meta runs={params.runs} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


# — 10. One npz per setting, keyed by mu ————————————————————————
rule combine_mu:
    input:
        script = f"{SCRIPTS}/combine_summaries.py",
        npz = lambda wc: [per_mu_npz(wc.setting, mu) for mu in MU_VALUES],
        jsn = lambda wc: [per_mu_json(wc.setting, mu) for mu in MU_VALUES],
    output:
        npz  = f"{OUTROOT}/{GENE}/{GENE}_{{setting}}_varying_mu.npz",
        json = f"{OUTROOT}/{GENE}/{GENE}_{{setting}}_varying_mu.json",
    resources:
        partition     = "short",
        time          = "'02:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "16000M",
    shell:
        r"""
        python3 {input.script} {input.npz} \
          --out-npz {output.npz} --out-json {output.json}
        """
