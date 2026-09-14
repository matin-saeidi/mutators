# ============================================================================
# Varying-dominance simulations for the two presumed-recessive mutators, MPG and
# XPC, under the SOUTH ASIAN demographic history.
#
# For each gene, h_m is swept over 41 values from 0 to 1, at THREE selection
# coefficients: s_center/10, s_center, and 10*s_center, where s_center is the
# value implied by that mutator's effect size (the one used in the constant-phi
# and varying-s runs).
#
# Final product per gene: THREE npz files, one per s value, each keyed by h.
#     MPG_sas_varying_h_smaller_s.npz  ->  "0.000", "0.025", ... "1.000"
#     MPG_sas_varying_h_center_s.npz
#     MPG_sas_varying_h_larger_s.npz
#   and the same three for XPC.
#
# Run from the repository root:
#   tmux new -s sasvh
#   cd <repo root>
#   snakemake --snakefile workflow/varying_dominance.smk \
#             --profile . 2>&1 | tee logs/varying-h-tmux.log
# or to inspect first:
#   snakemake --snakefile workflow/varying_dominance.smk -np
# ============================================================================

import math
import numpy as np
import os

# — 0. Which demographic history —————————————————————————————————
POP = config.get("pop", "sas")   # "sas", "eur" or "afr"

# — 1. Paths ————————————————————————————————————————————————
# Every path is derived from where this file sits, so a clone runs anywhere.
# Send results elsewhere with:  --config results_dir=/scratch/you/mutators
from pathlib import Path

REPO    = Path(workflow.basedir).parent.resolve()
RESULTS = Path(config.get("results_dir", REPO / "results")).resolve()
SCRIPTS = str(REPO / "src" / "summaries")
SIMBIN  = REPO / "src" / "simulations" / "bin"

OUTROOT  = f"{RESULTS}/{POP}/simulations_varying_h"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
SINGLE_SITE_BIN = str(SIMBIN / "single_site_simulator")
COMP_HET_BIN    = str(SIMBIN / "comp_het_lineage_tracker")

# — 1b. Disk policy —————————————————————————————————————————
# One XPC .out at 5e5 runs is ~8 GB, and there are 123 (h, s) combinations per
# gene, so keeping every raw file would be well over a terabyte. Each (h, s) is
# summarised on its own and its merged .out released immediately, capping disk
# at whatever is in flight. Set True to keep them -- budget ~1.5 TB first.
KEEP_RAW_OUT = False
maybe_temp = (lambda p: p) if KEEP_RAW_OUT else temp

# — 2. Constants shared by every mutator ————————————————————————
# h/s here describe the deleterious mutations the mutator causes, NOT the
# mutator's own dominance, which is the thing being swept.
H_DEL   = 0.5
S_DEL   = 0.001
G_SIZE  = 3e9
F_SEL   = 0.08

# — the dominance grid —
H_VALUES = ["{:.3f}".format(h) for h in np.linspace(0, 1, 41)]

# — the three selection coefficients —
S_LABELS = {"smaller_s": 0.1, "center_s": 1.0, "larger_s": 10.0}
S_FMT    = "{:.6f}"

BASE_SEED = 10

# single-site simulator: mut_uncert=0, dem_uncert=0, demographic_model=1
OTHER_ARGS      = "0 0 1"
NE_CONST        = "1e6"    # ignored when demographic_model = 1
BURN_IN_SINGLE  = "2.5e5"  # ignored when demographic_model = 1 (code forces 250000)

# compound-het simulator
BURN_IN_COMPHET = "2.5e5"
BURNIN_NE       = "20000"
UFACTOR         = "1.0"
ALLOW_BACK_MUT  = "1"

# Chunks are sized so each is ~CHUNK_TARGET_H hours nominal, which survives the
# ~10x slowdown seen on contended nodes inside the 11:59:00 limit.
CHUNK_TARGET_H = 1.2
MAX_CHUNKS     = 60


# — 3. The two recessive mutators ————————————————————————————
# cost[s_label] = measured seconds/run at h = 0.0, 0.5, 1.0 under this history.
# Cost falls steeply with h (selection acts on heterozygotes, so the allele is
# lost sooner) and with s, so chunk counts are computed per (s, h) rather than
# fixed at the worst corner -- that alone halves the XPC job count.
GENES = {
    "MPG": dict(
        kind="single", mut_rate="1.0353e-07",
        phi=4.2379e-08, phi_scope="hom", runs=1000000,
        cost={"smaller_s": [0.02048, 0.01544, 0.01462],
              "center_s":  [0.01730, 0.01252, 0.01175],
              "larger_s":  [0.01503, 0.01016, 0.00967]},
        walltime="'11:59:00'", mem="500M",
    ),
    "XPC": dict(
        kind="comphet", mut_rate="4.4355e-06",
        phi=7.7921e-08, phi_scope="hom", runs=500000,
        cost={"smaller_s": [0.4053, 0.2853, 0.2453],
              "center_s":  [0.3433, 0.1687, 0.1367],
              "larger_s":  [0.2673, 0.0720, 0.0547]},
        walltime="'11:59:00'", mem="1000M",
        variants=[("XPC", "7.922075e-08")],
        summary_gene_mu="4.4355e-06",
        npz_key_style="bare",     # single variant -> key by h alone
    ),
}

SINGLE_GENES  = sorted(g for g, c in GENES.items() if c["kind"] == "single")
COMPHET_GENES = sorted(g for g, c in GENES.items() if c["kind"] == "comphet")
ALL_GENES     = SINGLE_GENES + COMPHET_GENES


def s_center(gene):
    c = GENES[gene]
    phi_hom = 2.0 * c["phi"] if c["phi_scope"] == "het" else c["phi"]
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


def s_value(gene, slab):
    return S_FMT.format(S_LABELS[slab] * s_center(gene))


# s strings must stay distinct, otherwise two runs would collide on disk
for _g in ALL_GENES:
    _vals = [s_value(_g, _l) for _l in S_LABELS]
    if len(set(_vals)) != len(_vals):
        raise ValueError(f"{_g}: s values collide after formatting with {S_FMT}: {_vals}")

H_INDEX = {h: i for i, h in enumerate(H_VALUES)}
SEED_OFFSET = {g: 1000000 * (i + 1) for i, g in enumerate(ALL_GENES)}
SLAB_OFFSET = {l: 100000 * i for i, l in enumerate(S_LABELS)}


def cost_per_run(gene, slab, h):
    """Interpolate the measured seconds/run at this h."""
    return float(np.interp(float(h), [0.0, 0.5, 1.0], GENES[gene]["cost"][slab]))


def n_chunks(gene, slab, h):
    total_s = GENES[gene]["runs"] * cost_per_run(gene, slab, h)
    return max(1, min(MAX_CHUNKS, math.ceil(total_s / (CHUNK_TARGET_H * 3600))))


def chunk_list(gene, slab, h):
    return list(range(n_chunks(gene, slab, h)))


def chunk_runs(gene, slab, h, chunk):
    """Runs in one chunk; the chunks sum to exactly runs."""
    n = n_chunks(gene, slab, h)
    base, rem = divmod(GENES[gene]["runs"], n)
    return base + (1 if int(chunk) < rem else 0)


def seed_for(gene, slab, h, chunk):
    return (BASE_SEED + SEED_OFFSET[gene] + SLAB_OFFSET[slab]
            + 100 * H_INDEX[h] + int(chunk))


def variant_args(gene):
    return " ".join(f"--variant {n}:{mu}" for n, mu in GENES[gene]["variants"])


def per_h_npz(gene, slab, h):
    return f"{OUTROOT}/{gene}/{slab}/per_h/{gene}_{slab}_h_{h}.npz"


def per_h_json(gene, slab, h):
    return f"{OUTROOT}/{gene}/{slab}/per_h/{gene}_{slab}_h_{h}.json"


wildcard_constraints:
    gene  = "|".join(ALL_GENES),
    slab  = "|".join(S_LABELS),
    h     = r"[0-9.]+",
    chunk = r"\d+",

localrules: all


# — 4. Targets ——————————————————————————————————————————————
rule all:
    input:
        [f"{OUTROOT}/{g}/{g}_{POP}_varying_h_{l}.npz"
         for g in ALL_GENES for l in S_LABELS],
        [f"{OUTROOT}/{g}/{g}_{POP}_varying_h_{l}.json"
         for g in ALL_GENES for l in S_LABELS],


# — 5. Simulation, chunked ————————————————————————————————————
rule simulate_single_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/{{slab}}/h_{{h}}_chunk{{chunk}}.out"),
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        binary     = SINGLE_SITE_BIN,
        mut_rate   = lambda wc: GENES[wc.gene]["mut_rate"],
        s          = lambda wc: s_value(wc.gene, wc.slab),
        runs       = lambda wc: chunk_runs(wc.gene, wc.slab, wc.h, wc.chunk),
        other_args = OTHER_ARGS,
        Ne         = NE_CONST,
        seed       = lambda wc: seed_for(wc.gene, wc.slab, wc.h, wc.chunk),
        burn_in    = BURN_IN_SINGLE,
        pop_label  = POP,
    resources:
        partition     = "short",
        time          = lambda wc: GENES[wc.gene]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: GENES[wc.gene]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        # stdbuf -o 4M: the simulator writes through std::cout, whose default
        # stdio buffer is 8 KiB, so a multi-GB .out becomes ~10^6 tiny write()
        # syscalls per job. A 4 MB buffer makes them ~500x fewer and larger,
        # which is what Lustre wants. Output bytes are byte-for-byte identical.
        stdbuf -o 4M {params.binary} \
          {params.mut_rate} {params.s} {wildcards.h} {params.runs} \
          {params.other_args} {params.Ne} {params.seed} {params.burn_in} \
          {params.pop_label} \
          > {output.chunk_out}
        """


rule simulate_comphet_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/{{slab}}/h_{{h}}_chunk{{chunk}}.out"),
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        binary    = COMP_HET_BIN,
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        s         = lambda wc: s_value(wc.gene, wc.slab),
        runs      = lambda wc: chunk_runs(wc.gene, wc.slab, wc.h, wc.chunk),
        ufactor   = UFACTOR,
        seed      = lambda wc: seed_for(wc.gene, wc.slab, wc.h, wc.chunk),
        back_mut  = ALLOW_BACK_MUT,
        burn_in   = BURN_IN_COMPHET,
        pop_label = POP,
        burnin_Ne = BURNIN_NE,
    resources:
        partition     = "short",
        time          = lambda wc: GENES[wc.gene]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: GENES[wc.gene]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        # stdbuf -o 4M: the simulator writes through std::cout, whose default
        # stdio buffer is 8 KiB, so a multi-GB .out becomes ~10^6 tiny write()
        # syscalls per job. A 4 MB buffer makes them ~500x fewer and larger,
        # which is what Lustre wants. Output bytes are byte-for-byte identical.
        stdbuf -o 4M {params.binary} \
          {params.runs} {params.s} {wildcards.h} {params.mut_rate} \
          {params.ufactor} {params.seed} {params.back_mut} {params.burn_in} \
          {params.pop_label} {params.burnin_Ne} \
          > {output.chunk_out}
        """


# — 6. Merge the chunks of one (gene, s, h) ————————————————————
rule merge_chunks:
    input:
        chunks = lambda wc: expand(
            f"{TMPROOT}/{{gene}}/{{slab}}/h_{{h}}_chunk{{chunk}}.out",
            gene=wc.gene, slab=wc.slab, h=wc.h,
            chunk=chunk_list(wc.gene, wc.slab, wc.h),
        ),
    output:
        out         = maybe_temp(f"{OUTROOT}/{{gene}}/{{slab}}/h_{{h}}.out"),
        params_file = f"{OUTROOT}/{{gene}}/{{slab}}/h_{{h}}.params.txt",
    params:
        kind      = lambda wc: GENES[wc.gene]["kind"],
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        s         = lambda wc: s_value(wc.gene, wc.slab),
        s_centre  = lambda wc: S_FMT.format(s_center(wc.gene)),
        s_mult    = lambda wc: S_LABELS[wc.slab],
        runs      = lambda wc: GENES[wc.gene]["runs"],
        n_chunks  = lambda wc: n_chunks(wc.gene, wc.slab, wc.h),
        seed      = lambda wc: seed_for(wc.gene, wc.slab, wc.h, 0),
        pop_label = POP,
        phi       = lambda wc: GENES[wc.gene]["phi"],
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
          echo "gene={wildcards.gene}"
          echo "kind={params.kind}"
          echo "population={params.pop_label}"
          echo "h_mutator={wildcards.h}"
          echo "s={params.s}"
          echo "s_centre={params.s_centre}"
          echo "s_multiplier={params.s_mult}"
          echo "s_label={wildcards.slab}"
          echo "mut_rate={params.mut_rate}"
          echo "phi={params.phi}"
          echo "h_deleterious={H_DEL}"
          echo "s_deleterious={S_DEL}"
          echo "genome_size={G_SIZE}"
          echo "fraction_selected={F_SEL}"
          echo "runs={params.runs}"
          echo "n_chunks={params.n_chunks}"
          echo "base_random_seed={params.seed}"
        }} > {output.params_file}
        """


# — 7. Summarise one (gene, s, h), keyed by h ——————————————————
rule summarize_single_one_h:
    input:
        script = f"{SCRIPTS}/summarize_single_site.py",
        out = f"{OUTROOT}/{{gene}}/{{slab}}/h_{{h}}.out",
    output:
        npz  = f"{OUTROOT}/{{gene}}/{{slab}}/per_h/{{gene}}_{{slab}}_h_{{h}}.npz",
        json = f"{OUTROOT}/{{gene}}/{{slab}}/per_h/{{gene}}_{{slab}}_h_{{h}}.json",
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        pop_label = POP,
        s         = lambda wc: s_value(wc.gene, wc.slab),
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "8000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --key {wildcards.h} \
          --out-json {output.json} --out-npz {output.npz} \
          --gene {wildcards.gene} --population {params.pop_label} \
          --meta h_mutator={wildcards.h} \
          --meta s={params.s} \
          --meta s_label={wildcards.slab} \
          --meta mut_rate={params.mut_rate} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


rule summarize_comphet_one_h:
    input:
        script = f"{SCRIPTS}/summarize_comp_het.py",
        out = f"{OUTROOT}/{{gene}}/{{slab}}/h_{{h}}.out",
    output:
        npz  = f"{OUTROOT}/{{gene}}/{{slab}}/per_h/{{gene}}_{{slab}}_h_{{h}}.npz",
        json = f"{OUTROOT}/{{gene}}/{{slab}}/per_h/{{gene}}_{{slab}}_h_{{h}}.json",
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        pop_label = POP,
        gene_mu   = lambda wc: GENES[wc.gene]["summary_gene_mu"],
        variants  = lambda wc: variant_args(wc.gene),
        keystyle  = lambda wc: GENES[wc.gene]["npz_key_style"],
        s         = lambda wc: s_value(wc.gene, wc.slab),
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
    resources:
        partition     = "short",
        time          = "'02:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "16000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --gene-mu {params.gene_mu} {params.variants} \
          --npz-key-style {params.keystyle} --no-ages \
          --key {wildcards.h} \
          --out-json {output.json} --out-npz {output.npz} \
          --gene {wildcards.gene} --population {params.pop_label} \
          --meta h_mutator={wildcards.h} \
          --meta s={params.s} \
          --meta s_label={wildcards.slab} \
          --meta simulator_mut_rate={params.mut_rate} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


# — 8. One npz per (gene, s value), keyed by h ——————————————————
rule combine_h:
    input:
        script = f"{SCRIPTS}/combine_summaries.py",
        npz = lambda wc: [per_h_npz(wc.gene, wc.slab, h) for h in H_VALUES],
        jsn = lambda wc: [per_h_json(wc.gene, wc.slab, h) for h in H_VALUES],
    output:
        npz  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_varying_h_{{slab}}.npz",
        json = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_varying_h_{{slab}}.json",
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
