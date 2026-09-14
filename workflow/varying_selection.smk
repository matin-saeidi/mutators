# ============================================================================
# Varying-selection-coefficient simulations for all 7 mutator alleles under the
# SOUTH ASIAN demographic history.
#
# For each mutator, s is swept on a log grid from s_center/50 to 50*s_center,
# where s_center is that mutator's empirically estimated selection coefficient
# (the value used in the constant-phi_G runs). A neutral s = 0 point is
# prepended, matching the existing per-gene hms*_grid Snakefiles.
#
# One job graph covering all five simulations, plus the summarising:
#   single site  : MPG, POLD1, POLE      -> single_site_simulator
#   compound het : XPC, MUTYH            -> comp_het_lineage_tracker
#
# Final product per gene: ONE npz keyed by selection coefficient.
#   MPG / POLD1 / POLE / XPC : "<s>"                       e.g. "0.01017096"
#   MUTYH                    : "<variant>_freq_s=<s>"      e.g. "Y179C_freq_s=0.00199795"
# MUTYH keeps the variant prefix because its three variants share one
# simulation, so a bare "<s>" key could not say which variant it refers to.
#
# THE SAVED DISTRIBUTIONS ARE UNCONDITIONAL. Every simulated replicate reaches
# the npz, a replicate in which the mutator was lost entering as a frequency of
# exactly 0, so every array has one entry per run and MUTYH's three arrays are
# aligned run-for-run at each s. Conditioning is a downstream filter, not
# something baked in here:
#
#     q = np.load(npz)["0.01017096"];  q[q > 0]     # one mutator, segregating
#     keep = (y > 0) & (v > 0) & (g > 0)             # MUTYH, all three
#
# so one grid serves both the conditional and the unconditional analysis.
#
# Run from the repository root:
#   tmux new -s sasvs
#   cd <repo root>
#   snakemake --snakefile workflow/varying_selection.smk \
#             --profile . 2>&1 | tee logs/varying-s-tmux.log
# or, to inspect first:
#   snakemake --snakefile workflow/varying_selection.smk -np
#
# Setting POP = "eur"/"afr" reruns the whole grid under another history.
# ============================================================================

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

OUTROOT  = f"{RESULTS}/{POP}/simulations_varying_s"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
SINGLE_SITE_BIN = str(SIMBIN / "single_site_simulator")
COMP_HET_BIN    = str(SIMBIN / "comp_het_lineage_tracker")

# — 1b. Disk policy —————————————————————————————————————————
# The raw .out for one XPC s value is ~8 GB, so the full grid would be ~380 GB
# (XPC alone ~340 GB) held permanently. Each s value is therefore summarised on
# its own and its merged .out is released as soon as that summary exists, which
# caps disk at whatever is in flight rather than the total. Set this True to
# keep every raw .out instead -- budget ~380 GB before you do.
KEEP_RAW_OUT = False
maybe_temp = (lambda p: p) if KEEP_RAW_OUT else temp

# — 2. Constants shared by every mutator ————————————————————————
# h/s here describe the deleterious mutations the mutator causes, NOT the
# mutator allele itself (that dominance is per-gene, h_mut below).
H_DEL   = 0.5
S_DEL   = 0.001
G_SIZE  = 3e9
F_SEL   = 0.08

# — the selection grid —
S_SPAN          = 50     # s_center/50 .. 50*s_center
N_S_POINTS      = 41     # log-spaced points across that span
INCLUDE_NEUTRAL = True   # prepend s = 0
S_FMT           = "{:.8f}"

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


def selection_coefficient(phi, phi_scope):
    """Selection coefficient of the mutator in a HOMOZYGOTE."""
    phi_hom = 2.0 * phi if phi_scope == "het" else phi
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


# — 3. The seven mutators ————————————————————————————————————
# runs / chunks are sized from measured per-run costs at the SLOWEST point of
# the grid (small s), so a chunk is ~1.2 h nominal and survives the ~10x
# slowdown seen on contended nodes inside the 11:59:00 limit.
GENES = {
    "MPG": dict(
        kind="single", mut_rate="1.0353e-07", h_mut="0",
        phi=4.2379e-08, phi_scope="hom",
        runs=1000000, chunks=6,          # 0.0217 s/run -> ~6.0 h per s value
        walltime="'11:59:00'", mem="500M",
    ),
    "POLD1": dict(
        kind="single", mut_rate="7.0543E-09", h_mut="0.5",
        phi=1.9211e-08, phi_scope="het",
        runs=1000000, chunks=3,          # 0.0095 s/run -> ~2.6 h per s value
        walltime="'11:59:00'", mem="500M",
    ),
    "POLE": dict(
        kind="single", mut_rate="3.7048E-09", h_mut="0.5",
        phi=3.9161e-08, phi_scope="het",
        runs=1000000, chunks=3,          # 0.0090 s/run -> ~2.5 h per s value
        walltime="'11:59:00'", mem="500M",
    ),
    "XPC": dict(
        kind="comphet", mut_rate="4.4355e-06", h_mut="0",
        phi=7.7921e-08, phi_scope="hom",
        runs=500000, chunks=52,          # 0.445 s/run -> ~62 h per s value
        walltime="'11:59:00'", mem="1000M",
        variants=[("XPC", "7.922075e-08")],
        summary_gene_mu="4.4355e-06",
        npz_key_style="bare",            # single variant -> key by s alone
    ),
    "MUTYH": dict(
        kind="comphet", mut_rate="2.9029E-08", h_mut="0",
        phi=8.3248e-09, phi_scope="hom",
        # Every replicate is saved, so the run count is sized by the most
        # demanding analysis the npz has to support: the one conditioning on
        # all three variants segregating. That subset is ~12.5% of runs and,
        # unlike the single-site genes, its size is flat across the whole s
        # grid -- with h_mut = 0 selection acts only on homozygotes, which are
        # vanishingly rare at these frequencies. 4e6 runs therefore leaves
        # ~500k triple-segregating runs at every s.
        runs=4000000, chunks=33,         # 0.0348 s/run -> ~39 h per s value
        walltime="'11:59:00'", mem="1000M",
        variants=[("Y179C", "1.0049e-08"),
                  ("V234M", "1.1317e-08"),
                  ("G368D", "7.6633e-09")],
        summary_gene_mu="sum",
        npz_key_style="prefixed",        # three variants -> must name them
    ),
}

SINGLE_GENES  = sorted(g for g, c in GENES.items() if c["kind"] == "single")
COMPHET_GENES = sorted(g for g, c in GENES.items() if c["kind"] == "comphet")
ALL_GENES     = SINGLE_GENES + COMPHET_GENES


def s_grid(gene):
    """Log-spaced s values for one gene, as formatted strings."""
    c = selection_coefficient(GENES[gene]["phi"], GENES[gene]["phi_scope"])
    v = np.logspace(np.log10(c / S_SPAN), np.log10(S_SPAN * c), N_S_POINTS)
    if INCLUDE_NEUTRAL:
        v = np.insert(v, 0, 0.0)
    out = [S_FMT.format(x) for x in v]
    if len(set(out)) != len(out):
        raise ValueError(f"{gene}: s values collide after formatting with {S_FMT}")
    return out


S_VALUES  = {g: s_grid(g) for g in ALL_GENES}
S_INDEX   = {g: {s: i for i, s in enumerate(S_VALUES[g])} for g in ALL_GENES}
S_CENTER  = {g: selection_coefficient(GENES[g]["phi"], GENES[g]["phi_scope"]) for g in ALL_GENES}

# distinct seed block per gene; within a gene, per (s, chunk)
SEED_OFFSET = {g: 1000000 * (i + 1) for i, g in enumerate(ALL_GENES)}


def gene_runs(gene):
    return GENES[gene]["runs"]


def chunk_runs(gene, chunk):
    """Runs in one chunk; the chunks sum to exactly gene_runs(gene)."""
    n = GENES[gene]["chunks"]
    base, rem = divmod(gene_runs(gene), n)
    return base + (1 if int(chunk) < rem else 0)


def chunk_list(gene):
    return list(range(GENES[gene]["chunks"]))


def seed_for(gene, s, chunk):
    return BASE_SEED + SEED_OFFSET[gene] + 1000 * S_INDEX[gene][s] + int(chunk)


def variant_args(gene):
    return " ".join(f"--variant {n}:{mu}" for n, mu in GENES[gene]["variants"])


def per_s_npz(gene, s):
    return f"{OUTROOT}/{gene}/per_s/{gene}_s_{s}.npz"


def per_s_json(gene, s):
    return f"{OUTROOT}/{gene}/per_s/{gene}_s_{s}.json"


wildcard_constraints:
    gene  = "|".join(ALL_GENES),
    s     = r"[0-9.]+",
    chunk = r"\d+",

localrules: all


# — 4. Targets ——————————————————————————————————————————————
rule all:
    input:
        [f"{OUTROOT}/{g}/{g}_{POP}_varying_s.npz"  for g in ALL_GENES],
        [f"{OUTROOT}/{g}/{g}_{POP}_varying_s.json" for g in ALL_GENES],


# — 5. Simulation, chunked ————————————————————————————————————
rule simulate_single_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out"),
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        binary     = SINGLE_SITE_BIN,
        mut_rate   = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut      = lambda wc: GENES[wc.gene]["h_mut"],
        runs       = lambda wc: chunk_runs(wc.gene, wc.chunk),
        other_args = OTHER_ARGS,
        Ne         = NE_CONST,
        seed       = lambda wc: seed_for(wc.gene, wc.s, wc.chunk),
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
          {params.mut_rate} {wildcards.s} {params.h_mut} {params.runs} \
          {params.other_args} {params.Ne} {params.seed} {params.burn_in} \
          {params.pop_label} \
          > {output.chunk_out}
        """


rule simulate_comphet_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out"),
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        binary    = COMP_HET_BIN,
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
        runs      = lambda wc: chunk_runs(wc.gene, wc.chunk),
        ufactor   = UFACTOR,
        seed      = lambda wc: seed_for(wc.gene, wc.s, wc.chunk),
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
          {params.runs} {wildcards.s} {params.h_mut} {params.mut_rate} \
          {params.ufactor} {params.seed} {params.back_mut} {params.burn_in} \
          {params.pop_label} {params.burnin_Ne} \
          > {output.chunk_out}
        """


# — 6. Merge the chunks of one s value ————————————————————————
rule merge_chunks:
    input:
        chunks = lambda wc: expand(
            f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out",
            gene=wc.gene, s=wc.s, chunk=chunk_list(wc.gene),
        ),
    output:
        out         = maybe_temp(f"{OUTROOT}/{{gene}}/s_{{s}}.out"),
        params_file = f"{OUTROOT}/{{gene}}/s_{{s}}.params.txt",
    params:
        kind       = lambda wc: GENES[wc.gene]["kind"],
        mut_rate   = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut      = lambda wc: GENES[wc.gene]["h_mut"],
        runs       = lambda wc: gene_runs(wc.gene),
        n_chunks   = lambda wc: GENES[wc.gene]["chunks"],
        seed       = lambda wc: seed_for(wc.gene, wc.s, 0),
        pop_label  = POP,
        phi        = lambda wc: GENES[wc.gene]["phi"],
        phi_scope  = lambda wc: GENES[wc.gene]["phi_scope"],
        s_center   = lambda wc: "{:.8f}".format(S_CENTER[wc.gene]),
        s_index    = lambda wc: S_INDEX[wc.gene][wc.s],
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
          echo "s={wildcards.s}"
          echo "s_center={params.s_center}"
          echo "s_index={params.s_index}"
          echo "mut_rate={params.mut_rate}"
          echo "h_mutator={params.h_mut}"
          echo "phi={params.phi}"
          echo "phi_scope={params.phi_scope}"
          echo "h_deleterious={H_DEL}"
          echo "s_deleterious={S_DEL}"
          echo "genome_size={G_SIZE}"
          echo "fraction_selected={F_SEL}"
          echo "runs={params.runs}"
          echo "n_chunks={params.n_chunks}"
          echo "base_random_seed={params.seed}"
        }} > {output.params_file}
        """


# — 7. Summarise one s value ————————————————————————————————
rule summarize_single_one_s:
    input:
        script = f"{SCRIPTS}/summarize_single_site.py",
        out = f"{OUTROOT}/{{gene}}/s_{{s}}.out",
    output:
        npz  = f"{OUTROOT}/{{gene}}/per_s/{{gene}}_s_{{s}}.npz",
        json = f"{OUTROOT}/{{gene}}/per_s/{{gene}}_s_{{s}}.json",
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        pop_label = POP,
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "8000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --out-json {output.json} --out-npz {output.npz} \
          --gene {wildcards.gene} --population {params.pop_label} \
          --meta mut_rate={params.mut_rate} \
          --meta h_mutator={params.h_mut} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


rule summarize_comphet_one_s:
    input:
        script = f"{SCRIPTS}/summarize_comp_het.py",
        out = f"{OUTROOT}/{{gene}}/s_{{s}}.out",
    output:
        npz  = f"{OUTROOT}/{{gene}}/per_s/{{gene}}_s_{{s}}.npz",
        json = f"{OUTROOT}/{{gene}}/per_s/{{gene}}_s_{{s}}.json",
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        pop_label = POP,
        gene_mu   = lambda wc: GENES[wc.gene]["summary_gene_mu"],
        variants  = lambda wc: variant_args(wc.gene),
        keystyle  = lambda wc: GENES[wc.gene]["npz_key_style"],
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
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
          --out-json {output.json} --out-npz {output.npz} \
          --gene {wildcards.gene} --population {params.pop_label} \
          --meta simulator_mut_rate={params.mut_rate} \
          --meta h_mutator={params.h_mut} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


# — 8. Stitch the per-s summaries into one npz per gene ————————
rule combine_grid:
    input:
        script = f"{SCRIPTS}/combine_summaries.py",
        npz = lambda wc: [per_s_npz(wc.gene, s) for s in S_VALUES[wc.gene]],
        jsn = lambda wc: [per_s_json(wc.gene, s) for s in S_VALUES[wc.gene]],
    output:
        npz  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_varying_s.npz",
        json = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_varying_s.json",
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
