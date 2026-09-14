# ============================================================================
# AGE DISTRIBUTION of every mutator allele under the SOUTH ASIAN demography.
#
# Companion to Snakefile_SAS_dem_const_phiG_all_genes, which gives the
# *frequency* distributions. Same demography, same s, same mutation rates --
# what changes is that EVERY mutator goes through the compound-het lineage
# tracker, because that is the only simulator that records lineage ages:
#
#   age of the mutator in one replicate
#       = age of the OLDEST surviving lineage assigned to that mutator
#
# For MPG, POLD1 and POLE the rate handed to the simulator is the SINGLE-VARIANT
# rate, so mu_variant/mu_gene = 1 and every lineage is the focal variant: the
# run's age is simply the oldest surviving lineage. This is what the old
# Snakefile_comp_het_{POLE,POLD1}_Sch_dem_for_aging and
# Snakefile_comp_het_MPG_var_for_aging did, only under the SAS history and
# inside one job graph with the rest.
#
# For XPC and MUTYH the simulated allele is the gene-wide LoF allele, so each
# surviving lineage is first assigned to a focal variant with probability
# mu_variant/mu_gene and the oldest lineage ASSIGNED TO THE FOCAL VARIANT is
# taken. XPC keeps ~1.8% of lineages; MUTYH's three variants exhaust the gene
# rate, so every lineage lands on one of them.
#
# Conditioning is PER VARIANT: a run enters a variant's age distribution iff at
# least one lineage in it was assigned to that variant. Age has no value in a run
# where the variant is absent, so unlike the frequency pipeline this one cannot
# simply keep every replicate -- see the docstring of summarize_ages.py.
#
# 7 mutators (MUTYH is three variants of one simulation) from 5 selected
# simulations, plus 6 neutral s = 0 controls: one at each mutator's own mutation
# rate, so that only s differs between a mutator and its control, plus one at the
# human per-site rate 1.25e-08.
#
# Final product:
#   simulations_ages/<label>/<label>_sas_ages.{json,npz}   per mutator
#   simulations_ages/all_mutators_sas_ages.{npz,json,tsv}  everything, combined
#
# Run from the repository root:
#   snakemake --snakefile workflow/mutator_ages.smk -np
#   tmux new -s sasages
#   cd <repo root>
#   snakemake --snakefile workflow/mutator_ages.smk \
#             --profile . 2>&1 | tee logs/ages-tmux.log
#
# Setting POP = "eur"/"afr" below reruns the whole thing under another history.
# ============================================================================

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

OUTROOT  = f"{RESULTS}/{POP}/simulations_ages"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
COMP_HET_BIN = str(SIMBIN / "comp_het_lineage_tracker")

# — 1b. Disk policy —————————————————————————————————————————
# The raw .out files hold one line per surviving lineage and are the only thing
# that could be re-summarised later (a different assignment seed, a revised
# variant table). The whole graph is ~4 GB, so they are kept. Set this False to
# release each merged .out as soon as its summary exists.
KEEP_RAW_OUT = True
maybe_temp = (lambda p: p) if KEEP_RAW_OUT else temp

# — 2. Constants shared by every mutator ————————————————————————
# h/s here describe the deleterious mutations the mutator CAUSES, not the
# mutator allele itself (that dominance is per-mutator, h_mut below).
H_DEL   = 0.5        # dominance of an induced deleterious mutation
S_DEL   = 0.001      # selection coefficient of an induced deleterious mutation
G_SIZE  = 3e9        # genome size (bp)
F_SEL   = 0.08       # fraction of the genome under selection

BASE_SEED = 20       # a different block from the const_phiG graph's 10

# compound-het simulator; identical settings to the const_phiG frequency run
BURN_IN        = "2.5e5"   # start generation. AGES ARE CENSORED HERE.
BURN_IN_GENS   = 250000    # the same number, for --burn-in on the summariser
BURNIN_NE      = "20000"   # Ne held during the burn-in, i.e. for gen > T[0]
UFACTOR        = "1.0"
ALLOW_BACK_MUT = "1"

S_FMT = "{:.4f}"


def selection_coefficient(phi, phi_scope):
    """Selection coefficient of the mutator in a HOMOZYGOTE.

    phi_scope == "hom": phi is already the homozygous effect size.
    phi_scope == "het": phi is the effect in a heterozygous parent, so the
                        homozygous effect is 2*phi.
    """
    phi_hom = 2.0 * phi if phi_scope == "het" else phi
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


# — 3. The simulations ————————————————————————————————————————
#   mut_rate  : rate handed to the simulator. Single-variant rate for MPG /
#               POLD1 / POLE, gene-wide LoF rate for XPC / MUTYH.
#   h_mut     : dominance of the MUTATOR allele itself (0 = recessive). Recorded
#               but inert for the s = 0 controls.
#   phi       : effect size; replaced by neutral=True for the s = 0 controls.
#   variants  : focal variant names and rates, for the lineage assignment. The
#               neutral control of a mutator reuses that mutator's variant names
#               so the two npz key sets line up.
#   gene_mu   : denominator of the assignment. "sum" means the focal variants
#               account for the whole gene rate (MUTYH).
#   runs      : total replicates. Sized as 100k / (measured per-variant presence
#               rate) so every mutator clears ~100k usable runs, matching the
#               const_phiG frequency runs. See the README table.
#   chunks    : sized so one chunk is ~1 h at the measured uncontended rate, so
#               that it still fits 11:59:00 after the 6-10x slowdown that
#               oversubscribed nodes inflict (see the note in the main README).
MUTATORS = {
    # ---- the seven mutators at their empirically estimated phi_G ----------
    "MPG": dict(
        mut_rate="1.0353e-07", h_mut="0", phi=4.2379e-08, phi_scope="hom",
        variants=[("MPG", "1.0353e-07")], gene_mu="1.0353e-07",
        runs=100000, chunks=2, mem="500M",      # presence 100.0%, 0.045 s/run
    ),
    "POLD1": dict(
        mut_rate="7.0543E-09", h_mut="0.5", phi=1.9211e-08, phi_scope="het",
        variants=[("POLD1", "7.0543E-09")], gene_mu="7.0543E-09",
        runs=270000, chunks=3, mem="500M",      # presence 37.5%,  0.037 s/run
    ),
    "POLE": dict(
        mut_rate="3.7048E-09", h_mut="0.5", phi=3.9161e-08, phi_scope="het",
        variants=[("POLE", "3.7048E-09")], gene_mu="3.7048E-09",
        runs=500000, chunks=4, mem="500M",      # presence 20.5%,  0.027 s/run
    ),
    "XPC": dict(
        mut_rate="4.4355e-06", h_mut="0", phi=7.7921e-08, phi_scope="hom",
        variants=[("XPC", "7.922075e-08")], gene_mu="4.4355e-06",
        runs=110000, chunks=16, mem="1000M",    # presence 97.8%,  0.50 s/run
    ),
    "MUTYH": dict(
        mut_rate="2.9029E-08", h_mut="0", phi=8.3248e-09, phi_scope="hom",
        variants=[("Y179C", "1.0049e-08"),
                  ("V234M", "1.1317e-08"),
                  ("G368D", "7.6633e-09")],
        gene_mu="sum",
        runs=220000, chunks=3, mem="1000M",     # presence 45.7 / 48.6 / 55.4%, 0.034 s/run
    ),

    # ---- the neutral controls, s = 0 -------------------------------------
    # One per mutation rate, so a mutator and its control differ only in s, plus
    # one at the human per-site rate. NOTE: neutral ages run right up against the
    # burn-in, so their upper tail is censored at 250,000 generations. The
    # summariser reports the censored fraction per mutator.
    "MPG_neutral": dict(
        mut_rate="1.0353e-07", h_mut="0", neutral=True,
        variants=[("MPG", "1.0353e-07")], gene_mu="1.0353e-07",
        runs=100000, chunks=2, mem="500M",      # presence 100.0%, 0.050 s/run
    ),
    "POLD1_neutral": dict(
        mut_rate="7.0543E-09", h_mut="0.5", neutral=True,
        variants=[("POLD1", "7.0543E-09")], gene_mu="7.0543E-09",
        runs=250000, chunks=3, mem="500M",      # presence 40.9%,  0.032 s/run
    ),
    "POLE_neutral": dict(
        mut_rate="3.7048E-09", h_mut="0.5", neutral=True,
        variants=[("POLE", "3.7048E-09")], gene_mu="3.7048E-09",
        runs=450000, chunks=4, mem="500M",      # presence 22.9%,  0.031 s/run
    ),
    "XPC_neutral": dict(
        mut_rate="4.4355e-06", h_mut="0", neutral=True,
        variants=[("XPC", "7.922075e-08")], gene_mu="4.4355e-06",
        runs=140000, chunks=24, mem="1000M",    # presence 72.6%,  0.60 s/run
    ),
    "MUTYH_neutral": dict(
        mut_rate="2.9029E-08", h_mut="0", neutral=True,
        variants=[("Y179C", "1.0049e-08"),
                  ("V234M", "1.1317e-08"),
                  ("G368D", "7.6633e-09")],
        gene_mu="sum",
        runs=240000, chunks=3, mem="1000M",     # presence 42.2 / 50.1 / 56.2%, 0.035 s/run
    ),
    "human_neutral": dict(
        mut_rate="1.25e-08", h_mut="0", neutral=True,
        variants=[("neutral", "1.25e-08")], gene_mu="1.25e-08",
        runs=170000, chunks=2, mem="500M",      # presence 60.8%,  0.039 s/run
    ),
}

WALLTIME = "'11:59:00'"   # the short partition maximum, for the reason in the README

# Longest label first, so the wildcard regex cannot match "MPG" inside
# "MPG_neutral" if snakemake ever stops anchoring the alternation.
LABELS = sorted(MUTATORS, key=lambda g: (-len(g), g))

# s is a deterministic function of the mutator, so label<->s pairs travel together
S = {g: S_FMT.format(0.0 if c.get("neutral")
                     else selection_coefficient(c["phi"], c["phi_scope"]))
     for g, c in MUTATORS.items()}

# distinct seed block per mutator so no two jobs share a seed
SEED_OFFSET = {g: 100000 * (i + 1) for i, g in enumerate(sorted(MUTATORS))}


def gene_runs(gene):
    return MUTATORS[gene]["runs"]


def chunk_runs(gene, chunk):
    """Runs in one chunk; the chunks sum to exactly gene_runs(gene)."""
    n = MUTATORS[gene]["chunks"]
    base, rem = divmod(gene_runs(gene), n)
    return base + (1 if int(chunk) < rem else 0)


def chunk_list(gene):
    return list(range(MUTATORS[gene]["chunks"]))


def sim_out(gene):
    return f"{OUTROOT}/{gene}/s_{S[gene]}.out"


def variant_args(gene):
    return " ".join(f"--variant {name}:{mu}" for name, mu in MUTATORS[gene]["variants"])


def ages_json(gene):
    return f"{OUTROOT}/{gene}/{gene}_{POP}_ages.json"


def ages_npz(gene):
    return f"{OUTROOT}/{gene}/{gene}_{POP}_ages.npz"


COMBINED_NPZ  = f"{OUTROOT}/all_mutators_{POP}_ages.npz"
COMBINED_JSON = f"{OUTROOT}/all_mutators_{POP}_ages.json"
COMBINED_TSV  = f"{OUTROOT}/all_mutators_{POP}_ages.tsv"


wildcard_constraints:
    gene  = "|".join(LABELS),
    s     = r"[0-9.]+",
    chunk = r"\d+",

localrules: all


# — 4. Targets ——————————————————————————————————————————————
rule all:
    input:
        COMBINED_NPZ,
        COMBINED_JSON,
        COMBINED_TSV,
        [ages_json(g) for g in LABELS],
        [ages_npz(g) for g in LABELS],


# — 5. Simulations: every mutator goes through the lineage tracker ————
rule simulate_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out"),
    params:
        binary    = COMP_HET_BIN,
        mut_rate  = lambda wc: MUTATORS[wc.gene]["mut_rate"],
        h_mut     = lambda wc: MUTATORS[wc.gene]["h_mut"],
        runs      = lambda wc: chunk_runs(wc.gene, wc.chunk),
        ufactor   = UFACTOR,
        seed      = lambda wc: BASE_SEED + SEED_OFFSET[wc.gene] + int(wc.chunk),
        back_mut  = ALLOW_BACK_MUT,
        burn_in   = BURN_IN,
        pop_label = POP,
        burnin_Ne = BURNIN_NE,
    resources:
        partition     = "short",
        time          = WALLTIME,
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: MUTATORS[wc.gene]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        {params.binary} \
          {params.runs} {wildcards.s} {params.h_mut} {params.mut_rate} \
          {params.ufactor} {params.seed} {params.back_mut} {params.burn_in} \
          {params.pop_label} {params.burnin_Ne} \
          > {output.chunk_out}
        """


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
        binary    = COMP_HET_BIN,
        mut_rate  = lambda wc: MUTATORS[wc.gene]["mut_rate"],
        h_mut     = lambda wc: MUTATORS[wc.gene]["h_mut"],
        runs      = lambda wc: gene_runs(wc.gene),
        n_chunks  = lambda wc: MUTATORS[wc.gene]["chunks"],
        ufactor   = UFACTOR,
        seed      = lambda wc: BASE_SEED + SEED_OFFSET[wc.gene],
        back_mut  = ALLOW_BACK_MUT,
        burn_in   = BURN_IN,
        pop_label = POP,
        burnin_Ne = BURNIN_NE,
        neutral   = lambda wc: int(bool(MUTATORS[wc.gene].get("neutral"))),
        phi       = lambda wc: MUTATORS[wc.gene].get("phi", "NA"),
        phi_scope = lambda wc: MUTATORS[wc.gene].get("phi_scope", "NA"),
        variants  = lambda wc: ",".join(f"{n}:{m}" for n, m in MUTATORS[wc.gene]["variants"]),
        gene_mu   = lambda wc: MUTATORS[wc.gene]["gene_mu"],
    resources:
        partition     = "short",
        # concatenating the 24 XPC_neutral chunks is ~1.3 GB of Lustre I/O
        time          = "'04:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "1000M",
    shell:
        r"""
        mkdir -p $(dirname {output.out})
        cat {input.chunks} > {output.out}

        {{
          echo "mutator={wildcards.gene}"
          echo "kind=compound_het_lineage_tracker_for_ages"
          echo "population={params.pop_label}"
          echo "simulator={params.binary}"
          echo "mut_rate={params.mut_rate}"
          echo "s={wildcards.s}"
          echo "neutral_control={params.neutral}"
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
          echo "ufactor={params.ufactor}"
          echo "allow_back_mut={params.back_mut}"
          echo "burn_in={params.burn_in}"
          echo "burnin_Ne={params.burnin_Ne}"
          echo "summary_variants={params.variants}"
          echo "summary_gene_mu={params.gene_mu}"
        }} > {output.params_file}
        """


# — 6. Ages: oldest surviving lineage per run, per focal variant ————
rule summarize_ages:
    input:
        script = f"{SCRIPTS}/summarize_ages.py",
        out = lambda wc: sim_out(wc.gene),
    output:
        json_file = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_ages.json",
        npz_file  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_ages.npz",
    params:
        pop_label = POP,
        gene_mu   = lambda wc: MUTATORS[wc.gene]["gene_mu"],
        variants  = lambda wc: variant_args(wc.gene),
        mut_rate  = lambda wc: MUTATORS[wc.gene]["mut_rate"],
        h_mut     = lambda wc: MUTATORS[wc.gene]["h_mut"],
        neutral   = lambda wc: int(bool(MUTATORS[wc.gene].get("neutral"))),
        phi       = lambda wc: MUTATORS[wc.gene].get("phi", "NA"),
        burn_in   = BURN_IN_GENS,
    resources:
        partition     = "short",
        # XPC is ~35M lineage lines to parse; ~4 min uncontended, but the same
        # 6-10x node contention that the simulation chunks defend against
        # applies here, so it gets the full short-partition walltime too
        time          = WALLTIME,
        cpus_per_task = 1,
        mem_per_cpu   = "8000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --gene-mu {params.gene_mu} \
          {params.variants} \
          --burn-in {params.burn_in} \
          --out-json {output.json_file} \
          --out-npz  {output.npz_file} \
          --gene {wildcards.gene} \
          --population {params.pop_label} \
          --meta simulator_mut_rate={params.mut_rate} \
          --meta h_mutator={params.h_mut} \
          --meta neutral_control={params.neutral} \
          --meta phi={params.phi} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL}
        """


# — 7. One npz / json / tsv with every mutator in it ————————————
rule combine_ages:
    input:
        script = f"{SCRIPTS}/combine_ages.py",
        npz = [ages_npz(g) for g in LABELS],
        jsn = [ages_json(g) for g in LABELS],
    output:
        npz  = COMBINED_NPZ,
        jsn  = COMBINED_JSON,
        tsv  = COMBINED_TSV,
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "8000M",
    shell:
        r"""
        python3 {input.script} {input.npz} \
          --out-npz  {output.npz} \
          --out-json {output.jsn} \
          --out-tsv  {output.tsv}
        """
