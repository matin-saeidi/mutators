# ============================================================================
# Frequency-distribution simulations for all 7 mutator alleles under the
# SOUTH ASIAN demographic history, at each mutator's constant (empirically
# estimated) effect size phi_G -- but with BACK MUTATION TURNED OFF.
#
# This is Snakefile_SAS_dem_const_phiG_all_genes with one substantive change:
# the back-mutation rate is 0 instead of being equal to the forward rate.
# Every other parameter -- demography, phi_G, s, h_mut, mutation rates, run
# counts, chunking, seeds -- is identical to the baseline, so the two sets are
# directly comparable. The s values are the baseline's, to 4 decimals, so the
# npz keys match the baseline's key for key.
#
# ---------------------------------------------------------------------------
# How back mutation is controlled, per simulator (checked in the sources, and
# the two are NOT alike):
#
# COMPOUND HET (comp_het_lineage_tracker) -- a runtime argument.
#   argv[7] is allow_back_mutation, read as `(atoi(argv[7]) != 0)`, and argv[5]
#   is ufactor, with back rate = ufactor * mutU. The baseline passes
#   ufactor = 1.0 and allow_back_mutation = 1, i.e. back rate == forward rate.
#   Setting argv[7] = 0 skips the back-mutation block outright; nothing divides
#   by the back rate, and this simulator does no jumping, so 0 needs no other
#   change. ufactor is left at 1.0 -- it is dead when the flag is off, and
#   keeping it makes the two command lines differ in exactly one place.
#
# SINGLE SITE (single_site_simulator) -- NOT a runtime argument.
#   The back rate is a compile-time `#define ratio 1` (ratio = back/forward),
#   so there is no flag to set and turning it off needs a separate binary.
#   Naively compiling with ratio = 0 is broken in three places:
#     1. basedownVAR = 1/(ratio*halfthetaVAR) becomes 1/0 = inf, and the
#        int(-log(u)*inf) that consumes it is undefined behaviour;
#     2. the "derived allele fixed" branch draws an exponential waiting time
#        for a back mutation that can never happen;
#     3. poicond1(l) is Poisson CONDITIONED ON >= 1, so poicond1(0) still
#        returns a back mutation rather than none.
#   AutosomeSimulate_smallback_multidem_back_mu_0.cpp applies the same three
#   guards Sara had already applied to the single-demography
#   AutosomeSimulate_smallback_back_mu_0.cpp, and is built as
#   single_site_simulator_no_back_mut:
#
#     cd <repo root>/AutosomeSimulations
#     g++ -O3 -std=c++11 BRand.cpp population.cpp \
#         AutosomeSimulate_smallback_multidem_back_mu_0.cpp \
#         -o single_site_simulator_no_back_mut
#
#   (boost_poi(0) does return 0, so the segregating branch needed no guard.)
#   Checked after building: --dump-demography sas still passes all epoch
#   checks, and 3,000 MPG runs give finite frequencies in [0,1] with no zeros
#   pathology and no run-time change (57.2 s vs 57.5 s for ratio = 1).
#
# Expect the effect to be SMALL for the single-site genes: at s > 0 the derived
# allele essentially never fixes (Fixed: 0 generations in all 3,000 test runs),
# so the branch that differs is never entered, and back mutation on a
# segregating allele at mutU ~ 1e-7 removes a negligible number of copies. The
# margin should be largest for XPC, whose gene-wide LoF rate (4.4e-6) is the
# highest here; a 250-run test moved its median gene frequency by ~17%
# (KS p = 0.13, so not resolved at that sample size).
# ---------------------------------------------------------------------------
#
# Outputs live in their own tree (simulations_no_back_mut/) and carry the
# _no_back_mut suffix, so nothing here can overwrite the baseline results.
#
# One snakemake job graph covering:
#   single site  : MPG, POLD1, POLE      -> single_site_simulator_no_back_mut
#   compound het : XPC, MUTYH            -> comp_het_lineage_tracker (argv[7]=0)
#
# MUTYH is simulated once as a pooled gene-wide LoF allele (the three known
# mutator variants share a selection coefficient); the summarise step splits
# the surviving lineages into Y179C / V234M / G368D in proportion to their
# mutation rates. That is 5 simulations for 7 mutators.
#
# Run from the repository root:
#   snakemake --snakefile workflow/const_phiG_no_back_mut.smk -np
#   snakemake --snakefile workflow/const_phiG_no_back_mut.smk --profile .
#
# Setting POP = "eur"/"afr" below reruns the NFE or African histories through
# the exact same code path, which is handy as a control.
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

OUTROOT  = f"{RESULTS}/{POP}/simulations_no_back_mut"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
# suffix on every summary file, so these never collide with the baseline run
TAG          = "no_back_mut"

# the back-mutation-free build; see the header for how it differs
SINGLE_SITE_BIN = str(SIMBIN / "single_site_simulator_no_back_mut")
COMP_HET_BIN    = str(SIMBIN / "comp_het_lineage_tracker")

# — 2. Constants shared by every mutator ————————————————————————
# Fitness effect of the *deleterious mutations the mutator causes* (NOT the
# mutator's own dominance, which is per-gene and lives in GENES below).
H_DEL   = 0.5        # dominance of an induced deleterious mutation
S_DEL   = 0.001      # selection coefficient of an induced deleterious mutation
G_SIZE  = 3e9        # genome size (bp)
F_SEL   = 0.08       # fraction of the genome under selection

RUNS        = 100000   # default; a gene may override it with runs=... below
BASE_SEED   = 10

# single-site simulator: mut_uncert=0, dem_uncert=0, demographic_model=1
OTHER_ARGS      = "0 0 1"
NE_CONST        = "1e6"    # ignored when demographic_model = 1
BURN_IN_SINGLE  = "2.5e5"  # ignored when demographic_model = 1 (code forces 250000)

# back/forward mutation-rate ratio compiled into SINGLE_SITE_BIN. Recorded in
# the params file so the .out can never be mistaken for a baseline run; the
# binary is the thing that actually sets it.
BACK_MUT_RATIO_SINGLE = "0"

# compound-het simulator
BURN_IN_COMPHET = "2.5e5"
BURNIN_NE       = "20000"  # Ne held during the burn-in, matching the NFE/AFR runs
UFACTOR         = "1.0"    # dead while ALLOW_BACK_MUT = 0; kept at the baseline value
ALLOW_BACK_MUT  = "0"      # <-- the change; baseline is "1"


def selection_coefficient(phi, phi_scope):
    """Selection coefficient of the mutator in a HOMOZYGOTE.

    phi_scope == "hom": phi is already the homozygous effect size.
    phi_scope == "het": phi is the effect in a heterozygous parent, so the
                        homozygous effect is 2*phi.
    """
    phi_hom = 2.0 * phi if phi_scope == "het" else phi
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


# — 3. The seven mutators ————————————————————————————————————
#   h_mut     : dominance of the MUTATOR allele itself (0 = recessive)
#   phi       : effect size, i.e. increase in the per-generation mutation rate
#   phi_scope : whether phi is the homozygous or heterozygous effect
#   mut_rate  : mutation rate fed to the simulator (gene-wide LoF rate for the
#               compound-het genes, single-variant rate for the single-site ones)
# Identical to the baseline Snakefile -- only the back-mutation rate differs.
GENES = {
    "MPG": dict(
        kind="single", mut_rate="1.0353e-07", h_mut="0",
        phi=4.2379e-08, phi_scope="hom", chunks=1,
        walltime="'11:59:00'", mem="500M",
    ),
    "POLD1": dict(
        kind="single", mut_rate="7.0543E-09", h_mut="0.5",
        phi=1.9211e-08, phi_scope="het", chunks=1,
        walltime="'11:59:00'", mem="500M",
    ),
    "POLE": dict(
        kind="single", mut_rate="3.7048E-09", h_mut="0.5",
        phi=3.9161e-08, phi_scope="het", chunks=1,
        walltime="'11:59:00'", mem="500M",
    ),
    "XPC": dict(
        kind="comphet", mut_rate="4.4355e-06", h_mut="0",
        phi=7.7921e-08, phi_scope="hom", chunks=16,
        walltime="'11:59:00'", mem="1000M",
        # one focal variant inside a much larger gene-wide LoF pool
        variants=[("XPC", "7.922075e-08")],
        summary_gene_mu="4.4355e-06",
    ),
    "MUTYH": dict(
        kind="comphet", mut_rate="2.9029E-08", h_mut="0",
        phi=8.3248e-09, phi_scope="hom",
        # Every replicate is saved; the run count is sized by the analysis
        # that conditions on all three variants segregating. That subset is
        # 12.1% of runs at baseline (90,703 of 750,000);
        # a 3,000-run test with back mutation off gave 2.116 lineages per run
        # against the baseline's 2.084, so the keep rate is unchanged to within
        # noise and 7.5e5 runs again yields ~90k usable runs.
        runs=750000, chunks=12,
        walltime="'11:59:00'", mem="1000M",
        # the three known mutator variants account for the whole gene rate,
        # so the summariser divides by their sum (they round to 2.9029E-08)
        variants=[("Y179C", "1.0049e-08"),
                  ("V234M", "1.1317e-08"),
                  ("G368D", "7.6633e-09")],
        summary_gene_mu="sum",
    ),
}

SINGLE_GENES  = sorted(g for g, c in GENES.items() if c["kind"] == "single")
COMPHET_GENES = sorted(g for g, c in GENES.items() if c["kind"] == "comphet")
ALL_GENES     = SINGLE_GENES + COMPHET_GENES

# s is a deterministic function of the gene, so gene<->s pairs travel together.
# 4 decimals, exactly as in the baseline, so every npz key matches the
# baseline's and the two runs can be read with one set of keys.
S = {g: "{:.4f}".format(selection_coefficient(c["phi"], c["phi_scope"]))
     for g, c in GENES.items()}

# distinct seed block per gene so no two jobs share a seed
SEED_OFFSET = {g: 1000 * (i + 1) for i, g in enumerate(ALL_GENES)}


def gene_runs(gene):
    """Total simulated runs for a gene (RUNS unless the gene overrides it)."""
    return GENES[gene].get("runs", RUNS)


def chunk_runs(gene, chunk):
    """Runs in one chunk; the chunks sum to exactly gene_runs(gene)."""
    n = GENES[gene]["chunks"]
    base, rem = divmod(gene_runs(gene), n)
    return base + (1 if int(chunk) < rem else 0)


def chunk_list(gene):
    return list(range(GENES[gene]["chunks"]))


def sim_out(gene):
    return f"{OUTROOT}/{gene}/s_{S[gene]}.out"


def variant_args(gene):
    return " ".join(f"--variant {name}:{mu}" for name, mu in GENES[gene]["variants"])


wildcard_constraints:
    gene  = "|".join(ALL_GENES),
    s     = r"[0-9.]+",
    chunk = r"\d+",

localrules: all


# — 4. Targets ——————————————————————————————————————————————
rule all:
    input:
        [sim_out(g) for g in ALL_GENES],
        [f"{OUTROOT}/{g}/{g}_{POP}_dem_const_phiG_{TAG}.json" for g in ALL_GENES],
        [f"{OUTROOT}/{g}/{g}_{POP}_dem_const_phiG_{TAG}.npz" for g in ALL_GENES],


# — 5. Single-site simulations (MPG, POLD1, POLE) ————————————————
rule simulate_single:
    output:
        out         = f"{OUTROOT}/{{gene}}/s_{{s}}.out",
        params_file = f"{OUTROOT}/{{gene}}/s_{{s}}.params.txt",
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        binary      = SINGLE_SITE_BIN,
        mut_rate    = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut       = lambda wc: GENES[wc.gene]["h_mut"],
        runs        = lambda wc: gene_runs(wc.gene),
        other_args  = OTHER_ARGS,
        Ne          = NE_CONST,
        seed        = lambda wc: BASE_SEED + SEED_OFFSET[wc.gene],
        burn_in     = BURN_IN_SINGLE,
        pop_label   = POP,
        phi         = lambda wc: GENES[wc.gene]["phi"],
        phi_scope   = lambda wc: GENES[wc.gene]["phi_scope"],
    resources:
        partition     = "short",
        time          = lambda wc: GENES[wc.gene]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: GENES[wc.gene]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.out})

        {params.binary} \
          {params.mut_rate} {wildcards.s} {params.h_mut} {params.runs} \
          {params.other_args} {params.Ne} {params.seed} {params.burn_in} \
          {params.pop_label} \
          > {output.out}

        {{
          echo "gene={wildcards.gene}"
          echo "kind=single_site"
          echo "population={params.pop_label}"
          echo "simulator={params.binary}"
          echo "back_mutation_ratio={BACK_MUT_RATIO_SINGLE}"
          echo "mut_rate={params.mut_rate}"
          echo "s={wildcards.s}"
          echo "h_mutator={params.h_mut}"
          echo "phi={params.phi}"
          echo "phi_scope={params.phi_scope}"
          echo "h_deleterious={H_DEL}"
          echo "s_deleterious={S_DEL}"
          echo "genome_size={G_SIZE}"
          echo "fraction_selected={F_SEL}"
          echo "runs={params.runs}"
          echo "other_args={params.other_args}"
          echo "Ne_const_ignored={params.Ne}"
          echo "random_seed={params.seed}"
          echo "burn_in_ignored_under_dem_model_1={params.burn_in}"
        }} > {output.params_file}
        """


# — 6. Compound-het simulations (XPC, MUTYH), chunked ————————————
rule simulate_comphet_chunk:
    output:
        chunk_out = temp(f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out"),
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        binary   = COMP_HET_BIN,
        mut_rate = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut    = lambda wc: GENES[wc.gene]["h_mut"],
        runs     = lambda wc: chunk_runs(wc.gene, wc.chunk),
        ufactor  = UFACTOR,
        seed     = lambda wc: BASE_SEED + SEED_OFFSET[wc.gene] + int(wc.chunk),
        back_mut = ALLOW_BACK_MUT,
        burn_in  = BURN_IN_COMPHET,
        pop_label   = POP,
        burnin_Ne = BURNIN_NE,
    resources:
        partition     = "short",
        time          = lambda wc: GENES[wc.gene]["walltime"],
        cpus_per_task = 1,
        mem_per_cpu   = lambda wc: GENES[wc.gene]["mem"],
    shell:
        r"""
        mkdir -p $(dirname {output.chunk_out})

        {params.binary} \
          {params.runs} {wildcards.s} {params.h_mut} {params.mut_rate} \
          {params.ufactor} {params.seed} {params.back_mut} {params.burn_in} \
          {params.pop_label} {params.burnin_Ne} \
          > {output.chunk_out}
        """


rule merge_comphet_chunks:
    input:
        chunks = lambda wc: expand(
            f"{TMPROOT}/{{gene}}/s_{{s}}_chunk{{chunk}}.out",
            gene=wc.gene, s=wc.s, chunk=chunk_list(wc.gene),
        ),
    output:
        out         = f"{OUTROOT}/{{gene}}/s_{{s}}.out",
        params_file = f"{OUTROOT}/{{gene}}/s_{{s}}.params.txt",
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        binary    = COMP_HET_BIN,
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
        runs      = lambda wc: gene_runs(wc.gene),
        n_chunks  = lambda wc: GENES[wc.gene]["chunks"],
        ufactor   = UFACTOR,
        seed      = lambda wc: BASE_SEED + SEED_OFFSET[wc.gene],
        back_mut  = ALLOW_BACK_MUT,
        burn_in   = BURN_IN_COMPHET,
        pop_label   = POP,
        burnin_Ne = BURNIN_NE,
        phi       = lambda wc: GENES[wc.gene]["phi"],
        phi_scope = lambda wc: GENES[wc.gene]["phi_scope"],
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
          echo "kind=compound_het"
          echo "population={params.pop_label}"
          echo "simulator={params.binary}"
          echo "mut_rate={params.mut_rate}"
          echo "s={wildcards.s}"
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
          echo "ufactor_unused_when_back_mut_off={params.ufactor}"
          echo "allow_back_mut={params.back_mut}"
          echo "burn_in={params.burn_in}"
          echo "burnin_Ne={params.burnin_Ne}"
        }} > {output.params_file}
        """


# — 7. Summaries ————————————————————————————————————————————
rule summarize_single:
    input:
        script = f"{SCRIPTS}/summarize_single_site.py",
        out = lambda wc: sim_out(wc.gene),
    output:
        json_file = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG_{TAG}.json",
        npz_file  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG_{TAG}.npz",
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        pop_label   = POP,
        mut_rate = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut    = lambda wc: GENES[wc.gene]["h_mut"],
        phi      = lambda wc: GENES[wc.gene]["phi"],
        back_mut = BACK_MUT_RATIO_SINGLE,
    resources:
        partition     = "short",
        time          = "'01:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "4000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --out-json {output.json_file} \
          --out-npz  {output.npz_file} \
          --gene {wildcards.gene} \
          --population {params.pop_label} \
          --meta mut_rate={params.mut_rate} \
          --meta h_mutator={params.h_mut} \
          --meta phi={params.phi} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL} \
          --meta back_mutation_ratio={params.back_mut}
        """


rule summarize_comphet:
    input:
        script = f"{SCRIPTS}/summarize_comp_het.py",
        out = lambda wc: sim_out(wc.gene),
    output:
        json_file = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG_{TAG}.json",
        npz_file  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG_{TAG}.npz",
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        pop_label   = POP,
        gene_mu   = lambda wc: GENES[wc.gene]["summary_gene_mu"],
        variants  = lambda wc: variant_args(wc.gene),
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
        phi       = lambda wc: GENES[wc.gene]["phi"],
        back_mut  = ALLOW_BACK_MUT,
    resources:
        partition     = "short",
        time          = "'02:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "4000M",
    shell:
        r"""
        python3 {input.script} {input.out} \
          --gene-mu {params.gene_mu} \
          {params.variants} \
          --out-json {output.json_file} \
          --out-npz  {output.npz_file} \
          --gene {wildcards.gene} \
          --population {params.pop_label} \
          --meta simulator_mut_rate={params.mut_rate} \
          --meta h_mutator={params.h_mut} \
          --meta phi={params.phi} \
          --meta h_deleterious={H_DEL} \
          --meta s_deleterious={S_DEL} \
          --meta allow_back_mutation={params.back_mut}
        """
