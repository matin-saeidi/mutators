# ============================================================================
# Frequency-distribution simulations for all 7 mutator alleles under the
# SOUTH ASIAN demographic history, at each mutator's constant (empirically
# estimated) effect size phi_G.
#
# One snakemake job graph covering:
#   single site  : MPG, POLD1, POLE      -> single_site_simulator
#   compound het : XPC, MUTYH            -> comp_het_lineage_tracker
#
# MUTYH is simulated once as a pooled gene-wide LoF allele (the three known
# mutator variants share a selection coefficient); the summarise step splits
# the surviving lineages into Y179C / V234M / G368D in proportion to their
# mutation rates. That is 5 simulations for 7 mutators.
#
# THE SAVED DISTRIBUTIONS ARE UNCONDITIONAL. Every simulated replicate reaches
# the npz, a replicate in which the mutator was lost entering as a frequency of
# exactly 0, so the arrays have one entry per run and MUTYH's three arrays are
# aligned run-for-run. Conditioning is a downstream filter, not something baked
# in here:
#
#     q = np.load(npz)["0.0102"];  q[q > 0]          # one mutator, segregating
#     keep = (y > 0) & (v > 0) & (g > 0)              # MUTYH, all three
#
# so one set of simulations serves both the conditional and the unconditional
# analysis.
#
# Run from the repository root:
#   snakemake --snakefile workflow/const_phiG.smk -np
#   snakemake --snakefile workflow/const_phiG.smk --profile .
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

OUTROOT  = f"{RESULTS}/{POP}/simulations"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
SINGLE_SITE_BIN = str(SIMBIN / "single_site_simulator")
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

# compound-het simulator
BURN_IN_COMPHET = "2.5e5"
BURNIN_NE       = "20000"  # Ne held during the burn-in, matching the NFE/AFR runs
UFACTOR         = "1.0"
ALLOW_BACK_MUT  = "1"


def selection_coefficient(phi, phi_scope):
    """Selection coefficient of the mutator in a HOMOZYGOTE.

    phi_scope == "hom": phi is already the homozygous effect size.
    phi_scope == "het": phi is the effect in a heterozygous parent, so the
                        homozygous effect is 2*phi.
    """
    phi_hom = 2.0 * phi if phi_scope == "het" else phi
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


# — 2b. Run counts, per demographic history ————————————————————————
# MUTYH run counts differ by history. The fraction of replicates in which all
# three MUTYH variants segregate at once varies with the present-day Ne, and
# these counts leave roughly 90k such replicates in each case:
#
#   history  present-day Ne  mean lineages/run  all three segregating  runs
#   NFE        31,498,556          9.00                ~83%            1e5
#   SAS         4,837,454          2.13                12.4%           7.5e5
#   AFR         2,763,103          1.36                 4.5%           2e6
#
# The other genes use the default RUNS.
RUN_PLAN = {
    "eur": dict(XPC_chunks=12, MUTYH_runs=100000,  MUTYH_chunks=4),
    "sas": dict(XPC_chunks=16, MUTYH_runs=750000,  MUTYH_chunks=12),
    "afr": dict(XPC_chunks=12, MUTYH_runs=2000000, MUTYH_chunks=24),
}
if POP not in RUN_PLAN:
    raise ValueError(f"unknown population {POP!r}; expected one of {sorted(RUN_PLAN)}")
PLAN = RUN_PLAN[POP]

# — 3. The seven mutators ————————————————————————————————————
#   h_mut     : dominance of the MUTATOR allele itself (0 = recessive)
#   phi       : effect size, i.e. increase in the per-generation mutation rate
#   phi_scope : whether phi is the homozygous or heterozygous effect
#   mut_rate  : mutation rate fed to the simulator (gene-wide LoF rate for the
#               compound-het genes, single-variant rate for the single-site ones)
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
        phi=7.7921e-08, phi_scope="hom", chunks=PLAN["XPC_chunks"],
        walltime="'11:59:00'", mem="1000M",
        # one focal variant inside a much larger gene-wide LoF pool
        variants=[("XPC", "7.922075e-08")],
        summary_gene_mu="4.4355e-06",
    ),
    "MUTYH": dict(
        kind="comphet", mut_rate="2.9029E-08", h_mut="0",
        phi=8.3248e-09, phi_scope="hom",
        # Run count is population-dependent; see RUN_PLAN above. The SAS
        # reasoning, as an example of how each was chosen:
        # Every replicate is saved, so the run count is sized by the most
        # demanding analysis the npz has to support: the one conditioning on
        # all three variants segregating. That subset is 12.4% of runs under
        # the South Asian history -- mean 2.13 surviving lineages per run, 37%
        # of runs with the >=3 lineages that are necessary. It sits between the
        # African 4.5% and the NFE 83%, as the present-day Ne does (4.84M vs
        # 2.76M and 31.5M). 7.5e5 runs therefore leaves ~93k triple-segregating
        # runs, matching the African run and comfortably above the NFE run's
        # 83,244.
        runs=PLAN["MUTYH_runs"], chunks=PLAN["MUTYH_chunks"],
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

# s is a deterministic function of the gene, so gene<->s pairs travel together
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
        [f"{OUTROOT}/{g}/{g}_{POP}_dem_const_phiG.json" for g in ALL_GENES],
        [f"{OUTROOT}/{g}/{g}_{POP}_dem_const_phiG.npz" for g in ALL_GENES],


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
          echo "ufactor={params.ufactor}"
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
        json_file = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG.json",
        npz_file  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG.npz",
    wildcard_constraints:
        gene = "|".join(SINGLE_GENES),
    params:
        pop_label   = POP,
        mut_rate = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut    = lambda wc: GENES[wc.gene]["h_mut"],
        phi      = lambda wc: GENES[wc.gene]["phi"],
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
          --meta s_deleterious={S_DEL}
        """


rule summarize_comphet:
    input:
        script = f"{SCRIPTS}/summarize_comp_het.py",
        out = lambda wc: sim_out(wc.gene),
    output:
        json_file = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG.json",
        npz_file  = f"{OUTROOT}/{{gene}}/{{gene}}_{POP}_dem_const_phiG.npz",
    wildcard_constraints:
        gene = "|".join(COMPHET_GENES),
    params:
        pop_label   = POP,
        gene_mu   = lambda wc: GENES[wc.gene]["summary_gene_mu"],
        variants  = lambda wc: variant_args(wc.gene),
        mut_rate  = lambda wc: GENES[wc.gene]["mut_rate"],
        h_mut     = lambda wc: GENES[wc.gene]["h_mut"],
        phi       = lambda wc: GENES[wc.gene]["phi"],
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
          --meta s_deleterious={S_DEL}
        """
