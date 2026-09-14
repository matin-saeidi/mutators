# ============================================================================
# Model-validation simulations: does the mutator's selection coefficient behave
# the way the derivation says, across mutator dominance h?
#
# Two products per h, one per row of the figure:
#
#   trajectories/trajectories_h_{h}.npz
#       50 replicates started at q0 = 0.5, followed for 200 generations, recording
#       lambda_tau (excess DNMs linked to a mutator haplotype) and q_tau.
#
#   s_vs_q/s_vs_q_h_{h}.npz
#       10,000 replicates, each starting from its own q ~ Uniform(0.05, 0.95) and
#       run for 50 generations, recording the sampled q, the frequency actually
#       reached, the observed s, and the s the theory predicts at that frequency.
#
# h is the only wildcard: 4 values x 2 rows = 8 cluster jobs, unchunked.
#
# Measured cost (one core, N = 20,000):
#   trajectories   3.3 s per replicate  ->  ~3 min per h
#   s_vs_q         0.55 s per replicate ->  ~1.5 h per h at 10,000 replicates
# Both fit inside `short` with room to spare, so each h runs as a single job.
# If one ever does hit the wall, simulate_s_vs_q.py still takes --rep-start, so a
# run can be split without changing any number: each replicate's RNG stream is
# keyed by (seed, replicate index), not by its position in the run.
#
# Run from the repository root, in a tmux session so it survives a
# dropped connection (SNAKEMAKE=~/anaconda3/envs/snakemake/bin/snakemake):
#   $SNAKEMAKE --snakefile workflow/model_validation.smk --profile .
# or to inspect first, without submitting anything:
#   $SNAKEMAKE --snakefile workflow/model_validation.smk -np
# ============================================================================

# — 1. Paths ————————————————————————————————————————————————
# Every path is derived from where this file sits, so a clone runs anywhere.
# Send results elsewhere with:  --config results_dir=/scratch/you/mutators
from pathlib import Path

REPO    = Path(workflow.basedir).parent.resolve()
RESULTS = Path(config.get("results_dir", REPO / "results")).resolve()
SCRIPTS = str(REPO / "src" / "model_validation")
SIMBIN  = REPO / "src" / "simulations" / "bin"

OUTROOT  = f"{RESULTS}/model_validation"
TMPROOT  = f"{OUTROOT}/tmp_chunks"
# — 2. The dominance grid, one column of the figure each ——————————
H_VALUES = ["0.00", "0.25", "0.75", "1.00"]

# — 3. Model parameters, shared by both rows ————————————————————
N        = 20000      # diploid population size
PHI_G    = 100.0      # excess DNMs per gamete from a homozygous mutator
F_SEL    = 0.08       # fraction of DNMs that are deleterious
S_HET    = 0.0005     # fitness cost per excess deleterious DNM carried

# top row: trajectories
TRAJ_Q0      = 0.5
TRAJ_N_REPS  = 50
TRAJ_N_GENS  = 200
TRAJ_SEED    = 9000

# bottom row: s versus the frequency actually reached
SVQ_N_REPS   = 10000
SVQ_N_GENS   = 50
SVQ_Q_MIN    = 0.01
SVQ_Q_MAX    = 0.99
SVQ_SEED     = 1000

H_INDEX = {h: i for i, h in enumerate(H_VALUES)}

wildcard_constraints:
    h = r"[0-9.]+",

localrules: all


# — 4. Targets ——————————————————————————————————————————————
rule all:
    input:
        [f"{OUTROOT}/trajectories/trajectories_h_{h}.npz" for h in H_VALUES],
        [f"{OUTROOT}/s_vs_q/s_vs_q_h_{h}.npz"             for h in H_VALUES],


# — 5. Top row: trajectories of q and of the linked DNMs ————————————
rule trajectories:
    output:
        npz = f"{OUTROOT}/trajectories/trajectories_h_{{h}}.npz",
    params:
        script = f"{SCRIPTS}/simulate_trajectories.py",
        seed   = lambda wc: TRAJ_SEED + 100 * H_INDEX[wc.h],
    resources:
        partition     = "short",
        time          = "'02:00:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "500M",
    shell:
        r"""
        mkdir -p $(dirname {output.npz})

        python3 {params.script} \
          --h {wildcards.h} \
          --N {N} --phi-G {PHI_G} --f {F_SEL} --s-het {S_HET} \
          --q0 {TRAJ_Q0} --n-reps {TRAJ_N_REPS} --n-gens {TRAJ_N_GENS} \
          --seed {params.seed} \
          --out {output.npz}
        """


# — 6. Bottom row: s at the frequency actually reached ————————————
rule s_vs_q:
    output:
        npz = f"{OUTROOT}/s_vs_q/s_vs_q_h_{{h}}.npz",
    params:
        script = f"{SCRIPTS}/simulate_s_vs_q.py",
        seed   = lambda wc: SVQ_SEED + 100 * H_INDEX[wc.h],
    resources:
        partition     = "short",
        time          = "'11:59:00'",
        cpus_per_task = 1,
        mem_per_cpu   = "500M",
    shell:
        r"""
        mkdir -p $(dirname {output.npz})

        python3 {params.script} \
          --h {wildcards.h} \
          --N {N} --phi-G {PHI_G} --f {F_SEL} --s-het {S_HET} \
          --q-min {SVQ_Q_MIN} --q-max {SVQ_Q_MAX} --n-gens {SVQ_N_GENS} \
          --n-reps {SVQ_N_REPS} \
          --seed {params.seed} \
          --out {output.npz}
        """
