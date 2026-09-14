"""Shared helpers for the figure scripts: paths, style, the gnomAD and
demographic-model loaders, exact binomial confidence intervals, and the
log-axis floor handling.

The published figures are unconditional: every simulated replicate is included,
and a replicate in which the mutator was lost enters at a frequency of exactly
0. `require_unconditional` checks that a summary was written that way and stops
if it was not.
"""

from __future__ import annotations

import collections
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta
from scipy.special import gammaln, xlogy, xlog1py

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def results_root() -> Path:
    """Where the simulation outputs live.

    Defaults to <repo>/results, which is where the workflows write. Point it
    somewhere else with the MUTATORS_RESULTS environment variable, matching the
    `--config results_dir=...` you gave snakemake.
    """
    return Path(os.environ.get("MUTATORS_RESULTS", REPO / "results")).resolve()


def figures_root() -> Path:
    out = Path(os.environ.get("MUTATORS_FIGURES", REPO / "figures" / "output"))
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
# Sized for a 6.5-inch LaTeX text width. fonttype 42 embeds TrueType rather than
# Type 3, which is what journals ask for and what keeps text selectable.
RC = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.titleweight": "normal", "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
}

GNOMAD_COLOR = "#d62728"
BOX_FACECOLOR = "darkgray"
BOX_ALPHA = 0.7
CI_ALPHA = 0.20          # opacity of the shaded confidence band

MUTYH_VARIANTS_ALL = ("Y179C", "V234M", "G368D")


# --------------------------------------------------------------------------- #
# gnomAD allele counts
# --------------------------------------------------------------------------- #
def load_gnomad(path: Path | None = None) -> dict[str, dict[str, tuple[int, int]]]:
    """Allele counts from data/gnomad_v4.1.1_allele_counts.csv.

    Returns {mutator: {POPULATION: (k, n)}}, where k is the allele count and n
    the allele number. A gene contributing more than one mutator is keyed
    "<GENE>_<variant_short>" (MUTYH_Y179C); a gene contributing one is keyed by
    the gene symbol alone. Population codes are upper-cased from the file
    (NFE, SAS, AFR, ...).
    """
    path = path or DATA / "gnomad_v4.1.1_allele_counts.csv"
    with open(path) as fh:
        rows = list(csv.DictReader(r for r in fh if not r.startswith("#")))

    # count the distinct variants per gene, to decide how to key each mutator
    n_variants = collections.Counter(
        gene for gene, _ in {(r["gene"], r["variant_short"]) for r in rows})

    counts: dict[str, dict[str, tuple[int, int]]] = {}
    for r in rows:
        gene = r["gene"]
        key = f"{gene}_{r['variant_short']}" if n_variants[gene] > 1 else gene
        counts.setdefault(key, {})[r["population"].upper()] = (
            int(r["allele_count"]), int(r["allele_number"]))
    return counts


def load_demography(population: str) -> tuple[list[int], list[int]]:
    """(Ne, T) for 'eur' | 'sas' | 'afr', from data/demographic_models/.

    Dumped from the simulators, so these are the values the simulations used.
    The tables are in generations before the present: epoch i has size Ne[i] and
    covers T[i] < gen <= T[i-1], and Ne[0] applies to every generation older
    than T[0].
    """
    stem = {"eur": "NFE_schiffels_durbin_schraiber",
            "sas": "SAS_schiffels_durbin_kar",
            "afr": "AFR_schiffels_durbin_kar"}[population]
    ne, t = [], []
    with open(DATA / "demographic_models" / f"{stem}.tsv") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("epoch"):
                continue
            _, gen_hi, _, size = line.rstrip("\n").split("\t")
            ne.append(int(size))
            if gen_hi != "inf":
                t.append(int(gen_hi))
    # T[i] is the YOUNGER boundary of epoch i, so T[i] == gen_hi of epoch i+1 and
    # the youngest epoch runs down to the present. The dump has no row below the
    # last epoch, so that final 0 is appended here.
    t.append(0)
    return ne, t


# --------------------------------------------------------------------------- #
# Binomial sampling of the observed counts
# --------------------------------------------------------------------------- #
def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial (Clopper-Pearson) confidence interval on an allele frequency,
    obtained by inverting the exact binomial test.

    k > 0 -> the two-sided 1-alpha interval.
    k = 0 -> a one-sided interval, (0, upper], with upper = 1 - alpha**(1/n).
    """
    lo = float(beta.ppf(alpha / 2.0, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1.0 - (alpha / 2.0 if k > 0 else alpha), k + 1, n - k))
    return lo, hi


def freq_and_ci(k, n):
    """(frequency, was it observed, CI lower, CI upper) for one count.

    The frequency is k / n throughout, so a variant that was never seen returns
    0.0 rather than None. Use the second element to decide whether to draw the
    observed-frequency line.
    """
    lo, hi = clopper_pearson(k, n)
    return k / n, k > 0, lo, hi


def binom_logpmf(k, n, p):
    """log Binom(k | n, p) over an array of p, from the same primitives
    scipy.stats.binom.logpmf uses and numerically identical to it.

    xlogy and xlog1py apply the 0 * log(0) = 0 convention, so p = 0 with k = 0
    gives 0 rather than nan. The unconditional distributions contain exact
    zeros, and POLE and POLD1 have k = 0.
    """
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + xlogy(k, p) + xlog1py(n - k, -p))


# --------------------------------------------------------------------------- #
# Guard: the summary must have kept every replicate
# --------------------------------------------------------------------------- #
def summary_is_unconditional(npz_path) -> bool:
    """True if the companion .json says every replicate was kept.

    The summarisers record this as "conditioned_on_segregating": false. A
    summary that says true, or omits the key, kept only the replicates in which
    the mutator was present.
    """
    npz_path = str(npz_path)
    jpath = npz_path[:-4] + ".json" if npz_path.endswith(".npz") else npz_path + ".json"
    try:
        with open(jpath) as fh:
            summary = json.load(fh)
    except (OSError, ValueError):
        return False
    entries = [e for e in summary.values() if isinstance(e, dict)]
    return bool(entries) and all(e.get("conditioned_on_segregating") is False
                                 for e in entries)


def require_unconditional(npz_path, workflow: str) -> None:
    """Stop with a useful message rather than plotting the wrong replicates."""
    if not summary_is_unconditional(npz_path):
        raise SystemExit(
            f"\n{npz_path}\nwas summarised BEFORE the summarisers kept every "
            f"replicate, so the zeros are missing and this figure would be "
            f"silently conditional.\n\nRe-run:  snakemake --snakefile "
            f"workflow/{workflow} --profile workflow/profiles/slurm\n")


# --------------------------------------------------------------------------- #
# Drawing zeros on a log axis
# --------------------------------------------------------------------------- #
class FloorTracker:
    """Clip values to a log axis's floor, recording what that hid.

    A log axis cannot show 0. A box whose 2.5th percentile, lower quartile or
    median is 0 is drawn at the floor instead, and `report` prints its label and
    the fraction of replicates at zero.
    """

    def __init__(self, floor: float):
        self.floor = floor
        self.floored: list[tuple[str, float]] = []

    def __call__(self, arr, label):
        """Clip to the axis floor for drawing; record whether that hid anything."""
        arr = np.asarray(arr, dtype=float)
        if arr.size:
            lo = np.percentile(arr, [2.5, 25, 50])
            if (lo <= self.floor).any():
                self.floored.append((label, float((arr <= self.floor).mean())))
                return np.maximum(arr, self.floor), True
        return np.maximum(arr, self.floor), False

    def report(self, width: int = 22):
        if not self.floored:
            return
        print(f"\nnote: drawn at the axis floor ({self.floor:g}), because at least one of the "
              f"2.5th percentile,\n      lower quartile and median is 0 -- the box does not "
              f"reach the value shown:")
        for label, frac in self.floored:
            print(f"        {label:<{width}s} {100 * frac:5.1f}% of replicates are 0")
