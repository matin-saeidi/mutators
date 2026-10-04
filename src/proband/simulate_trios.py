#!/usr/bin/env python3
"""
Simulate trio studies (Supplementary Section S6; Figures 4 and S12). Each
replicate is one cohort of n trios: M mutator frequencies are drawn from
--freqs, parents' genotypes from Hardy-Weinberg proportions, and the script
records whether the cohort has a mutator parent and the probability that it has
a proband (Eqs. S32, S33).

--age-sex uses the parental age and sex model (Figure 4). Without it, every
parent has the same baseline mutation rate and a proband is a child with at
least 150 DNMs (Figure S12).

Usage:
    python3 src/proband/simulate_trios.py --freqs s_8.000000e-04.npz \\
        --key 8.000000e-04 --h 0 --phi-G 10 --M 10 --age-sex --seed 1 --out out.npz

Output npz:
    prob_proband          Pr(>= 1 proband), one value per cohort
    any_mutator_parent    1 if the cohort has a mutator parent, one value per cohort
    meta                  the arguments, and the means pr_proband and pr_mutator_parent
The same metadata is also written to a .json next to it.

If the run is stopped, run the same command again to continue from <out>.ckpt.
"""
import argparse
import json
import os
import pickle
import signal
import sys
import time

import numpy as np
from scipy.stats import poisson

from analytic import (AGE_F, AGE_M, dnms_father, dnms_mother, fold_minus_one,
                      proband_threshold)

G = 3e9
U_HAT = 1.25e-8
THRESHOLD_NO_AGE_SEX = int(2 * np.ceil(2 * G * U_HAT))       # 150


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", required=True,
                   help="npz of present-day mutator frequencies (src/summaries/summarize_single_site.py)")
    p.add_argument("--key", required=True, help="array in --freqs to use")
    p.add_argument("--h", type=float, required=True, choices=(0.0, 0.5),
                   help="dominance of the mutator")
    p.add_argument("--phi-G", type=float, required=True,
                   help="extra DNMs genome-wide transmitted by a mutator homozygote")
    p.add_argument("--M", type=int, required=True, help="number of modifier sites")
    p.add_argument("--n-trios", type=int, default=22000)
    p.add_argument("--reps", type=int, default=250000)
    p.add_argument("--age-sex", action="store_true",
                   help="parental age and sex effects, per-trio threshold")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--checkpoint-every", type=float, default=300,
                   help="seconds between checkpoints")
    return p.parse_args()


args = parse_args()
rng = np.random.default_rng(args.seed)
freqs = np.load(args.freqs)[args.key].astype(np.float64)
n, M, h = args.n_trios, args.M, args.h
phi = args.phi_G / G            # increase in per-site mutation rate, per homozygous site


def sample_frequencies():
    """M frequencies drawn from the simulated ones; sites at q = 0 are dropped."""
    q = freqs[rng.integers(0, freqs.size, M)]
    return q[q > 0]


def distinct_parents(counts):
    """For site j, counts[j] distinct parents drawn uniformly from n; parents
    may repeat across sites. Returns the parent index of every draw, site by site."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    site = np.repeat(np.arange(counts.size), counts).astype(np.int64)
    idx = rng.integers(0, n, total)
    for _ in range(64):
        key = site * n + idx
        order = np.argsort(key, kind="stable")
        tie = key[order][1:] == key[order][:-1]
        if not tie.any():
            break
        dup = order[1:][tie]
        idx[dup] = rng.integers(0, n, dup.size)
    return idx


def sample_dosages(qs):
    """Dosage of each of n parents. At each site the number of homozygotes is
    Binomial(n, q^2) and, when h > 0, the number of heterozygotes among the rest
    Binomial(n - n_hom, 2q / (1 + q)); carriers are then assigned to distinct
    parents."""
    dosage = np.zeros(n)
    if qs.size == 0:
        return dosage
    n_hom = rng.binomial(n, qs ** 2)
    if h == 0:
        np.add.at(dosage, distinct_parents(n_hom), 1.0)
        return dosage
    n_het = rng.binomial(n - n_hom, 2.0 * qs / (1.0 + qs))
    carriers = n_hom + n_het
    idx = distinct_parents(carriers)
    if idx.size == 0:
        return dosage
    # within each site's block of draws, the first n_hom are the homozygotes
    start = np.concatenate(([0], np.cumsum(carriers)[:-1]))
    pos = np.arange(idx.size) - np.repeat(start, carriers)
    is_hom = pos < np.repeat(n_hom, carriers)
    np.add.at(dosage, idx[is_hom], 1.0)
    np.add.at(dosage, idx[~is_hom], h)
    return dosage


def sample_ages(mean, sd, lo, hi):
    """n ages from a normal truncated to [lo, hi], by rejection."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        draw = rng.normal(mean, sd, int((n - filled) * 1.15) + 16)
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(draw.size, n - filled)
        out[filled:filled + take] = draw[:take]
        filled += take
    return out


def replicate():
    """One cohort: (Pr(>= 1 proband), whether it has a mutator parent)."""
    qs = sample_frequencies()
    d_f = sample_dosages(qs)
    d_m = sample_dosages(qs)
    S = float(np.sum(qs ** 2 + 2.0 * h * qs * (1.0 - qs))) if qs.size else 0.0

    if args.age_sex:
        nu_f = dnms_father(sample_ages(**AGE_F))
        nu_m = dnms_mother(sample_ages(**AGE_M))
        fm1 = fold_minus_one(args.phi_G)
        lam = ((1.0 + fm1 * d_f) * nu_f + (1.0 + fm1 * d_m) * nu_m) / (1.0 + fm1 * S)
        rho = poisson.sf(proband_threshold(nu_f, nu_m) - 1, lam)
    else:
        u0 = max(0.0, U_HAT - phi * S)
        lam = 2.0 * u0 * G + (d_f + d_m) * args.phi_G
        values, inverse = np.unique(lam, return_inverse=True)
        rho = poisson.sf(THRESHOLD_NO_AGE_SEX - 1, values)[inverse]

    if np.any(rho >= 1.0):
        prob = 1.0
    else:
        prob = float(-np.expm1(np.sum(np.log1p(-rho))))
    return prob, bool(np.any(d_f > 0) or np.any(d_m > 0))


def main():
    checkpoint = args.out + ".ckpt"
    prob_proband = np.zeros(args.reps)
    any_mutator_parent = np.zeros(args.reps, np.uint8)
    start = 0
    if os.path.exists(checkpoint):
        with open(checkpoint, "rb") as fh:
            state = pickle.load(fh)
        start = state["completed"]
        prob_proband[:start] = state["prob_proband"]
        any_mutator_parent[:start] = state["any_mutator_parent"]
        rng.bit_generator.state = state["rng"]
        print(f"resuming from {checkpoint} at replicate {start}", flush=True)

    def save_checkpoint(completed):
        tmp = checkpoint + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"completed": completed,
                         "prob_proband": prob_proband[:completed],
                         "any_mutator_parent": any_mutator_parent[:completed],
                         "rng": rng.bit_generator.state}, fh)
        os.replace(tmp, checkpoint)
        print(f"checkpoint written at replicate {completed}", flush=True)

    stop = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))

    t_last = time.time()
    for rep in range(start, args.reps):
        prob_proband[rep], any_mutator_parent[rep] = replicate()
        if stop:
            save_checkpoint(rep + 1)
            sys.exit(1)
        if time.time() - t_last >= args.checkpoint_every:
            save_checkpoint(rep + 1)
            t_last = time.time()

    meta = {k: v for k, v in vars(args).items() if k != "checkpoint_every"}
    meta["pr_proband"] = float(prob_proband.mean())
    meta["pr_mutator_parent"] = float(any_mutator_parent.mean())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, prob_proband=prob_proband.astype(np.float32),
                            any_mutator_parent=any_mutator_parent,
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"wrote {args.out}: Pr(proband) = {meta['pr_proband']:.4g}, "
          f"Pr(mutator parent) = {meta['pr_mutator_parent']:.4g}", flush=True)


if __name__ == "__main__":
    main()
