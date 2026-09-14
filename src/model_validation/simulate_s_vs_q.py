#!/usr/bin/env python3
"""
Bottom row of the model-validation figure: the mutator's selection coefficient as a
function of the frequency the population has actually reached, for one dominance
coefficient h.

Each replicate draws its starting frequency q_init from Uniform(q_min, q_max), runs for
n_gens generations, and contributes one point:

    q_init    the sampled starting frequency
    q_current the mutator frequency at the END of the run
    s_obs     observed  = s_het * (excess deleterious DNMs linked to a mutator
                                   haplotype at the end of the run)
    s_exp     expected  = 2 * f * lambda(q_current) * s_het
                        = 2 * f * phi_G * (h + q_current(1 - 2h)) * s_het
              i.e. the theory evaluated at the CURRENT frequency, not the sampled one.

Replicates are independent, so a run can be split into chunks: --rep-start selects
which slice of the replicate index range this process computes, and each replicate's
RNG stream is keyed by (seed, replicate index).  Chunk boundaries therefore never
change the numbers, and chunks can be concatenated in replicate order.

Output npz:
    q_init, q_current, s_obs, s_exp, lam_del  (each length n_reps)
    h, N, phi_G, f, s_het, q_min, q_max, n_gens, seed, rep_start, n_reps   scalars

A replicate whose mutator is lost or fixed by the final generation has NaN in s_obs and
lam_del (q_current is still recorded, so those replicates are identifiable).

Example:
    python3 simulate_s_vs_q.py --h 0.25 --n-reps 10000 --out s_vs_q_h_0.25.npz
"""

import argparse
import time

import numpy as np

from mutator_wf import simulate_del, s_expected, make_rng


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h",         type=float, required=True, help="mutator dominance")
    p.add_argument("--out",       required=True, help="output .npz path")
    p.add_argument("--N",         type=int,   default=20000, help="diploid population size")
    p.add_argument("--phi-G",     type=float, default=100.0,
                   help="excess DNMs per gamete from a homozygous mutator")
    p.add_argument("--f",         type=float, default=0.08,
                   help="fraction of DNMs that are deleterious")
    p.add_argument("--s-het",     type=float, default=0.5 * 0.001,
                   help="fitness cost per excess deleterious DNM carried")
    p.add_argument("--q-min",     type=float, default=0.05)
    p.add_argument("--q-max",     type=float, default=0.95)
    p.add_argument("--n-gens",    type=int,   default=50)
    p.add_argument("--n-reps",    type=int,   default=10000,
                   help="replicates computed by THIS process")
    p.add_argument("--rep-start", type=int,   default=0,
                   help="index of the first replicate, for chunked runs")
    p.add_argument("--seed",      type=int,   default=1000)
    args = p.parse_args()

    n = args.n_reps
    q_init    = np.empty(n)
    q_current = np.empty(n)
    lam_del   = np.full(n, np.nan)
    s_obs     = np.full(n, np.nan)

    t0 = time.time()
    for i in range(n):
        rep = args.rep_start + i
        rng = make_rng(args.seed, rep)
        # drawn from this replicate's own stream, so it does not depend on the chunking
        q0 = rng.uniform(args.q_min, args.q_max)
        lam_tr, q_tr = simulate_del(
            args.N, args.phi_G, args.f, args.h, q0, args.n_gens, args.s_het, rng)
        q_init[i]    = q0
        q_current[i] = q_tr[-1]
        lam_del[i]   = lam_tr[-1]
        s_obs[i]     = args.s_het * lam_tr[-1]
        if (i + 1) % 100 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"[h={args.h:g} reps {args.rep_start}-{args.rep_start + n - 1}] "
                  f"{i + 1}/{n}, {el:.0f}s elapsed, "
                  f"{el / (i + 1):.3f}s per replicate, "
                  f"ETA {el / (i + 1) * (n - i - 1) / 60:.1f} min", flush=True)

    # theory evaluated at the frequency each replicate actually reached
    s_exp = s_expected(args.phi_G, args.f, args.h, q_current, args.s_het)

    np.savez_compressed(
        args.out,
        q_init=q_init, q_current=q_current, s_obs=s_obs, s_exp=s_exp, lam_del=lam_del,
        h=args.h, N=args.N, phi_G=args.phi_G, f=args.f, s_het=args.s_het,
        q_min=args.q_min, q_max=args.q_max, n_gens=args.n_gens, seed=args.seed,
        rep_start=args.rep_start, n_reps=n,
    )
    lost = int(np.isnan(s_obs).sum())
    print(f"[h={args.h:g}] wrote {args.out} in {time.time() - t0:.0f}s "
          f"({lost}/{n} replicates lost or fixed the mutator)", flush=True)


if __name__ == "__main__":
    main()
