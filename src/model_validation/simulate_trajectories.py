#!/usr/bin/env python3
"""
Top row of the model-validation figure: trajectories of mutator frequency and of the
excess DNMs linked to a mutator haplotype, for one dominance coefficient h.

Every replicate starts at the same mutator frequency q0 and is followed for n_gens
generations.  Raw (un-normalised) trajectories are saved; the figure normalises
lambda_tau by 2*lambda evaluated at the CURRENT q of each generation, and q_tau by q0,
but that is a plotting choice and is left to the plotting code.

Output npz:
    lam           (n_reps, n_gens+1)  lambda_tau, excess DNMs linked to a mutator haplotype
    lam_del       (n_reps, n_gens+1)  the same restricted to deleterious DNMs
    q             (n_reps, n_gens+1)  mutator frequency each generation
    two_lambda    (n_reps, n_gens+1)  2*lambda at the CURRENT q of that generation, i.e.
                                      the asymptote lambda_tau is heading for right then
    h, N, phi_G, f, s_het, q0, n_reps, n_gens, seed   scalars, for the record

Everything needed to normalise is in the file: lam / two_lambda is the ratio the figure
plots, and q / q0 the other one.

lam / lam_del are NaN in generations where the mutator is lost or fixed.

Example:
    python3 simulate_trajectories.py --h 0.25 --out trajectories_h_0.25.npz
"""

import argparse
import time

import numpy as np

from mutator_wf import simulate_dnms, two_lambda, make_rng


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h",      type=float, required=True, help="mutator dominance")
    p.add_argument("--out",    required=True, help="output .npz path")
    p.add_argument("--N",      type=int,   default=20000, help="diploid population size")
    p.add_argument("--phi-G",  type=float, default=100.0,
                   help="excess DNMs per gamete from a homozygous mutator")
    p.add_argument("--f",      type=float, default=0.08,
                   help="fraction of DNMs that are deleterious")
    p.add_argument("--s-het",  type=float, default=0.5 * 0.001,
                   help="fitness cost per excess deleterious DNM carried")
    p.add_argument("--q0",     type=float, default=0.5,
                   help="mutator frequency at the start of every replicate")
    p.add_argument("--n-reps", type=int,   default=50)
    p.add_argument("--n-gens", type=int,   default=200)
    p.add_argument("--seed",   type=int,   default=9000)
    args = p.parse_args()

    lam     = np.empty((args.n_reps, args.n_gens + 1))
    lam_del = np.empty((args.n_reps, args.n_gens + 1))
    q       = np.empty((args.n_reps, args.n_gens + 1))

    t0 = time.time()
    for r in range(args.n_reps):
        rng = make_rng(args.seed, r)
        lam[r], lam_del[r], q[r] = simulate_dnms(
            args.N, args.phi_G, args.f, args.h, args.q0, args.n_gens, args.s_het, rng)
        if (r + 1) % 10 == 0 or r + 1 == args.n_reps:
            el = time.time() - t0
            print(f"[h={args.h:g}] {r + 1}/{args.n_reps} replicates, "
                  f"{el:.0f}s elapsed, {el / (r + 1):.2f}s per replicate", flush=True)

    # the asymptote lambda_tau approaches, at the frequency reached in that generation
    two_lam = two_lambda(args.phi_G, args.h, q)

    np.savez_compressed(
        args.out,
        lam=lam, lam_del=lam_del, q=q,
        two_lambda=two_lam,
        h=args.h, N=args.N, phi_G=args.phi_G, f=args.f, s_het=args.s_het,
        q0=args.q0, n_reps=args.n_reps, n_gens=args.n_gens, seed=args.seed,
    )
    print(f"[h={args.h:g}] wrote {args.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
