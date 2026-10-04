#!/usr/bin/env python3
"""
Run one replicate of the mutator simulation and save the mutator frequencies at
generations burn_in, burn_in + sample_every, burn_in + 2 sample_every, ...
(generations are numbered from 0). Defaults are the parameters of Figure 1.

Usage:
    python3 src/mutator_simulation/simulate.py --phi-G 100 --seed 1 --out rep_1.npz

Output npz:
    freqs         (n_samples, M)  mutator frequency at each modifier site
    generations   (n_samples,)    generation of each row
    N, M, phi_G, G, f, s_het, u0, mu, burn_in, sample_every, seed, n_generations

If the run is stopped (SIGTERM, e.g. at the wall-clock limit), run the same
command again to continue from <out>.ckpt.
"""
import argparse
import os
import pickle
import signal
import sys
import time

import numpy as np

from population import Population


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-G", type=float, required=True,
                   help="extra DNMs genome-wide transmitted by a mutator homozygote")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--N", type=int, default=2000, help="population size")
    p.add_argument("--M", type=int, default=1000, help="number of modifier sites")
    p.add_argument("--G", type=float, default=3e8, help="haploid genome length")
    p.add_argument("--f", type=float, default=0.08,
                   help="fraction of the genome under selection")
    p.add_argument("--s-het", type=float, default=0.005,
                   help="fitness cost of one deleterious allele")
    p.add_argument("--u0", type=float, default=1.25e-7,
                   help="per-site mutation rate without mutators")
    p.add_argument("--mu", type=float, default=1.25e-7,
                   help="mutation rate at modifier sites")
    p.add_argument("--generations", type=int, default=400000)
    p.add_argument("--burn-in", type=int, default=20000)
    p.add_argument("--sample-every", type=int, default=16000)
    p.add_argument("--checkpoint-every", type=float, default=1800,
                   help="seconds between checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = args.out + ".ckpt"

    if os.path.exists(checkpoint):
        with open(checkpoint, "rb") as fh:
            pop, start, samples, sample_gens = pickle.load(fh)
        print(f"resuming from {checkpoint} at generation {start}", flush=True)
    else:
        rng = np.random.RandomState(args.seed)
        pop = Population(args.N, args.M, args.phi_G, args.G, args.f, args.s_het,
                         args.u0, args.mu, rng)
        start, samples, sample_gens = 0, [], []

    def save_checkpoint(next_gen):
        tmp = checkpoint + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump((pop, next_gen, samples, sample_gens), fh)
        os.replace(tmp, checkpoint)
        print(f"checkpoint written at generation {next_gen}", flush=True)

    stop = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))

    t_last = time.time()
    for gen in range(start, args.generations):
        freqs = pop.next_generation()
        if gen >= args.burn_in and (gen - args.burn_in) % args.sample_every == 0:
            samples.append(freqs)
            sample_gens.append(gen)
            print(f"generation {gen}: mean mutator frequency {freqs.mean():.4g}, "
                  f"{len(pop.fixed)} sites fixed", flush=True)
        if stop:
            save_checkpoint(gen + 1)
            sys.exit(1)
        if time.time() - t_last >= args.checkpoint_every:
            save_checkpoint(gen + 1)
            t_last = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, freqs=np.array(samples), generations=np.array(sample_gens),
            N=args.N, M=args.M, phi_G=args.phi_G, G=args.G, f=args.f,
            s_het=args.s_het, u0=args.u0, mu=args.mu, burn_in=args.burn_in,
            sample_every=args.sample_every, seed=args.seed,
            n_generations=args.generations)
    os.replace(tmp, args.out)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
