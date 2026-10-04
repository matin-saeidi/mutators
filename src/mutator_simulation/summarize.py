#!/usr/bin/env python3
"""
Pool the replicates of the mutator simulation at each phi_G (all saved
frequencies of all replicates, n values of q) and write the statistics plotted
in Figures 1, S2 and S3.

Usage:
    python3 src/mutator_simulation/summarize.py rep_*.npz --out summary.json

Output JSON, keyed by phi_G ("%.4e"):
    phi_G, n_replicates, n
    mean_q,  se_mean_q       mean of q,   std(q) / sqrt(n)
    mean_q2, se_mean_q2      mean of q^2, std(q^2) / sqrt(n)
    var_q,   se_var_q        var(q),      std((q - mean_q)^2) / sqrt(n)
    N, M, G, f, s_het, u0, mu, burn_in, sample_every, n_generations
"""
import argparse
import collections
import json

import numpy as np

PARAMS = ("N", "M", "G", "f", "s_het", "u0", "mu", "burn_in", "sample_every",
          "n_generations")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replicates", nargs="+", help="npz files written by simulate.py")
    p.add_argument("--out", required=True, help="output .json")
    args = p.parse_args()

    pooled = collections.defaultdict(list)
    params = {}
    for path in args.replicates:
        with np.load(path) as z:
            key = f"{float(z['phi_G']):.4e}"
            pooled[key].append(z["freqs"].ravel())
            these = {k: z[k].item() for k in PARAMS}
        if params.setdefault(key, these) != these:
            raise SystemExit(f"{path}: parameters differ from other replicates at phi_G = {key}")

    summary = {}
    for key in sorted(pooled, key=float):
        q = np.concatenate(pooled[key])
        n = q.size
        mean = q.mean()
        summary[key] = {
            "phi_G": float(key),
            "n_replicates": len(pooled[key]),
            "n": n,
            "mean_q": mean,
            "se_mean_q": np.std(q, ddof=1) / np.sqrt(n),
            "mean_q2": np.mean(q ** 2),
            "se_mean_q2": np.std(q ** 2, ddof=1) / np.sqrt(n),
            "var_q": np.var(q, ddof=1),
            "se_var_q": np.std((q - mean) ** 2, ddof=1) / np.sqrt(n),
            **params[key],
        }
        summary[key] = {k: (v.item() if isinstance(v, np.generic) else v)
                        for k, v in summary[key].items()}
        print(f"phi_G = {key}: {len(pooled[key])} replicates, n = {n}, E[q] = {mean:.4e}")

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
