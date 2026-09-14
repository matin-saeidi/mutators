#!/usr/bin/env python3
"""
Summarise single-site simulator output (src/simulations single_site_simulator).

Paths and metadata are taken on the command line.

Input format: one line per run, whose last four whitespace-separated tokens are
    <hom_derived> <het> <hom_ancestral> <mutU>
so the derived allele frequency of a run is (2*hom_derived + het) / (2*total).

EVERY simulated run is recorded, including the ones in which the mutator was
lost, which enter the array as a frequency of exactly 0. The saved array is
therefore the UNCONDITIONAL frequency distribution and has exactly
n_runs_total entries. Conditioning on the mutator segregating is a downstream
one-liner:

    q = np.load(npz)["0.0102"]
    q_seg = q[q > 0]          # conditional on segregating
    q                         # unconditional

Nothing is thrown away at this stage, so both analyses come out of one set of
simulations.

Output:
  * .npz  : one float32 array per s value, keyed by the *plain* s string
            (e.g. "0.0102") -- the same convention as the existing
            sum_<GENE>_single_site*.py scripts and the plotting notebook.
  * .json : summary statistics per s value, plus the run metadata. The
            top-level statistics describe the array as saved, i.e. they are
            UNCONDITIONAL; the nested "segregating" block holds the same
            statistics over the q > 0 subset.
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infiles", nargs="+",
                   help="simulator .out files, named s_<value>.out")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--gene", default=None, help="gene/mutator label, recorded in the JSON")
    p.add_argument("--population", default=None, help="demography label, recorded in the JSON")
    p.add_argument("--meta", action="append", default=[],
                   help="extra key=value metadata to record in the JSON; repeatable")
    p.add_argument("--key", default=None,
                   help="npz key to use instead of the s value parsed from the filename. "
                        "Needed when the grid varies something other than s (e.g. a "
                        "dominance sweep, where the key should be h). Requires exactly "
                        "one input file, since one key cannot label several arrays.")
    return p.parse_args()


def s_from_filename(path):
    """'s_0.0102.out' -> '0.0102'"""
    base = os.path.basename(path)
    if not (base.startswith("s_") and base.endswith(".out")):
        raise ValueError(f"unexpected filename {base!r}; expected s_<value>.out")
    return base[2:-4]


def read_freqs(path):
    """Return (all_freqs, n_parse_failures). One frequency per simulated run."""
    freqs = []
    skipped = 0
    with open(path) as f:
        # Read in 4 MB gulps. Text-mode iteration otherwise issues 8 KiB
        # read() syscalls, which is punishing on Lustre for a multi-GB .out.
        f._CHUNK_SIZE = 4 << 20
        for line in f:
            parts = line.split()
            # Last four tokens are: hom_derived, het, hom_ancestral, mutU
            try:
                hom_derived = int(parts[-4])
                het_count = int(parts[-3])
                hom_anc = int(parts[-2])
            except (IndexError, ValueError):
                skipped += 1
                continue
            total = hom_derived + het_count + hom_anc
            if total > 0:
                freqs.append((2 * hom_derived + het_count) / (2 * total))
            else:
                skipped += 1
    return np.asarray(freqs, dtype=float), skipped


def summarize(arr):
    n = arr.size
    if n == 0:
        return {}
    out = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "variance": float(np.var(arr, ddof=1)) if n > 1 else None,
        "std_error": float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "ci_lower_50": float(np.percentile(arr, 25)),
        "ci_upper_50": float(np.percentile(arr, 75)),
        "maximum": float(arr.max()),
        "minimum": float(arr.min()),
        "n": int(n),
    }
    return out


def main():
    args = parse_args()

    meta = {}
    for kv in args.meta:
        if "=" not in kv:
            raise SystemExit(f"--meta expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        meta[k] = v

    summary = {}
    freq_data = {}

    if args.key is not None and len(args.infiles) != 1:
        raise SystemExit(f"--key needs exactly one input file, got {len(args.infiles)}")

    for path in sorted(args.infiles):
        s_str = args.key if args.key is not None else s_from_filename(path)
        all_freqs, skipped = read_freqs(path)

        n_total = all_freqs.size
        seg = all_freqs[all_freqs > 0]
        n_seg = seg.size

        # the array as saved is unconditional: every run, zeros included
        freq_data[s_str] = all_freqs.astype(np.float32)

        stats = summarize(all_freqs)
        stats.update({
            "npz_key": s_str,
            # only meaningful when the key IS the selection coefficient; for a
            # dominance sweep the caller passes s and h through --meta instead
            **({} if args.key is not None else {"selection_coefficient": float(s_str)}),
            "n_runs_total": int(n_total),
            "n_runs_segregating": int(n_seg),
            "p_segregating": (float(n_seg) / n_total) if n_total else None,
            # the array holds one entry per simulated run, in run order
            "conditioned_on_segregating": False,
            # statistics of the q > 0 subset, i.e. what the old conditional
            # summaries reported; recover the subset itself with q[q > 0]
            "segregating": summarize(seg),
            "unparsed_lines": int(skipped),
            "source_file": os.path.abspath(path),
        })
        if args.gene:
            stats["gene"] = args.gene
        if args.population:
            stats["population"] = args.population
        stats.update(meta)

        summary[s_str] = stats

        print(f"{os.path.basename(path)}: {n_total} runs kept (all of them), "
              f"{n_seg} segregating ({100.0 * n_seg / max(n_total, 1):.2f}%), "
              f"mean = {stats.get('mean')}, "
              f"mean|seg = {stats['segregating'].get('mean')}")
        if skipped:
            print(f"  note: {skipped} lines could not be parsed and were skipped")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **freq_data)
    print(f"\nSaved frequency arrays to {args.out_npz}")
    print(f"  npz keys: {sorted(freq_data)}")

    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote summary statistics to {args.out_json}")


if __name__ == "__main__":
    main()
