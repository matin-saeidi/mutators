#!/usr/bin/env python3
"""
Combine per-selection-coefficient summaries into one npz + one json.

The varying-s grid summarises each s value on its own, so that the very large
raw .out file for that s can be reclaimed immediately instead of all of them
having to sit on disk at once (the XPC grid alone would otherwise be ~340 GB).
This step then stitches the per-s summaries into the single npz the analysis
actually wants.

Each input is a per-s .npz; the matching .json is found by swapping the
extension. Keys are merged as-is, so the final npz is keyed by selection
coefficient exactly as the per-s summarisers wrote them:

    single site / XPC : "<s>"
    MUTYH             : "<variant>_freq_s=<s>"

A key appearing in two inputs is a bug (two s values collapsing to the same
string), so it is a hard error rather than a silent overwrite.
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_files", nargs="+", help="per-s .npz files")
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    merged = {}
    origin = {}
    summary = {}

    for path in sorted(args.npz_files):
        with np.load(path) as z:
            for k in z.files:
                if k in merged:
                    raise SystemExit(
                        f"duplicate npz key {k!r}: present in both {origin[k]} and "
                        f"{path}. Two s values probably format to the same string.")
                merged[k] = z[k]
                origin[k] = path

        jpath = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(jpath):
            raise SystemExit(f"missing companion json for {path}: {jpath}")
        with open(jpath) as fh:
            entry = json.load(fh)
        for k, v in entry.items():
            if k in summary:
                raise SystemExit(f"duplicate json key {k!r} while reading {jpath}")
            summary[k] = v

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **merged)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    total = sum(a.nbytes for a in merged.values())
    print(f"combined {len(args.npz_files)} per-s summaries")
    print(f"  {len(merged)} npz keys, {total/1024**2:.1f} MB uncompressed")
    print(f"  wrote {args.out_npz}")
    print(f"  wrote {args.out_json}  ({len(summary)} entries)")


if __name__ == "__main__":
    main()
