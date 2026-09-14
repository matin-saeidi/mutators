#!/usr/bin/env python3
"""
Combine the per-mutator age summaries into one npz + one json + one tsv.

Each input is a per-mutator .npz written by summarize_ages.py; the matching
.json is found by swapping the extension. The mutator label is read from the
JSON ("gene"), not from the filename, so the label in the combined product is
always the one the summariser actually recorded.

Unlike combine_summaries.py, which merges a selection-coefficient grid for a
single gene and therefore keys everything by s, this merges different mutators
that may share an s (all six neutral runs are at s = 0). Every npz key is
therefore namespaced by the mutator label:

    "<label>__<variant>_maxage_s=<s>"   int32    per-run allele age, generations
    "<label>__<variant>_freq_s=<s>"     float32  that run's variant frequency
    "<label>__<variant>_nlin_s=<s>"     int32    lineages assigned to the variant

The tsv is a flat one-row-per-(mutator, variant) table of the age summary, for
reading straight into a plot without touching the JSON.
"""

import argparse
import json
import os

import numpy as np

TSV_COLUMNS = [
    "label", "variant", "s", "mut_rate", "population",
    "n_runs_total", "n_runs_with_variant", "p_variant_present",
    "mean_lineages_per_run",
    "mean_age", "median_age", "std_error",
    "p2.5", "p25", "p75", "p97.5", "min_age", "max_age",
    "frac_censored_at_burn_in",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_files", nargs="+", help="per-mutator .npz files from summarize_ages.py")
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-tsv", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    merged = {}
    summary = {}
    rows = []

    for path in sorted(args.npz_files):
        jpath = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(jpath):
            raise SystemExit(f"missing companion json for {path}: {jpath}")
        with open(jpath) as fh:
            entry_by_s = json.load(fh)

        labels = {e.get("gene") for e in entry_by_s.values()}
        if labels != {l for l in labels if l}:
            raise SystemExit(f"{jpath}: every entry needs a 'gene' label")
        if len(labels) != 1:
            raise SystemExit(f"{jpath}: expected one mutator label, got {sorted(labels)}")
        label = labels.pop()
        if label in summary:
            raise SystemExit(f"duplicate mutator label {label!r} (second one from {jpath})")

        summary[label] = entry_by_s

        with np.load(path) as z:
            for k in z.files:
                merged[f"{label}__{k}"] = z[k]

        for s_str, e in entry_by_s.items():
            for variant, p in e["variant_probabilities"].items():
                st = e.get(variant, {})
                pct = st.get("percentiles", {})
                rows.append({
                    "label": label,
                    "variant": variant,
                    "s": s_str,
                    "mut_rate": e.get("simulator_mut_rate", e.get("gene_mu")),
                    "population": e.get("population", ""),
                    "n_runs_total": e.get("n_runs_total"),
                    "n_runs_with_variant": st.get("n_runs_with_variant"),
                    "p_variant_present": st.get("p_variant_present"),
                    "mean_lineages_per_run": e.get("mean_lineages_per_run"),
                    "mean_age": st.get("mean"),
                    "median_age": st.get("median"),
                    "std_error": st.get("std_error"),
                    "p2.5": pct.get("2.5"),
                    "p25": pct.get("25"),
                    "p75": pct.get("75"),
                    "p97.5": pct.get("97.5"),
                    "min_age": st.get("minimum"),
                    "max_age": st.get("maximum"),
                    "frac_censored_at_burn_in": st.get("frac_censored_at_burn_in"),
                })

    for out in (args.out_npz, args.out_json, args.out_tsv):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    np.savez_compressed(args.out_npz, **merged)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.out_tsv, "w") as fh:
        fh.write("\t".join(TSV_COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else str(r[c]) for c in TSV_COLUMNS) + "\n")

    total = sum(a.nbytes for a in merged.values())
    print(f"combined {len(args.npz_files)} per-mutator age summaries "
          f"({len(summary)} labels, {len(rows)} mutator x variant rows)")
    print(f"  {len(merged)} npz keys, {total/1024**2:.1f} MB uncompressed")
    print(f"  wrote {args.out_npz}")
    print(f"  wrote {args.out_json}")
    print(f"  wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
