#!/usr/bin/env python3
"""
Summarise compound-heterozygote lineage-tracker output
(src/simulations comp_het_lineage_tracker).

The simulator tracks every LoF lineage in the gene, so the simulated allele is
the *gene-wide* LoF allele with rate --gene-mu. Each surviving lineage is then
assigned to one of the focal mutator variants with probability
mu_variant / mu_gene:

  * XPC   : one focal variant, mu_var/mu_gene << 1, so most lineages are other
            LoF alleles and are discarded.
  * MUTYH : three focal variants whose rates sum to the gene rate, so every
            lineage is assigned to one of them.

Per run, the assigned lineage frequencies are summed per variant to give that
run's variant frequency q_v.


NOTHING IS CONDITIONED AWAY HERE
--------------------------------
EVERY simulated replicate is recorded, in run order, for every variant. A
replicate in which a variant got no lineage -- including a replicate in which
the gene carries no surviving lineage at all -- contributes q_v = 0. So for N
simulated runs and V variants the npz holds V arrays of exactly N entries, and
entry i of every one of them is the same run i. That alignment is the whole
point: it lets the conditioning be chosen downstream instead of being baked in
here.

    z = np.load(npz)
    y = z["Y179C_freq_s=0.00199795"]
    v = z["V234M_freq_s=0.00199795"]
    g = z["G368D_freq_s=0.00199795"]

    y                                  # unconditional
    y[y > 0]                           # Y179C conditional on ITSELF segregating
    keep = (y > 0) & (v > 0) & (g > 0)  # runs where all three segregate
    y[keep], v[keep], g[keep]           # the old "require all" run set

The three replaced --require modes are all recoverable by filtering:
    old "any"  ->  (q_1 + ... + q_V) > 0
    old "all"  ->  (q_1 > 0) & ... & (q_V > 0)
    old "none" ->  no filter, i.e. what is saved

Output:
  * .npz  : "<variant>_freq_s=<s>" -> float32, ONE ENTRY PER SIMULATED RUN
            "<variant>_age_s=<s>"  -> int32,   one entry per assigned lineage
            (the age arrays are per LINEAGE, so they are NOT run-aligned with
            the frequency arrays; use summarize_ages.py for per-run ages)
            When there is exactly one variant, the unprefixed aliases
            "freq_s=<s>" / "age_s=<s>" are also written.
  * .json : summary statistics per s value and per variant. The statistics at
            the top of each variant's block describe the array as saved, i.e.
            they are UNCONDITIONAL. The nested "segregating" block holds the
            same statistics over that variant's q > 0 subset, and (with more
            than one variant) "all_segregating" holds them over the runs in
            which every variant segregates.
"""

import argparse
import json
import os
import re
from array import array

import numpy as np

LINEAGE_RE = re.compile(r"age=\s*(\d+).*?freq=\s*([0-9.eE+-]+)", re.I)
RUN_RE = re.compile(r"^Run\s+\d+\s+survivors\s+at\s+present:", re.I)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infiles", nargs="+",
                   help="simulator .out files, named s_<value>.out")
    p.add_argument("--gene-mu", required=True,
                   help="gene-wide LoF mutation rate used as the denominator of the "
                        "lineage->variant probabilities. Pass a number, or the literal "
                        "'sum' to use the sum of the --variant rates (what the MUTYH "
                        "script does, so that the probabilities sum to exactly 1). Note "
                        "this may differ in the last digit from the possibly-rounded rate "
                        "handed to the simulator.")
    p.add_argument("--variant", action="append", required=True, metavar="NAME:MU",
                   help="focal variant name and its mutation rate; repeatable")
    p.add_argument("--seed", type=int, default=20250810,
                   help="seed for the lineage->variant assignment (fixed for reproducibility)")
    p.add_argument("--npz-key-style", choices=["prefixed", "bare"], default="prefixed",
                   help="'prefixed' (default): '<variant>_freq_s=<s>' plus an unprefixed "
                        "'freq_s=<s>' alias when there is a single variant. "
                        "'bare': key the frequency array by the selection coefficient "
                        "alone, '<s>', matching summarize_single_site.py. 'bare' requires "
                        "exactly one variant, since with several the key would be ambiguous.")
    p.add_argument("--key", default=None,
                   help="npz key to use instead of the s value parsed from the filename. "
                        "Needed when the grid varies something other than s (e.g. a "
                        "dominance sweep, where the key should be h). Requires exactly "
                        "one input file, since one key cannot label several arrays.")
    p.add_argument("--no-ages", action="store_true",
                   help="do not write the per-lineage age arrays; keeps the npz small when "
                        "only the frequency distributions are wanted")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--gene", default=None)
    p.add_argument("--population", default=None)
    p.add_argument("--meta", action="append", default=[],
                   help="extra key=value metadata to record in the JSON; repeatable")
    return p.parse_args()


def s_from_filename(path):
    base = os.path.basename(path)
    if not (base.startswith("s_") and base.endswith(".out")):
        raise ValueError(f"unexpected filename {base!r}; expected s_<value>.out")
    return base[2:-4]


def summarize(run_sums):
    arr = np.asarray(run_sums, dtype=float)
    n = arr.size
    if n == 0:
        return {}
    return {
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
        "n_runs": int(n),
    }


def iter_runs(path):
    """Yield one list of (freq, age) per run in the file."""
    current = []
    started = False
    with open(path) as f:
        # Read in 4 MB gulps. Text-mode iteration otherwise issues 8 KiB
        # read() syscalls, which is punishing on Lustre for a multi-GB .out.
        f._CHUNK_SIZE = 4 << 20
        for line in f:
            if RUN_RE.match(line):
                if started:
                    yield current
                current = []
                started = True
                continue
            m = LINEAGE_RE.search(line)
            if m:
                age = int(m.group(1))
                freq = float(m.group(2))
                if freq > 0:
                    current.append((freq, age))
    if started:
        yield current


def main():
    args = parse_args()

    meta = {}
    for kv in args.meta:
        if "=" not in kv:
            raise SystemExit(f"--meta expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        meta[k] = v

    # ---- variant probabilities -------------------------------------------
    names, mus = [], []
    for spec in args.variant:
        if ":" not in spec:
            raise SystemExit(f"--variant expects NAME:MU, got {spec!r}")
        name, mu = spec.rsplit(":", 1)
        names.append(name)
        mus.append(float(mu))

    if str(args.gene_mu).strip().lower() == "sum":
        gene_mu = float(sum(mus))
    else:
        gene_mu = float(args.gene_mu)
    if gene_mu <= 0:
        raise SystemExit(f"--gene-mu must be positive, got {gene_mu!r}")

    probs = np.asarray(mus, dtype=float) / gene_mu
    if np.any(probs < 0):
        raise SystemExit(f"negative variant probability: {dict(zip(names, probs))}")
    total_p = probs.sum()
    # Allow for the gene rate having been rounded when it was handed to the
    # simulator (e.g. MUTYH: variants sum to 2.90293e-08, simulator got
    # 2.9029e-08), but refuse anything that is a genuine inconsistency.
    if total_p > 1.0 + 1e-4:
        raise SystemExit(
            f"variant rates sum to {sum(mus):.8g} > gene rate {gene_mu:.8g} "
            f"(probabilities sum to {total_p:.8f}); pass --gene-mu sum if the "
            f"focal variants are meant to account for the whole gene rate")
    clipped = False
    if total_p > 1.0:
        clipped = True

    # Cumulative edges; anything at or beyond the last edge is "some other LoF
    # allele in the gene" and is discarded.
    edges = np.cumsum(probs)
    if clipped:
        edges[-1] = 1.0

    print(f"gene_mu = {gene_mu:.8g}"
          + (" (sum of variant rates)" if str(args.gene_mu).strip().lower() == "sum" else ""))
    for nm, mu, pr in zip(names, mus, probs):
        print(f"  variant {nm:>8s}: mu = {mu:.6g}  p = {pr:.6f}")
    print(f"  sum of variant probabilities = {total_p:.8f}"
          f"{'  (remainder discarded as other LoF alleles)' if total_p < 1 - 1e-9 else ''}"
          f"{'  (last edge clipped to 1: gene rate was rounded)' if clipped else ''}")
    print(f"  every replicate is kept (q = 0 where a variant is absent), seed = {args.seed}\n")

    if args.npz_key_style == "bare" and len(names) != 1:
        raise SystemExit(
            f"--npz-key-style bare needs exactly one --variant, got {len(names)}: "
            f"{names}. With several variants a bare '<s>' key would be ambiguous.")

    rng = np.random.default_rng(args.seed)

    summary_by_s = {}
    freq_data = {}   # (variant, s) -> array over ALL runs
    age_data = {}    # (variant, s) -> array over assigned lineages

    if args.key is not None and len(args.infiles) != 1:
        raise SystemExit(f"--key needs exactly one input file, got {len(args.infiles)}")

    for path in sorted(args.infiles):
        s_str = args.key if args.key is not None else s_from_filename(path)

        # array('d') rather than a list: 8 bytes per run instead of ~32, which
        # matters at 4e6 runs x 3 variants
        run_freqs = {nm: array("d") for nm in names}
        all_ages = {nm: array("l") for nm in names}
        n_runs_total = 0
        n_runs_with_lineage = 0

        for lineages in iter_runs(path):
            n_runs_total += 1

            if lineages:
                n_runs_with_lineage += 1
                freqs = np.fromiter((f for f, _ in lineages), dtype=float, count=len(lineages))
                ages = np.fromiter((a for _, a in lineages), dtype=np.int64, count=len(lineages))

                u = rng.random(freqs.size)
                # assign each lineage to a variant (or to "other" if u >= edges[-1])
                masks = []
                lo = 0.0
                for hi in edges:
                    masks.append((u >= lo) & (u < hi))
                    lo = hi

                for nm, m in zip(names, masks):
                    run_freqs[nm].append(float(freqs[m].sum()))
                    all_ages[nm].extend(ages[m].tolist())
            else:
                # a replicate in which the gene carries no surviving lineage at
                # all. That is a real observation of q = 0 for every variant,
                # so it is recorded rather than dropped.
                for nm in names:
                    run_freqs[nm].append(0.0)

        arrays = {nm: np.frombuffer(run_freqs[nm], dtype=np.float64) for nm in names}
        # runs in which every named variant segregates -- the old --require all
        all_seg = np.ones(n_runs_total, dtype=bool)
        any_seg = np.zeros(n_runs_total, dtype=bool)
        for nm in names:
            positive = arrays[nm] > 0
            all_seg &= positive
            any_seg |= positive

        entry = {
            "npz_key": s_str,
            # only meaningful when the key IS the selection coefficient; for a
            # dominance sweep the caller passes s and h through --meta instead
            **({} if args.key is not None else {"selection_coefficient": float(s_str)}),
            "gene_mu": gene_mu,
            "seed": args.seed,
            # every replicate is saved, so the arrays are aligned run-for-run
            # across variants and conditioning is done downstream
            "conditioned_on_segregating": False,
            "run_index_aligned": True,
            "n_runs_total": int(n_runs_total),
            "n_runs_with_any_lineage": int(n_runs_with_lineage),
            "n_runs_any_variant_segregating": int(any_seg.sum()),
            "n_runs_all_variants_segregating": int(all_seg.sum()),
            "p_any_variant_segregating":
                (float(any_seg.sum()) / n_runs_total) if n_runs_total else None,
            "p_all_variants_segregating":
                (float(all_seg.sum()) / n_runs_total) if n_runs_total else None,
            "variant_probabilities": {nm: float(p) for nm, p in zip(names, probs)},
            "source_file": os.path.abspath(path),
        }
        if args.gene:
            entry["gene"] = args.gene
        if args.population:
            entry["population"] = args.population
        entry.update(meta)

        for nm in names:
            arr = arrays[nm]
            seg = arr[arr > 0]
            stats = summarize(arr)
            stats.update({
                "n_runs_segregating": int(seg.size),
                "p_segregating": (float(seg.size) / n_runs_total) if n_runs_total else None,
                # this variant conditional on ITSELF segregating
                "segregating": summarize(seg),
            })
            if len(names) > 1:
                # conditional on ALL variants segregating: the run set the old
                # --require all pipeline reported
                stats["all_segregating"] = summarize(arr[all_seg])
            entry[nm] = stats
            freq_data[(nm, s_str)] = arr.astype(np.float32)
            age_data[(nm, s_str)] = np.asarray(all_ages[nm], dtype=np.int32)

        summary_by_s[s_str] = entry

        print(f"{os.path.basename(path)}: {n_runs_total} runs, all saved "
              f"({entry['n_runs_with_any_lineage']} with >=1 surviving lineage, "
              f"{entry['n_runs_all_variants_segregating']} with every variant segregating)")
        for nm in names:
            st = entry[nm]
            print(f"    {nm:>8s}: mean = {st.get('mean')}  "
                  f"mean|seg = {st['segregating'].get('mean')}  "
                  f"p_seg = {st.get('p_segregating')}  "
                  f"n_lineages = {age_data[(nm, s_str)].size}")

    # ---- write outputs ----------------------------------------------------
    payload = {}
    if args.npz_key_style == "bare":
        # frequency array keyed by the selection coefficient alone
        for (_, s_str), arr in freq_data.items():
            payload[s_str] = arr
        if not args.no_ages:
            for (_, s_str), arr in age_data.items():
                payload[f"age_s={s_str}"] = arr
    else:
        for (nm, s_str), arr in freq_data.items():
            payload[f"{nm}_freq_s={s_str}"] = arr
        if not args.no_ages:
            for (nm, s_str), arr in age_data.items():
                payload[f"{nm}_age_s={s_str}"] = arr
        if len(names) == 1:
            # unprefixed aliases, for the single-variant case
            for (_, s_str), arr in freq_data.items():
                payload[f"freq_s={s_str}"] = arr
            if not args.no_ages:
                for (_, s_str), arr in age_data.items():
                    payload[f"age_s={s_str}"] = arr

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **payload)
    print(f"\nSaved frequency and age arrays to {args.out_npz}")
    print(f"  npz keys: {sorted(payload)}")

    with open(args.out_json, "w") as fh:
        json.dump(summary_by_s, fh, indent=2)
    print(f"Wrote summary statistics to {args.out_json}")


if __name__ == "__main__":
    main()
