#!/usr/bin/env python3
"""
Age distribution of a mutator allele from compound-het lineage-tracker output
(comp_het_lineage_tracker).

Companion to summarize_comp_het.py. That script summarises the *frequency*
distribution and, when asked, dumps every surviving lineage's age in one flat
array. This script instead answers the question "how old is the mutator allele
in this replicate?", which is a per-run quantity:

    age of the mutator in a run = age of the OLDEST surviving lineage
                                  that is assigned to that mutator

Every mutator is simulated through the compound-het lineage tracker, including
the ones that are single sites in the frequency pipeline (MPG, POLD1, POLE).
For those the mutation rate handed to the simulator IS the single-variant rate,
so mu_variant / mu_gene = 1 and every lineage belongs to the focal variant --
i.e. the run's age is simply the oldest surviving lineage, full stop.

For XPC and MUTYH the simulated allele is the gene-wide LoF allele, so each
surviving lineage is first assigned to a focal variant with probability
mu_variant / mu_gene (identical scheme, and identical default seed, to
summarize_comp_het.py) and the oldest lineage *assigned to that variant* is
taken. XPC's single focal variant takes ~1.8% of lineages and the rest are
discarded as other LoF alleles; MUTYH's three variants exhaust the gene rate,
so every lineage is assigned to one of them.

Conditioning is per variant: a run contributes to a variant's age distribution
iff at least one lineage in it was assigned to that variant. The three MUTYH
variants therefore have three independent (overlapping) run sets, and unlike
summarize_comp_het.py -- which keeps every replicate and records q = 0 where a
variant is absent -- this script cannot simply keep them all: a run in which the
variant never arose has no age, not an age of zero. Nor is the run set narrowed
to the replicates where all three variants segregate, which would both bias the
ages and throw away ~75% of the usable runs.

Ages are in GENERATIONS before the present. They are censored above at the
burn-in start generation: a lineage born in the very first simulated generation
has age == burn_in. Pass --burn-in to have the censored fraction counted and
reported; it matters only for s = 0, where a back-mutation-fixed lineage can be
as old as the simulation itself.

Output:
  * .npz : "<variant>_maxage_s=<s>"  int32   one entry per run containing the variant
           "<variant>_freq_s=<s>"    float32 that run's summed variant frequency
           "<variant>_nlin_s=<s>"    int32   lineages assigned to the variant in that run
           With exactly one variant the unprefixed aliases "maxage_s=<s>",
           "freq_s=<s>" and "nlin_s=<s>" are written too.
           The three arrays for a variant are aligned run-for-run.
  * .json : age summary statistics per s value and per variant.
"""

import argparse
import json
import os
import re

import numpy as np

LINEAGE_RE = re.compile(r"age=\s*(\d+).*?freq=\s*([0-9.eE+-]+)", re.I)
RUN_RE = re.compile(r"^Run\s+\d+\s+survivors\s+at\s+present:", re.I)

AGE_PERCENTILES = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infiles", nargs="+",
                   help="simulator .out files, named s_<value>.out")
    p.add_argument("--gene-mu", required=True,
                   help="gene-wide mutation rate handed to the simulator, the denominator "
                        "of the lineage->variant probabilities. Pass a number, or the "
                        "literal 'sum' to use the sum of the --variant rates (MUTYH, whose "
                        "three variants account for the whole gene rate). For a single-site "
                        "mutator pass its own rate, so the probability is 1.")
    p.add_argument("--variant", action="append", required=True, metavar="NAME:MU",
                   help="focal variant name and its mutation rate; repeatable")
    p.add_argument("--seed", type=int, default=20250810,
                   help="seed for the lineage->variant assignment. The default matches "
                        "summarize_comp_het.py's default, so a run summarised by both "
                        "scripts gets the same assignment.")
    p.add_argument("--burn-in", type=float, default=None,
                   help="burn-in start generation, i.e. the age at which the simulation "
                        "censors. Only used to report the censored fraction.")
    p.add_argument("--key", default=None,
                   help="npz key to use instead of the s value parsed from the filename. "
                        "Requires exactly one input file.")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--gene", default=None, help="mutator label, recorded in the JSON")
    p.add_argument("--population", default=None, help="demography label, recorded in the JSON")
    p.add_argument("--meta", action="append", default=[],
                   help="extra key=value metadata to record in the JSON; repeatable")
    return p.parse_args()


def s_from_filename(path):
    base = os.path.basename(path)
    if not (base.startswith("s_") and base.endswith(".out")):
        raise ValueError(f"unexpected filename {base!r}; expected s_<value>.out")
    return base[2:-4]


def summarize_ages(ages, burn_in):
    """Summary statistics of one variant's per-run allele ages, in generations."""
    arr = np.asarray(ages, dtype=float)
    n = arr.size
    if n == 0:
        return {}
    out = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "variance": float(np.var(arr, ddof=1)) if n > 1 else None,
        "std_error": float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
        "n_runs": int(n),
    }
    q = np.percentile(arr, AGE_PERCENTILES)
    out["percentiles"] = {str(p): float(v) for p, v in zip(AGE_PERCENTILES, q)}
    # the 95% and 50% intervals under the names the frequency summaries use
    out["ci_lower"] = out["percentiles"]["2.5"]
    out["ci_upper"] = out["percentiles"]["97.5"]
    out["ci_lower_50"] = out["percentiles"]["25"]
    out["ci_upper_50"] = out["percentiles"]["75"]
    if burn_in is not None:
        # a lineage born in the first simulated generation has age == burn_in;
        # anything at that ceiling is censored, not measured
        n_cens = int(np.count_nonzero(arr >= burn_in))
        n_near = int(np.count_nonzero(arr >= 0.99 * burn_in))
        out["burn_in"] = float(burn_in)
        out["n_censored_at_burn_in"] = n_cens
        out["frac_censored_at_burn_in"] = float(n_cens) / n
        out["frac_within_1pct_of_burn_in"] = float(n_near) / n
    return out


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
    # Tolerate the gene rate having been rounded on its way to the simulator
    # (MUTYH: variants sum to 2.90293e-08, simulator got 2.9029e-08), but refuse
    # a genuine inconsistency. Same tolerance as summarize_comp_het.py.
    if total_p > 1.0 + 1e-4:
        raise SystemExit(
            f"variant rates sum to {sum(mus):.8g} > gene rate {gene_mu:.8g} "
            f"(probabilities sum to {total_p:.8f}); pass --gene-mu sum if the "
            f"focal variants are meant to account for the whole gene rate")
    clipped = total_p > 1.0

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
    print(f"  seed = {args.seed}, conditioning = per-variant presence\n")

    if args.key is not None and len(args.infiles) != 1:
        raise SystemExit(f"--key needs exactly one input file, got {len(args.infiles)}")

    rng = np.random.default_rng(args.seed)

    summary_by_s = {}
    age_data = {}    # (variant, s) -> per-run oldest age
    freq_data = {}   # (variant, s) -> per-run summed variant frequency
    nlin_data = {}   # (variant, s) -> per-run number of lineages assigned

    for path in sorted(args.infiles):
        s_str = args.key if args.key is not None else s_from_filename(path)

        run_ages = {nm: [] for nm in names}
        run_freqs = {nm: [] for nm in names}
        run_nlin = {nm: [] for nm in names}
        n_runs_total = 0
        n_runs_any_lineage = 0
        n_lineages_total = 0

        for lineages in iter_runs(path):
            n_runs_total += 1
            if not lineages:
                # no surviving LoF lineage at all in this replicate: the mutator
                # is absent, so the run contributes to no variant's age
                continue
            n_runs_any_lineage += 1
            n_lineages_total += len(lineages)

            freqs = np.fromiter((f for f, _ in lineages), dtype=float, count=len(lineages))
            ages = np.fromiter((a for _, a in lineages), dtype=np.int64, count=len(lineages))

            u = rng.random(freqs.size)
            lo = 0.0
            for nm, hi in zip(names, edges):
                m = (u >= lo) & (u < hi)
                lo = hi
                k = int(np.count_nonzero(m))
                if k == 0:
                    # the variant is not segregating in this run: it has no age
                    continue
                run_ages[nm].append(int(ages[m].max()))
                run_freqs[nm].append(float(freqs[m].sum()))
                run_nlin[nm].append(k)

        entry = {
            "npz_key": s_str,
            **({} if args.key is not None else {"selection_coefficient": float(s_str)}),
            "gene_mu": gene_mu,
            "conditioning": "per-variant presence (>=1 lineage assigned to the variant)",
            "statistic": "age in generations of the oldest surviving lineage assigned to the variant",
            "seed": args.seed,
            "n_runs_total": int(n_runs_total),
            "n_runs_with_any_lineage": int(n_runs_any_lineage),
            "n_lineages_total": int(n_lineages_total),
            "mean_lineages_per_run": (float(n_lineages_total) / n_runs_total) if n_runs_total else None,
            "variant_probabilities": {nm: float(p) for nm, p in zip(names, probs)},
            "source_file": os.path.abspath(path),
        }
        if args.gene:
            entry["gene"] = args.gene
        if args.population:
            entry["population"] = args.population
        entry.update(meta)

        for nm in names:
            a = np.asarray(run_ages[nm], dtype=np.int64)
            entry[nm] = summarize_ages(a, args.burn_in)
            entry[nm]["n_runs_with_variant"] = int(a.size)
            entry[nm]["p_variant_present"] = (float(a.size) / n_runs_total) if n_runs_total else None
            age_data[(nm, s_str)] = a.astype(np.int32)
            freq_data[(nm, s_str)] = np.asarray(run_freqs[nm], dtype=np.float32)
            nlin_data[(nm, s_str)] = np.asarray(run_nlin[nm], dtype=np.int32)

        summary_by_s[s_str] = entry

        print(f"{os.path.basename(path)}: {n_runs_total} runs, "
              f"{n_runs_any_lineage} with a surviving lineage, "
              f"{entry['mean_lineages_per_run']:.3f} lineages/run")
        for nm in names:
            st = entry[nm]
            print(f"    {nm:>8s}: present in {st['n_runs_with_variant']} runs "
                  f"({100.0 * (st['p_variant_present'] or 0):.2f}%)  "
                  f"mean age = {st.get('mean')}  median = {st.get('median')}  "
                  f"max = {st.get('maximum')}"
                  + (f"  censored = {100.0 * st['frac_censored_at_burn_in']:.3f}%"
                     if "frac_censored_at_burn_in" in st else ""))

    # ---- write outputs ----------------------------------------------------
    payload = {}
    for (nm, s_str), arr in age_data.items():
        payload[f"{nm}_maxage_s={s_str}"] = arr
    for (nm, s_str), arr in freq_data.items():
        payload[f"{nm}_freq_s={s_str}"] = arr
    for (nm, s_str), arr in nlin_data.items():
        payload[f"{nm}_nlin_s={s_str}"] = arr
    if len(names) == 1:
        for (_, s_str), arr in age_data.items():
            payload[f"maxage_s={s_str}"] = arr
        for (_, s_str), arr in freq_data.items():
            payload[f"freq_s={s_str}"] = arr
        for (_, s_str), arr in nlin_data.items():
            payload[f"nlin_s={s_str}"] = arr

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **payload)
    print(f"\nSaved per-run allele ages to {args.out_npz}")
    print(f"  npz keys: {sorted(payload)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(summary_by_s, fh, indent=2)
    print(f"Wrote age summary statistics to {args.out_json}")


if __name__ == "__main__":
    main()
