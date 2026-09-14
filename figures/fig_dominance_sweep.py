#!/usr/bin/env python3
"""Figure 3. Relative log-likelihood of the mutator dominance coefficient h_m, scored against
the gnomAD v4.1.1 South Asian counts (k, n).

  Panel A (XPC)  three curves, at 0.1 s*, s* and 10 s*
  Panel B (MPG)  three curves, at 0.1 s*, s* and 10 s*

The counts are the same for the three curves of a panel; only the simulated
frequency distributions differ, one .npz per scaling of s.

  MPG  k = 35 / n = 91,070        XPC  k = 2 / n = 90,690

Thresholds, per curve and relative to its own maximum:
  interior MLE   -1.921   two-sided 95% CI
  boundary MLE   -1.353   one-sided 95% upper bound

CI_STYLE chooses how intervals are drawn: "bar" places colour-matched horizontal
bars in a strip below the curves, "span" shades them vertically.
SHOW_SEG_CORRECTION prints the log p_segregating(h) term for each curve.

Distributions include every simulated replicate; a replicate in which the
mutator was lost enters at a frequency of exactly 0.

Reads   results/sas/simulations_varying_h/  (workflow/varying_dominance.smk)
Writes  hm_profile_likelihood_s_scaling_panel_SAS.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (                       # noqa: E402
    RC as rc, GNOMAD_COLOR, BOX_FACECOLOR, BOX_ALPHA, CI_ALPHA,
    MUTYH_VARIANTS_ALL, clopper_pearson, freq_and_ci, binom_logpmf,
    summary_is_unconditional, FloorTracker, load_gnomad, load_demography,
    results_root, figures_root,
)

import os
import json
import numpy as np
from scipy.stats import chi2
from scipy.special import logsumexp, gammaln, xlogy, xlog1py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── config ──────────────────────────────────────────────────────────────────────
BASE = f"{results_root()}/sas/simulations_varying_h"
PDF_DIR = str(figures_root())
CI_STYLE = "bar"          # "bar" (recommended) or "span"
CURVE_YLIM = (-2.0, 0.05) # curve-panel y-range. Curves are plotted in full and simply
                          #   clipped at this edge, so where they plunge past -2 they run
                          #   off the bottom (they keep going to ~-600) rather than flatten.
YLIM = None               # span mode only: fixed (lo, hi) axis range, or None to auto-fit.
SHOW_SEG_CORRECTION = True   # print the size of the segregating-conditioning term

# scenario -> color and math label
COL    = {"0.1x": "#1B9E77", "1x": "#5B3A8C", "10x": "#D95F02"}
SLABEL = {"0.1x": r"$0.1\times\hat{s}^{*}$", "1x": r"$\hat{s}^{*}$",
          "10x": r"$10\times\hat{s}^{*}$"}

# ordered scenarios per gene (s* drawn/listed first).
# The SAS pipeline labels the three scalings smaller_s / center_s / larger_s.
SCEN = {
    "XPC": [
        ("1x",   "XPC/XPC_sas_varying_h_center_s.npz"),
        ("0.1x", "XPC/XPC_sas_varying_h_smaller_s.npz"),
        ("10x",  "XPC/XPC_sas_varying_h_larger_s.npz"),
    ],
    "MPG": [
        ("1x",   "MPG/MPG_sas_varying_h_center_s.npz"),
        ("0.1x", "MPG/MPG_sas_varying_h_smaller_s.npz"),
        ("10x",  "MPG/MPG_sas_varying_h_larger_s.npz"),
    ],
}

# observed South Asian counts (gnomAD v4.1.1, UKBB included)
COUNTS = {"XPC": (2, 90_690), "MPG": (35, 91_070)}
PANELS = [("XPC", "A"), ("MPG", "B")]

h_values = np.linspace(0, 1, 41)
H_STRS   = ["{:.3f}".format(h) for h in h_values]

THR_TWO = -chi2.ppf(0.95, 1) / 2.0   # -1.921  interior MLE -> two-sided 95% CI
THR_ONE = -chi2.ppf(0.90, 1) / 2.0   # -1.353  boundary MLE -> one-sided 95% bound

# threshold line styles keyed by threshold value: dashed for the two-sided cutoff
# (interior MLE), dotted for the one-sided cutoff (boundary MLE at h_m=0)
THR_STYLE = {
    THR_TWO: dict(ls=":",  label="CI threshold (two-sided)"),
    THR_ONE: dict(ls="--", label="CI threshold (one-sided)"),
}

TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 10, 10, 9, 8.5
LEG_SERIES_FS = 10   # larger font for the s* / 0.1 s* / 10 s* color-series legend
TITLE_X = -0.14   # x-offset for bold panel label (kept from original)
LABEL_Y = 1.16    # y-position for bold panel label above axes


# ── stats ─────────────────────────────────────────────────────────────────────
# ========================== CONDITIONING SWITCH ===========================
# True  -> each curve is conditional on the allele segregating (the published
#          behaviour, and what the NFE version of this figure does)
# False -> every simulated replicate, including those in which the allele was
#          lost, which enter with a frequency of exactly 0
#
# False needs a summary that kept every replicate. If the runs behind this figure
# were summarised before that was the case, load_h_npz() stops with a message
# naming the file and the Snakefile to re-run, rather than quietly returning the
# conditional sample under an unconditional label.
CONDITION_ON_SEGREGATING = False            # fixed: this is the unconditioned cell


def load_h_npz(path):
    if not CONDITION_ON_SEGREGATING and not summary_is_unconditional(path):
        raise SystemExit(
            f"{path}\n  kept only the replicates in which the allele was present, so "
            f"the unconditional\n  distribution cannot be recovered from it. Re-run "
            f"Snakefile_SAS_dem_varying_dominance_MPG_XPC.")
    raw = np.load(path, allow_pickle=False)
    return {(kk[2:] if kk.startswith("h=") else kk):
            np.asarray(raw[kk], dtype=float).ravel() for kk in raw.files}

def load_p_seg(path):
    """{h: p_segregating} from the sidecar .json, or {} if unavailable.

    A single-locus mutator records it at the top of its entry as 'p_segregating'.
    A compound-het mutator records it per variant, so it is looked up inside the
    per-variant block; older summaries of those runs put it at the top of the
    entry as 'p_run_kept'. All three mean the same thing here.
    """
    jpath = path[:-4] + ".json"
    if not os.path.exists(jpath):
        return {}
    with open(jpath) as fh:
        meta = json.load(fh)
    out = {}
    for hk, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        p = entry.get("p_segregating", entry.get("p_run_kept"))
        if p is None:
            # compound-het summary: the value lives in the per-variant block
            for v in entry.values():
                if isinstance(v, dict) and "p_segregating" in v:
                    p = v["p_segregating"]
                    break
        if p is not None:
            out[hk] = float(p)
    return out

def loglik_at_h(arr, k, n):
    """Composite log-likelihood at one h, averaged over the simulated replicates.

    Fixed replicates (q >= 1) and any non-finite value are always dropped. The
    zeros are dropped only when conditioning on the allele segregating; kept
    otherwise, where they contribute exactly zero to the average because
    Binom(k | n, 0) = 0 for k > 0, and enlarge the denominator.
    """
    keep = np.isfinite(arr) & (arr < 1)
    keep &= (arr > 0) if CONDITION_ON_SEGREGATING else (arr >= 0)
    arr = arr[keep]
    return (-np.inf if arr.size == 0
            else logsumexp(binom_logpmf(k, n, arr)) - np.log(arr.size))

def relative_loglik(data, k, n):
    return np.array([loglik_at_h(data.get(h, np.array([])), k, n) for h in H_STRS])

def ci_bounds(h, dll, imax, thr):
    lo = h[0]
    for i in range(imax, 0, -1):
        if (dll[i] - thr) * (dll[i - 1] - thr) < 0:
            t = (thr - dll[i]) / (dll[i - 1] - dll[i]); lo = h[i] + t * (h[i - 1] - h[i]); break
    hi = h[-1]
    for i in range(imax, len(h) - 1):
        if (dll[i] - thr) * (dll[i + 1] - thr) < 0:
            t = (thr - dll[i]) / (dll[i + 1] - dll[i]); hi = h[i] + t * (h[i + 1] - h[i]); break
    return lo, hi


# ── helper to add separated label + title ─────────────────────────────────────
def set_panel_label_and_title(ax, letter, title):
    # Bold panel label — left-shifted with padding (x kept from original TITLE_X)
    ax.text(TITLE_X, LABEL_Y, r"$\mathbf{%s.}$" % letter,
            transform=ax.transAxes, fontsize=TITLE_FS,
            va="bottom", ha="left", clip_on=False)
    # Centered title — standard set_title
    ax.set_title(title, fontsize=TITLE_FS, fontweight="normal")


# ── compute ─────────────────────────────────────────────────────────────────────
ihalf = int(np.argmin(np.abs(h_values - 0.5)))
results = {}   # (gene, scen) -> dict
for gene, _ in PANELS:
    k, n = COUNTS[gene]
    for scen, fname in SCEN[gene]:
        path = os.path.join(BASE, fname)
        data = load_h_npz(path)
        missing = [h for h in H_STRS if h not in data]
        if missing:
            raise SystemExit(f"{fname}: missing h keys {missing[:5]}"
                             f"{' ...' if len(missing) > 5 else ''}")
        ll   = relative_loglik(data, k, n)
        imax = int(np.argmax(ll)); llmax = ll[imax]; dll = ll - llmax
        thr  = THR_ONE if imax == 0 else THR_TWO
        lo, hi = ci_bounds(h_values, dll, imax, thr)
        LR0 = 2 * (llmax - ll[0]);     p0 = 0.5 * chi2.sf(LR0, 1)
        LRh = 2 * (llmax - ll[ihalf]); ph = chi2.sf(LRh, 1)
        results[(gene, scen)] = dict(k=k, n=n, dll=dll, imax=imax, hmle=h_values[imax],
                                     llmax=llmax, thr=thr, ci=(lo, hi),
                                     LR0=LR0, p0=p0, LRh=LRh, ph=ph,
                                     p_seg=load_p_seg(path),
                                     n_sim=np.array([data[h].size for h in H_STRS]))

# ── printout ──────────────────────────────────────────────────────────────────
for gene, letter in PANELS:
    k, n = COUNTS[gene]
    print(f"\n########## {letter}. {gene}  [gnomAD v4.1.1, South Asian]  k={k}, n={n} ##########")
    for scen, _ in SCEN[gene]:
        r = results[(gene, scen)]
        ci_type = "one-sided 95% upper bound" if r['imax'] == 0 else "two-sided 95% CI"
        print(f"\n  --- {scen}  ({SLABEL[scen]}) ---")
        print(f"    MLE h_m            : {r['hmle']:.3f}"
              + ("   (boundary h_m=0)" if r['imax'] == 0 else ""))
        print(f"    max log10 L        : {r['llmax']/np.log(10):.3f}")
        print(f"    {ci_type} (h_m): [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  (thr={r['thr']:.3f})")
        print(f"    recessive (h_m=0)  : LR={r['LR0']:.3f}, p={r['p0']:.3g}  "
              f"({'not rejected' if r['p0'] > 0.05 else 'REJECTED'}; 0.5chi2_0+0.5chi2_1)")
        print(f"    semi-dom (h_m=0.5) : LR={r['LRh']:.3f}, p={r['ph']:.3g}  "
              f"({'not rejected' if r['ph'] > 0.05 else 'REJECTED'}; chi2_1)")
        if SHOW_SEG_CORRECTION and r["p_seg"]:
            lp = np.array([np.log(r["p_seg"][h]) for h in H_STRS])
            print(f"    seg. conditioning  : simulated runs {r['n_sim'].min()}-{r['n_sim'].max()}; "
                  f"undoing it would shift the curve by at most "
                  f"{np.ptp(lp):.2e} log units (not applied)")

# ── figure ────────────────────────────────────────────────────────────────────
# CI strip rows ordered by descending selection coefficient (10s*, s*, 0.1s*),
# top to bottom; a fixed row per scenario keeps each bar at the same height in both panels.
ROW_ORDER = ["10x", "1x", "0.1x"]
n_rows    = len(ROW_ORDER)

if CI_STYLE == "bar":
    fig = plt.figure(figsize=(6.5, 3.9))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0])
    for i, (gene, letter) in enumerate(PANELS):
        axc = fig.add_subplot(gs[0, i])                 # curve panel
        axb = fig.add_subplot(gs[1, i], sharex=axc)     # CI strip below it
        scens = SCEN[gene]

        # threshold line(s)
        for thr in sorted(set(results[(gene, s)]["thr"] for s, _ in scens)):
            axc.axhline(thr, ls=THR_STYLE[thr]["ls"], lw=1.0, color="0.45", zorder=2)
        # curves plotted in full; ylim simply clips them at the bottom edge
        for scen, _ in scens:
            axc.plot(h_values, results[(gene, scen)]["dll"], "-", lw=1.6,
                     color=COL[scen], zorder=4)
        axc.set_ylim(*CURVE_YLIM)
        axc.set_xlim(-0.02, 1.02)

        # separated bold label + centered title
        set_panel_label_and_title(axc, letter, gene)

        axc.tick_params(labelsize=TICK_FS, labelbottom=False)
        axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
        if i == 0:
            axc.set_ylabel(r"$\ell - \ell_{\max}$", fontsize=LABEL_FS)

        # CI bars in the strip, one fixed row per scenario (10s* on top, 0.1s* on bottom)
        for scen, _ in scens:
            r = results[(gene, scen)]
            ybar = n_rows - 0.5 - ROW_ORDER.index(scen)
            lo, hi = r["ci"]
            axb.plot([lo, hi], [ybar, ybar], "-", lw=3.0, color=COL[scen],
                     solid_capstyle="butt", alpha=0.85, zorder=3)
            axb.plot([r["hmle"]], [ybar], marker="D", ms=6, color=COL[scen],
                     markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        axb.set_ylim(0, n_rows)
        axb.set_xlim(-0.02, 1.02)
        axb.set_yticks([])
        axb.set_xlabel(r"Dominance coefficient ($h$)", fontsize=LABEL_FS)
        axb.tick_params(labelsize=TICK_FS)
        for sp in ("top", "right", "left"):
            axb.spines[sp].set_visible(False)
        if i == 0:
            axb.set_ylabel("95% CI", fontsize=LABEL_FS - 1)
            axb.yaxis.set_label_coords(-0.045, 0.5)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.24,
                        wspace=0.22, hspace=0.30)

else:  # "span"
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True, sharey=True)
    for ax, (gene, letter) in zip(axes, PANELS):
        scens   = SCEN[gene]
        dll_min = min(results[(gene, s)]["dll"].min() for s, _ in scens)
        thr_min = min(results[(gene, s)]["thr"]       for s, _ in scens)
        for thr in sorted(set(results[(gene, s)]["thr"] for s, _ in scens)):
            ax.axhline(thr, ls=THR_STYLE[thr]["ls"], lw=1.0, color="0.45", zorder=2)
        for scen, _ in scens:
            r = results[(gene, scen)]
            ax.plot(h_values, r["dll"], "-", lw=1.6, color=COL[scen], zorder=4)
            ax.axvspan(*r["ci"], color=COL[scen], alpha=0.10, lw=0, zorder=1)
            ax.axvline(r["hmle"], color=COL[scen], lw=1.2, alpha=0.9, zorder=4)
        ax.set_ylim(YLIM if YLIM is not None
                    else (min(dll_min, thr_min) * 1.10, CURVE_YLIM[1]))
        ax.set_xlim(-0.02, 1.02)

        # separated bold label + centered title
        set_panel_label_and_title(ax, letter, gene)

        ax.set_xlabel(r"Dominance coefficient ($h$)", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\ell - \ell_{\max}$", fontsize=LABEL_FS)
    fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=2.0)

# ── legend ──────────────────────────────────────────────────────────────────────
order        = ["10x", "1x", "0.1x"]   # legend color order matches the CI strip
line_handles = [Line2D([0], [0], color=COL[s], lw=1.6, label=SLABEL[s]) for s in order]

# threshold handles only for thresholds actually present in the figure
present_thr = {results[(g, s)]["thr"] for g, _ in PANELS for s, _ in SCEN[g]}
thr_handles = [Line2D([0], [0], color="0.45", ls=THR_STYLE[t]["ls"], lw=1.0,
                      label=THR_STYLE[t]["label"])
               for t in (THR_ONE, THR_TWO) if t in present_thr]

if CI_STYLE == "bar":
    ci_handles = [
        Line2D([0], [0], color="0.35", lw=3.0, label="95% CI"),
        Line2D([0], [0], marker="D", ms=4, color="0.35", lw=0, label=r"MLE of $h$"),
    ]
else:
    ci_handles = [
        Patch(facecolor="0.35", alpha=0.25, label="95% CI"),
        Line2D([0], [0], color="0.35", lw=1.2, label=r"MLE of $h$"),
    ]

fig.legend(handles=line_handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=LEG_SERIES_FS)
fig.legend(handles=ci_handles + thr_handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.13), fontsize=LEG_FS)

os.makedirs(PDF_DIR, exist_ok=True)
fig.savefig(f"{PDF_DIR}/hm_profile_likelihood_s_scaling_panel_SAS.pdf", bbox_inches="tight")
print(f"\nSaved -> {PDF_DIR}/hm_profile_likelihood_s_scaling_panel_SAS.pdf")
plt.show()