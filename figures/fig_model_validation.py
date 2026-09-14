#!/usr/bin/env python3
"""Figure S1. Selection on a mutator and its linked DNMs, across mutator dominance
coefficients h. Plotting only; the simulations are run by
workflow/model_validation.smk.

2 x 4 figure sized for a 6.5 inch text width, one column per value of h.

Top row, from trajectories/trajectories_h_{h}.npz: 50 replicates started at
q0 = 0.5 and followed for 200 generations.
    lambda_tau / 2 lambda   excess DNMs linked to a mutator haplotype,
                            normalised by its asymptote at the current q
    q_tau / q               mutator frequency, normalised by its starting value

Bottom row, from s_vs_q/s_vs_q_h_{h}.npz: 10,000 replicates, each started from
its own q ~ Uniform(0.05, 0.95) and run for 50 generations. Each contributes one
point at the frequency it reached, q_tau, plotted as the ratio of the observed
selection coefficient

    s_obs = s_het * (excess deleterious DNMs linked to a mutator haplotype)

to the value predicted at that frequency,

    s_exp = 2 f lambda s_het = 2 f phi_G (h + q_tau(1 - 2h)) s_het

Transparent boxes summarise each equal-width frequency bin: median, IQR, and
whiskers at the 2.5th and 97.5th percentiles. All four panels share a y axis,
and a bin whose whiskers span more than MAX_BIN_SPAN is drawn without a box; the
script names any such bin on stdout, and its replicates remain in the scatter.

Notation differs from the other figures:
    h        (was h_m)      mutator dominance
    s        (was s_m)      selection coefficient on the mutator
    s_het    (was h_s_s_s)  fitness cost of one excess deleterious DNM
    2*lambda (was Lambda)   asymptotic excess DNMs linked to a mutator
                            haplotype, counting all DNMs rather than only the
                            deleterious fraction f

Reads   results/model_validation/   (workflow/model_validation.smk)
Writes  mutator_s_vs_h_grid.pdf
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# -- where the simulations landed -------------------------------------------
SIM_DIR = f"{results_root()}/model_validation"
PDF_DIR = str(figures_root())
H_VALUES = ["0.00", "0.25", "0.75", "1.00"]

# -- figure styling: scoped to this cell only (does NOT touch global rcParams) --
rc = {
    'font.size':         9,
    'axes.titlesize':    9.5,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'axes.titleweight':  'normal',
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
}

# -- colors ------------------------------------------------------------------
COL_SIM    = '#1f77b4'   # bottom row: individual replicates
COL_BOX    = '#0b3d62'   # bottom row: binned boxes
COL_THEORY = 'black'     # bottom row: the s_obs = s_exp reference
COL_LAMBDA = '#2ca02c'   # top row: linked DNMs, lambda_tau / 2 lambda
COL_Q      = '#d62728'   # top row: mutator frequency, q_tau / q

EQUILIBRATED_BY = 7      # generations for lambda_tau to reach its asymptote
N_BINS          = 8      # q bins for the boxes in the bottom row
MIN_PER_BIN     = 20
WHIS            = (2.5, 97.5)   # whisker percentiles: a box spans the central 95%
MAX_BIN_SPAN    = 1.0    # a bin wider than this does not get to set the y scale
CAP_INSET       = 0.055  # how far inside the frame a cut-off whisker's cap sits
LEGEND_HEADROOM = 0.26   # extra y room for the two legends, as a fraction of the range


def whisker_ends(v):
    """The percentiles the whiskers reach; used for the axis limits and the drop rule,
    so they cannot drift apart from what boxplot() actually draws."""
    lo, hi = np.percentile(v, WHIS)
    return lo, hi


def bin_by_q(x, y, n_bins=N_BINS, lo=0.0, hi=1.0):
    """Split y into n_bins equal-width bins of x; bins below MIN_PER_BIN are dropped."""
    edges = np.linspace(lo, hi, n_bins + 1)
    idx   = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    groups, centers = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= MIN_PER_BIN:
            groups.append(y[m])
            centers.append(0.5 * (edges[b] + edges[b + 1]))
    return groups, np.array(centers), (edges[1] - edges[0])


def padded(lo, hi, frac=0.07):
    """Axis limits that leave a little air around the extreme whiskers."""
    m = frac * (hi - lo)
    return lo - m, hi + m


# -- load --------------------------------------------------------------------
traj = {h: np.load(f"{SIM_DIR}/trajectories/trajectories_h_{h}.npz") for h in H_VALUES}
svq  = {h: np.load(f"{SIM_DIR}/s_vs_q/s_vs_q_h_{h}.npz")             for h in H_VALUES}

# -- bottom-row bins, and the y limits they imply ----------------------------
ratio_bins = {}
kept_ends  = []
for h in H_VALUES:
    d  = svq[h]
    ok = ~np.isnan(d['s_obs'])
    qc, ratio = d['q_current'][ok], d['s_obs'][ok] / d['s_exp'][ok]
    groups, centers, width = bin_by_q(qc, ratio)
    binned = [(g, c) + whisker_ends(g) for g, c in zip(groups, centers)]
    # every bin is drawn; only the well-behaved ones are allowed to set the scale
    for g, c, lo, hi in binned:
        if hi - lo > MAX_BIN_SPAN:
            print(f'h = {float(d["h"]):g}: whiskers at q = {c:.3f} reach '
                  f'[{lo:.2f}, {hi:.2f}] and are cut at the frame')
    ratio_bins[h] = (qc, ratio, binned, width)
    kept_ends.extend((lo, hi) for _, _, lo, hi in binned if hi - lo <= MAX_BIN_SPAN)

# one axis for the whole row, with room left top and bottom for the two legends
YLIM = padded(min(e[0] for e in kept_ends), max(e[1] for e in kept_ends),
              frac=LEGEND_HEADROOM)

# -- plot --------------------------------------------------------------------
with mpl.rc_context(rc):
    fig = plt.figure(figsize=(6.5, 4.4), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.02)

    ax_top = subfigs[0].subplots(1, 4, sharex=True, sharey=True)
    ax_bot = subfigs[1].subplots(1, 4, sharex=True, sharey=True)
    for ax in (*ax_top, *ax_bot):
        ax.set_box_aspect(1)

    subfigs[0].suptitle(r'$\bf{A.}$ Trajectories of mutator frequency and linked DNMs',
                        x=0.012, ha='left', fontsize=9.5)
    subfigs[1].suptitle(r'$\bf{B.}$ $s$ as a function of current mutator frequency',
                        x=0.012, ha='left', fontsize=9.5)

    for j, h in enumerate(H_VALUES):
        # ---- top row: trajectories ----
        d   = traj[h]
        tau = np.arange(int(d['n_gens']) + 1)
        lam_norm = d['lam'] / d['two_lambda']     # lambda_tau / 2 lambda at the current q
        q_norm   = d['q'] / float(d['q0'])        # q_tau / q

        ax = ax_top[j]
        for i in range(lam_norm.shape[0]):
            ax.plot(tau, lam_norm[i], color=COL_LAMBDA, alpha=0.15, lw=0.6)
            ax.plot(tau, q_norm[i],   color=COL_Q,      alpha=0.15, lw=0.6)
        ax.plot(tau, np.nanmean(lam_norm, 0), color=COL_LAMBDA, lw=1.4)
        ax.plot(tau, np.nanmean(q_norm, 0),   color=COL_Q,      lw=1.4)
        ax.axvline(EQUILIBRATED_BY, color='gray', lw=0.7, ls='--')
        ax.set_xlim(0, int(d['n_gens']))
        ax.set_ylim(0, 1.15)
        ax.set_xticks([0, 100, 200])
        ax.set_title(rf'$h = {float(d["h"]):g}$')
        ax.set_xlabel(r'Generation, $\tau$')
        if j == 0:
            ax.set_ylabel('Normalized value')
            # set along the dashed line rather than across the panel, so it cannot
            # run into the legend at any font size
            ax.text(EQUILIBRATED_BY + 6, 0.05, r'$\lambda$ equilibrated', color='gray',
                    fontsize=8, va='bottom', ha='left', rotation=90)
            ax.legend(handles=[
                Line2D([0], [0], color=COL_LAMBDA, lw=1.4,
                       label=r'$\lambda_\tau\,/\,2\lambda$'),
                Line2D([0], [0], color=COL_Q, lw=1.4,
                       label=r'$q_\tau\,/\,q_0$'),
            ], frameon=False, loc='lower right', labelspacing=0.25, borderpad=0.15,
               handlelength=1.2, handletextpad=0.4)

        # ---- bottom row: observed / expected s, against the frequency reached ----
        d = svq[h]
        qc, ratio, binned, width = ratio_bins[h]

        ax = ax_bot[j]
        ax.axhline(1.0, color=COL_THEORY, lw=1.0, zorder=4)
        ax.scatter(qc, ratio, color=COL_SIM, s=1.2, alpha=0.05, edgecolor='none',
                   zorder=2)
        bp = ax.boxplot([g for g, _, _, _ in binned],
                        positions=[c for _, c, _, _ in binned], widths=0.72 * width,
                        whis=WHIS, patch_artist=True, showfliers=False,
                        manage_ticks=False, zorder=5,
                        medianprops=dict(color=COL_BOX, lw=0.9),
                        whiskerprops=dict(color=COL_BOX, lw=0.6),
                        capprops=dict(color=COL_BOX, lw=0.6))
        for patch in bp['boxes']:
            patch.set(facecolor=COL_BOX, alpha=0.28, edgecolor=COL_BOX, linewidth=0.6)

        # A whisker running off the frame is trimmed to just inside it and capped with a
        # triangle pointing the way it continues, so the bin stays visible without being
        # allowed to dictate the scale. CAP_INSET keeps the whole marker clear of the
        # spine. boxplot() returns whiskers as [box0 lower, box0 upper, box1 lower, ...].
        span = YLIM[1] - YLIM[0]
        cap_hi = YLIM[0] + (1.0 - CAP_INSET) * span
        cap_lo = YLIM[0] + CAP_INSET * span
        for i, (_, c, lo, hi) in enumerate(binned):
            if hi > YLIM[1]:
                w = bp['whiskers'][2 * i + 1]
                w.set_ydata([w.get_ydata()[0], cap_hi])
                ax.plot([c], [cap_hi], marker='^', color=COL_BOX, ms=3.2, zorder=6)
            if lo < YLIM[0]:
                w = bp['whiskers'][2 * i]
                w.set_ydata([w.get_ydata()[0], cap_lo])
                ax.plot([c], [cap_lo], marker='v', color=COL_BOX, ms=3.2, zorder=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(*YLIM)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(rf'$h = {float(d["h"]):g}$')
        ax.set_xlabel(r'Mutator frequency, $q_\tau$')
        if j == 0:
            ax.set_ylabel(r'$s_{\mathrm{obs}}\,/\,s_{\mathrm{exp}}$')
            # split across the two bands YLIM leaves free, above and below the cloud,
            # so neither key sits on top of a point or a whisker
            top = ax.legend(handles=[
                Line2D([0], [0], color=COL_SIM, marker='o', lw=0, ms=3,
                       label='simulations')],
                frameon=False, loc='upper left', borderpad=0.1,
                handlelength=1.2, handletextpad=0.4)
            ax.add_artist(top)
            ax.legend(handles=[
                Line2D([0], [0], color=COL_BOX, lw=4, alpha=0.45,
                       label='binned freqs.')],
                frameon=False, loc='lower left', borderpad=0.1,
                handlelength=1.2, handletextpad=0.4)

    # no bbox_inches='tight' here: constrained_layout has already fitted everything
    # inside the canvas, and re-cropping would make the PDF something other than the
    # 6.5 inches of LaTeX text width it is sized for.
    plt.savefig(f'{PDF_DIR}/mutator_s_vs_h_grid.pdf')
    plt.show()