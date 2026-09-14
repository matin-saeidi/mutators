#!/usr/bin/env python3
"""Figure S4. The three demographic histories used throughout: non-Finnish European (NFE),
South Asian (SAS) and African / African-American (AFR).

Each history is a composite. The ancient portion is the MSMC inference of
Schiffels & Durbin (2014); the recent portion is from Schraiber et al. (2025)
for NFE and Kar et al. (2026) for SAS and AFR.

Colour encodes the population and linestyle the study:

  dashed   Schiffels & Durbin (2014)   ancient portion of all three
  solid    Schraiber et al. (2025)     recent portion, NFE
  dotted   Kar et al. (2026)           recent portion, SAS and AFR

Ne and T are read from data/demographic_models/, which is dumped from the
simulators. The tables are in generations before the present: epoch i has size
Ne[i] and covers T[i] < gen <= T[i-1], and Ne[0] applies to every generation
older than T[0]. The simulations themselves run forward in time.

SWITCH_IDX gives the first epoch of the recent portion of each history, so the
handover generation is T[SWITCH_IDX - 1]:

  NFE  517        SAS  1054        AFR  1200

Reads   data/demographic_models/*.tsv
Writes  demographic_histories_NFE_SAS_AFR.pdf
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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Demographic models, verbatim from the simulators.
# ---------------------------------------------------------------------------
# Demographic models, read from the files dumped by the simulators.
NE_EUR, T_EUR = load_demography("eur")
NE_SAS, T_SAS = load_demography("sas")
NE_AFR, T_AFR = load_demography("afr")

# first epoch of the recent (Schraiber / Kar) portion -- see the note above
SWITCH_IDX = {"NFE": 52, "SAS": 51, "AFR": 47}

# Okabe-Ito, matching the demography panels earlier in this notebook
COLORS = {"NFE": "#CC79A7", "SAS": "#E69F00", "AFR": "#009E73"}
RECENT_SOURCE = {"NFE": "Schraiber et al. (2025)",
                 "SAS": "Kar et al. (2026)",
                 "AFR": "Kar et al. (2026)"}

# linestyle per inference. Keyed off the source label rather than the population
# so the curves and the legend below cannot drift apart.
ANCIENT_SOURCE = "Schiffels & Durbin (2014)"
SOURCE_LS = {ANCIENT_SOURCE: "--",
             "Schraiber et al. (2025)": "-",
             "Kar et al. (2026)": ":"}

POPS = [
    ("NFE", "Non-Finnish European", NE_EUR, T_EUR),
    ("SAS", "South Asian",          NE_SAS, T_SAS),
    ("AFR", "African / African-American", NE_AFR, T_AFR),
]

T_MIN_PLOT = 1.0        # 1 generation ago (a log axis cannot show 0)
T_MAX_PLOT = 2.5e5      # the burn-in horizon used in the simulations


def demography_staircase(Ne, T, t_min=T_MIN_PLOT, t_max=T_MAX_PLOT):
    """Return (x, y) tracing Ne against generations before present.

    Segment k runs from boundaries[k] to boundaries[k+1] at height sizes[k],
    and corresponds to original epoch index len(Ne)-1-k (present -> past).
    """
    assert len(Ne) == len(T) and T[-1] == 0
    boundaries = [t_min] + list(reversed(T[:-1])) + [t_max]   # present -> past
    sizes      = list(reversed(Ne[1:])) + [Ne[0]]
    assert len(boundaries) == len(sizes) + 1
    return np.repeat(boundaries, 2)[1:-1], np.repeat(sizes, 2)


def split_at_epoch(x, y, n_epochs, switch_idx):
    """Cut the staircase into (recent, ancient) polylines.

    Epoch `switch_idx` and everything younger is 'recent'. In staircase order
    that is segments 0..K with K = n_epochs-1-switch_idx, i.e. points 0..2K+1;
    the slice is extended by one point so the two polylines share the vertical
    riser at the handover and join with no visible gap.
    """
    K = n_epochs - 1 - switch_idx
    cut = 2 * K + 2
    return (x[:cut + 1], y[:cut + 1]), (x[cut:], y[cut:])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
LW = 2.0
fig, ax = plt.subplots(figsize=(6.5, 4.2))

for key, label, Ne, T in POPS:
    x, y = demography_staircase(Ne, T)
    (xr, yr), (xa, ya) = split_at_epoch(x, y, len(Ne), SWITCH_IDX[key])
    ax.plot(xa, ya, color=COLORS[key], linewidth=LW, zorder=3,
            linestyle=SOURCE_LS[ANCIENT_SOURCE])
    ax.plot(xr, yr, color=COLORS[key], linewidth=LW, zorder=3,
            linestyle=SOURCE_LS[RECENT_SOURCE[key]])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(T_MIN_PLOT, T_MAX_PLOT)
ax.set_ylim(2e3, 1e8)
ax.set_xlabel("Time (generations ago)", fontsize=10.5)
ax.set_ylabel("Effective population size $N_e$", fontsize=10.5)
ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=1.2)
ax.tick_params(axis="both", which="minor", length=3, width=1.0)
ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
# population legend (colour), text coloured to match, as in the panels above
# colour swatches rather than line samples: linestyle is spoken for by the
# inference-source key below, so the population key must not imply one
pop_handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=lab)
               for k, lab, _, _ in POPS]
leg1 = ax.legend(handles=pop_handles, loc="upper right", frameon=False,
                 fontsize=9, handlelength=1.4, handleheight=1.1)
for txt, (k, _, _, _) in zip(leg1.get_texts(), POPS):
    txt.set_color(COLORS[k])
ax.add_artist(leg1)

# inference-source legend (linestyle), in neutral grey
# ordered oldest-inference-first: Schiffels & Durbin, then Schraiber, then Kar
src_handles = [
    plt.Line2D([0], [0], color="0.35", linewidth=LW,
               linestyle=SOURCE_LS[ANCIENT_SOURCE], label=ANCIENT_SOURCE),
    plt.Line2D([0], [0], color="0.35", linewidth=LW,
               linestyle=SOURCE_LS["Schraiber et al. (2025)"],
               label="Schraiber et al. (2025)"),
    plt.Line2D([0], [0], color="0.35", linewidth=LW,
               linestyle=SOURCE_LS["Kar et al. (2026)"],
               label="Kar et al. (2026)"),
]
# lower left is the one empty corner: every history is at its maximum in the
# recent past, so nothing is drawn down there
ax.legend(handles=src_handles, loc="lower left", frameon=False,
          fontsize=8.5, handlelength=2.4)

plt.tight_layout(pad=0.6)

PDF_DIR = str(figures_root())
os.makedirs(PDF_DIR, exist_ok=True)
fig.savefig(f"{PDF_DIR}/demographic_histories_NFE_SAS_AFR.pdf",
            format="pdf", bbox_inches="tight")

for key, label, Ne, T in POPS:
    i = SWITCH_IDX[key]
    print(f"{label:<28} {len(Ne):>3d} epochs | "
          f"{RECENT_SOURCE[key]:<24} from gen {T[i-1]:>5d} to present "
          f"({len(Ne)-i} epochs) | present-day Ne = {Ne[-1]:,}")
print(f"\nSaved -> {PDF_DIR}/demographic_histories_NFE_SAS_AFR.pdf")
plt.show()