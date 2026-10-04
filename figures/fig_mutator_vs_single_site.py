#!/usr/bin/env python3
"""Figure S3. Mean (A) and variance (B) of the mutator frequency against phi*G:
mutator simulations (2 N s_het = 20) and single-site simulations at N = 2000,
+/- 2 standard errors.

Reads   results/mutator_simulation/2Ns_het_20/summary.json  (workflow/mutator_simulation.smk)
        results/single_site_constant_N/single_site_recessive_N_2000.npz
                                                    (workflow/single_site_constant_N.smk)
Writes  mutator_vs_single_site_sim_comparison.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figures_root, load_mutator_summary, results_root   # noqa: E402

import numpy as np                                                     # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402

# -- mutator simulations -----------------------------------------------------
rows = load_mutator_summary(20)
N, F, S_HET = rows[0]["N"], rows[0]["f"], rows[0]["s_het"]
mut_x = np.array([r["phi_G"] for r in rows])
mut_mean = np.array([r["mean_q"] for r in rows])
mut_se_mean = np.array([r["se_mean_q"] for r in rows])
mut_var = np.array([r["var_q"] for r in rows])
mut_se_var = np.array([r["se_var_q"] for r in rows])

# -- single-site simulations ---------------------------------------------------
SLOPE = 2 * F * S_HET                     # s* per unit of phi*G
ss = np.load(results_root() / "single_site_constant_N" / f"single_site_recessive_N_{N}.npz")
ss_x, ss_mean, ss_se_mean, ss_var, ss_se_var = [], [], [], [], []
for key in sorted(ss.files, key=float):
    q = ss[key].astype(np.float64)
    n, m = q.size, q.mean()
    ss_x.append(float(key) / SLOPE)
    ss_mean.append(m)
    ss_se_mean.append(np.std(q, ddof=1) / np.sqrt(n))
    ss_var.append(np.var(q, ddof=1))
    ss_se_var.append(np.std((q - m) ** 2, ddof=1) / np.sqrt(n))
ss_x, ss_mean, ss_se_mean, ss_var, ss_se_var = map(
    np.array, (ss_x, ss_mean, ss_se_mean, ss_var, ss_se_var))

# -- 2 N s* = 1 and 10 ---------------------------------------------------------
phiG_2Ns1 = 1.0 / (2 * N * SLOPE)
phiG_2Ns10 = 10.0 / (2 * N * SLOPE)
XLIM = (1e-2, 1e3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.55))

for ax in (ax1, ax2):
    ax.axvspan(XLIM[0], phiG_2Ns1, color="gray", zorder=0, alpha=0.01)
    ax.axvspan(phiG_2Ns1, phiG_2Ns10, color="gray", zorder=0, alpha=0.10)
    ax.axvspan(phiG_2Ns10, XLIM[1], color="gray", zorder=0, alpha=0.20)
    ax.axvline(phiG_2Ns1, color="black", ls="--", lw=0.9, zorder=3)
    ax.axvline(phiG_2Ns10, color="black", ls="--", lw=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLIM)
    ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", labelsize=6)

POINTS = dict(markersize=4, alpha=0.8, capsize=1.5, capthick=0.6, elinewidth=0.6, zorder=5)
for ax, mut_y, mut_e, ss_y, ss_e in ((ax1, mut_mean, mut_se_mean, ss_mean, ss_se_mean),
                                     (ax2, mut_var, mut_se_var, ss_var, ss_se_var)):
    ax.errorbar(mut_x, mut_y, yerr=2 * mut_e, fmt="o", color="purple",
                label="Mutator simulation", **POINTS)
    ax.errorbar(ss_x, ss_y, yerr=2 * ss_e, fmt="s", color="steelblue",
                label="Single-site simulation", **POINTS)

ax1.set_ylim(1e-6, 1)
ax1.set_ylabel("Frequency", fontsize=10)
ax1.text(phiG_2Ns1 * 0.74, 1.3e-6, r"$2N_e s^*=1$", color="black", fontsize=8.5,
         ha="center", va="bottom", rotation=90, clip_on=False)
ax1.text(phiG_2Ns10 * 0.74, 1.3e-6, r"$2N_e s^*=10$", color="black", fontsize=8.5,
         ha="center", va="bottom", rotation=90, clip_on=False)
ax2.set_ylim(1e-7, 0.5)
ax2.set_ylabel("Variance", fontsize=10)

for ax, label, title in zip((ax1, ax2), ("A.", "B."),
                            ("Expected mutator frequency", "Variance in mutator frequency")):
    ax.text(-0.2, 1.15, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")
    ax.text(-0.11, 1.15, title, transform=ax.transAxes,
            fontsize=11, fontweight="normal", va="top", ha="left")

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
           frameon=True, framealpha=0.9, edgecolor="gray", borderpad=0.5)

out = figures_root() / "mutator_vs_single_site_sim_comparison.pdf"
plt.savefig(out, format="pdf", bbox_inches="tight")
print(f"wrote {out}")
