"""
Estimate mutator effect sizes from trio de novo mutation (DNM) counts
(Supplementary Section S4.1, Eqs. S22-S25; reproduces Table S3).

Input : data/trio_dnms.csv
Output: printed per-offspring and per-mutator estimates

Usage : python src/effect_sizes.py
"""
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
#   Age and sex effects on DNMs transmitted per parent (Jonsson et al. 2017), Eq. S23:
#     E(Y | A_F) = 6.05 + 1.51 A_F   (fathers)
#     E(Y | A_M) = 3.61 + 0.37 A_M   (mothers)
#   ACCESSIBLE_GENOME_JONSSON: genome size over which those rates were estimated;
#     used to rescale E(Y) for trios that report their own accessible genome size.
#   E_D: expected number of DNMs per offspring, averaged over parental ages (Eq. S25).
#   G, F, S_HET: Table 1 values used in Eq. 5, s* = 2 G f phi s_het.
# ---------------------------------------------------------------------------
ALPHA_F, BETA_F = 6.05, 1.51
ALPHA_M, BETA_M = 3.61, 0.37
ACCESSIBLE_GENOME_JONSSON = 2_682_890_000
E_D = 70.3
G, F, S_HET = 3e9, 0.08, 5e-4

DATA = Path(__file__).resolve().parents[1] / "data" / "trio_dnms.csv"


def per_offspring(df: pd.DataFrame) -> pd.DataFrame:
    """Eqs. S22-S24: imputed DNMs from the carrier parent and multiplicative effect size."""
    df = df.copy()
    is_father = df["carrier_parent"] == "father"

    # Eq. S22: scale the child's total DNMs by the fraction phased to the carrier parent
    phased_carrier = df["dnm_phased_paternal"].where(is_father, df["dnm_phased_maternal"])
    phased_total = df["dnm_phased_paternal"] + df["dnm_phased_maternal"]
    df["Y_tilde"] = df["dnm_total"] * phased_carrier / phased_total

    # Eq. S23: expected DNMs from a non-carrier parent of the same sex and age.
    # Age ranges (GEL trios) are replaced by their midpoint.
    age_F = (df["paternal_age_min"] + df["paternal_age_max"]) / 2
    age_M = (df["maternal_age_min"] + df["maternal_age_max"]) / 2
    df["E_Y"] = (ALPHA_F + BETA_F * age_F).where(is_father, ALPHA_M + BETA_M * age_M)

    # Trios that report an accessible genome size: rescale E(Y) to that genome size
    scale = (df["accessible_genome_bp"] / ACCESSIBLE_GENOME_JONSSON).fillna(1.0)
    df["E_Y_scaled"] = df["E_Y"] * scale

    # Eq. S24: multiplicative effect size of the mutator in this parent
    df["Phi_hat"] = df["Y_tilde"] / df["E_Y_scaled"]
    return df


def per_mutator(df: pd.DataFrame) -> pd.DataFrame:
    """Average offspring within each carrier parent, then across parents; Eq. S25 and Eq. 5."""
    used = df[df["include_in_estimate"]]
    by_parent = used.groupby(["gene", "carrier_parent_id"])["Phi_hat"].mean()
    out = by_parent.groupby("gene").agg(Phi_bar="mean", n_parents="size").reset_index()

    # Eq. S25: phi*G is the extra DNMs genome-wide in the carrier genotype. For heterozygous
    # carriers (POLE, POLD1) this is the heterozygote effect, h*phi*G.
    out["phi_G"] = 0.5 * (out["Phi_bar"] - 1) * E_D
    out["phi"] = out["phi_G"] / G
    # Eq. 5 applied to the carrier-genotype effect (= h*s* for heterozygous carriers)
    out["s_eq5"] = 2 * F * S_HET * out["phi_G"]
    return out


if __name__ == "__main__":
    trios = per_offspring(pd.read_csv(DATA))
    cols = ["offspring_id", "gene", "carrier_parent", "dnm_total", "Y_tilde", "E_Y", "E_Y_scaled", "Phi_hat"]
    print(trios[cols].round(3).to_string(index=False))
    print()
    order = ["XPC", "MPG", "POLE", "POLD1", "MUTYH"]
    print(per_mutator(trios).set_index("gene").loc[order].to_string(float_format=lambda x: f"{x:.4g}"))
