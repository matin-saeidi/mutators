"""
Proband discovery in trio studies without simulation (Supplementary Section S6):
the parental-age model (Eqs. S23, S29-S31), the proband threshold, Pr(proband)
in a trio with one mutator parent (Figures 4A, S11), Pr(mutator parent) at one
modifier site (Eq. 16; Figures S6, S12) and Pr(proband) with no mutator parent
(Eq. S36).

    python3 src/proband/analytic.py

prints Pr(proband) with no mutator parent, for one trio and for a cohort of
22,000 trios (Section S6.2).
"""
import numpy as np
from scipy.stats import poisson, truncnorm

ALPHA_F, BETA_F = 6.05, 1.51
ALPHA_M, BETA_M = 3.61, 0.37
AGE_F = dict(mean=32.0, sd=9.0, lo=17, hi=71)
AGE_M = dict(mean=28.2, sd=6.5, lo=16, hi=47)
E_D = 70.3                      # expected DNMs per child, averaged over ages


def dnms_father(age):
    return ALPHA_F + BETA_F * age


def dnms_mother(age):
    return ALPHA_M + BETA_M * age


def proband_threshold(nu_father, nu_mother):
    """Smallest DNM count that makes a child a proband."""
    return np.ceil(2.0 * (nu_father + nu_mother))


def fold_minus_one(phi_G):
    """Phi - 1 = 2 phi_G / E(D), Eq. S31."""
    return 2.0 * phi_G / E_D


def age_grid(mean, sd, lo, hi, n=200):
    """n equally spaced ages on [lo, hi] and their truncated-normal weights."""
    x = np.linspace(lo, hi, n)
    w = truncnorm.pdf(x, (lo - mean) / sd, (hi - mean) / sd, loc=mean, scale=sd)
    return x, w / w.sum()


def _trio_grid(n_ages):
    """Expected paternal and maternal DNMs, joint weight and threshold on an
    n_ages x n_ages grid of parental ages (fathers down rows)."""
    a_f, w_f = age_grid(**AGE_F, n=n_ages)
    a_m, w_m = age_grid(**AGE_M, n=n_ages)
    nu_f = dnms_father(a_f)[:, None]
    nu_m = dnms_mother(a_m)[None, :]
    return nu_f, nu_m, w_f[:, None] * w_m[None, :], proband_threshold(nu_f, nu_m)


def pr_proband_carrier(phi_G, dosage, carrier, n_ages=200):
    """Pr(proband) in a trio in which only the father or only the mother
    ('father' / 'mother') expresses one mutator, with the given dosage,
    averaged over parental ages (Figure S11). phi_G may be an array."""
    nu_f, nu_m, weight, k = _trio_grid(n_ages)
    out = []
    for x in np.atleast_1d(phi_G):
        fold = 1.0 + fold_minus_one(x) * dosage
        lam = fold * nu_f + nu_m if carrier == "father" else nu_f + fold * nu_m
        out.append(np.sum(weight * poisson.sf(k - 1, lam)))
    return np.array(out)


def pr_proband_given_mutator_parent(phi_G, dosage, n_ages=200):
    """The same, with the carrier equally likely to be either parent (Figure 4A)."""
    return 0.5 * (pr_proband_carrier(phi_G, dosage, "father", n_ages)
                  + pr_proband_carrier(phi_G, dosage, "mother", n_ages))


def pr_mutator_parent_one_site(q, n_trios, h):
    """Pr(at least one of 2n parents expresses the mutator) at a single
    modifier site, averaged over the frequencies q (Eq. 16, exact forms):
    1 - E[(1 - q^2)^2n] if h = 0, 1 - E[(1 - q)^4n] if h > 0."""
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    if h == 0:
        none = np.exp(2 * n_trios * np.log1p(-q ** 2))
    else:
        none = np.exp(4 * n_trios * np.log1p(-q))
    return 1.0 - np.mean(none)


def pr_proband_no_mutator(n_ages=2000):
    """Pr(proband) in a trio with no mutator parent, Eq. S36."""
    nu_f, nu_m, weight, k = _trio_grid(n_ages)
    return np.sum(weight * poisson.sf(k - 1, nu_f + nu_m))


if __name__ == "__main__":
    p = pr_proband_no_mutator()
    print(f"Pr(proband | no mutator parent)        = {p:.2e}")
    print(f"Pr(>= 1 such proband in 22,000 trios)  = {-np.expm1(22000 * np.log1p(-p)):.2e}")
