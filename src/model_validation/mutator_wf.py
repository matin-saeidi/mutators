"""
Wright-Fisher engine shared by the two model-validation simulation scripts.

A single modifier (mutator) site in a population of N diploids.  Individual i owns
haplotype slots 2i and 2i+1 of every per-haplotype array, so its modifier genotype is
mutator[2i] + mutator[2i+1] (0, 1 or 2 copies).

Each generation:
  * parents are sampled with replacement in proportion to
        w = (1 - s_het) ** (number of excess DELETERIOUS DNMs the individual carries)
  * a gamete takes the modifier allele from one of its parent's two haplotypes
  * free recombination between the modifier and everything else: each excess DNM the
    parent carries is passed to the gamete independently with probability 1/2
  * the parent adds new DNMs, Poisson with rate phi_G / h*phi_G / 0 for an
    MM / Mm / mm parent, of which a fraction f is deleterious (drawn by Poisson
    thinning: deleterious and neutral are independent Poissons)

Notation (matches the manuscript, and the notebook cell this replaces):
    h        mutator dominance
    phi_G    excess DNMs per gamete from a homozygous mutator
    f        fraction of DNMs that are deleterious
    s_het    fitness cost of carrying one excess deleterious DNM
    lambda   = phi_G * (h + q(1 - 2h))     excess DNMs linked to a mutator per generation
    2 lambda                               asymptotic excess DNMs linked to a mutator
    s        = 2 f lambda s_het            selection coefficient on the mutator

There are two entry points because the two figure rows need different things:

  simulate_dnms()  tracks deleterious AND neutral excess DNMs, so it can report
                   lambda_tau (all DNMs).  Needed by the trajectory row.
  simulate_del()   tracks deleterious excess DNMs only.  Fitness and the observed
                   selection coefficient depend on nothing else, so the dynamics are
                   identical in distribution while ~40% cheaper per generation.
                   Used by the s-versus-q row, where 10^4 replicates are run per h.
"""

import numpy as np


def lambda_per_gen(phi_G, h, q):
    """lambda: excess DNMs linked to a mutator haplotype per generation."""
    return phi_G * (h + q * (1.0 - 2.0 * h))


def two_lambda(phi_G, h, q):
    """2 lambda: asymptotic excess DNMs linked to a mutator haplotype."""
    return 2.0 * lambda_per_gen(phi_G, h, q)


def s_expected(phi_G, f, h, q, s_het):
    """s = 2 f lambda s_het: only the deleterious fraction f feeds back on fitness."""
    return s_het * f * two_lambda(phi_G, h, q)


def make_rng(seed, rep):
    """Independent stream per replicate, so chunked runs reproduce exactly."""
    return np.random.default_rng([int(seed), int(rep)])


def _sample_parents(rng, del_cnt, N, s_het):
    """Fitness-proportional parent indices for the 2N gametes of the next generation."""
    n_del = del_cnt[0::2] + del_cnt[1::2]          # deleterious DNMs per individual
    log_w = n_del * np.log1p(-s_het)               # log fitness, numerically stable
    log_w -= log_w.max()
    probs = np.exp(log_w)
    probs /= probs.sum()
    return rng.choice(N, size=2 * N, replace=True, p=probs)


def simulate_dnms(N, phi_G, f, h, q_init, n_gen, s_het, rng):
    """
    Track deleterious and neutral excess DNMs separately.

    Returns (lam, lam_del, q_track), each of length n_gen + 1:
        lam[t]     excess DNMs (all of them) on mutator vs non-mutator haplotypes
        lam_del[t] the same restricted to deleterious DNMs
        q_track[t] mutator frequency
    lam and lam_del are NaN in any generation where the mutator is absent or fixed.
    """
    mutator = rng.binomial(1, q_init, size=2 * N).astype(np.int8)
    del_cnt = np.zeros(2 * N, dtype=np.int64)
    neu_cnt = np.zeros(2 * N, dtype=np.int64)
    lam     = np.full(n_gen + 1, np.nan)
    lam_del = np.full(n_gen + 1, np.nan)
    q_track = np.empty(n_gen + 1)

    for t in range(n_gen + 1):
        is_M = mutator == 1
        q_track[t] = is_M.mean()
        if is_M.any() and (~is_M).any():
            dnm_cnt    = del_cnt + neu_cnt
            lam[t]     = dnm_cnt[is_M].mean() - dnm_cnt[~is_M].mean()
            lam_del[t] = del_cnt[is_M].mean() - del_cnt[~is_M].mean()
        if t == n_gen:
            break

        parent    = _sample_parents(rng, del_cnt, N, s_het)
        which_hap = rng.integers(0, 2, size=2 * N)
        new_mut   = mutator[2 * parent + which_hap]
        inh_del   = rng.binomial(del_cnt[2 * parent] + del_cnt[2 * parent + 1], 0.5)
        inh_neu   = rng.binomial(neu_cnt[2 * parent] + neu_cnt[2 * parent + 1], 0.5)

        geno = mutator[0::2].astype(np.int32) + mutator[1::2].astype(np.int32)
        rate = np.where(geno[parent] == 2,      phi_G,
               np.where(geno[parent] == 1,  h * phi_G, 0.0))

        mutator = new_mut
        del_cnt = inh_del + rng.poisson(f * rate)
        neu_cnt = inh_neu + rng.poisson((1.0 - f) * rate)

    return lam, lam_del, q_track


def simulate_del(N, phi_G, f, h, q_init, n_gen, s_het, rng):
    """
    Track deleterious excess DNMs only (neutral ones affect nothing measured here).

    Returns (lam_del, q_track), each of length n_gen + 1; lam_del[t] is NaN in any
    generation where the mutator is absent or fixed.
    """
    mutator = rng.binomial(1, q_init, size=2 * N).astype(np.int8)
    del_cnt = np.zeros(2 * N, dtype=np.int64)
    lam_del = np.full(n_gen + 1, np.nan)
    q_track = np.empty(n_gen + 1)

    for t in range(n_gen + 1):
        is_M = mutator == 1
        q_track[t] = is_M.mean()
        if is_M.any() and (~is_M).any():
            lam_del[t] = del_cnt[is_M].mean() - del_cnt[~is_M].mean()
        if t == n_gen:
            break

        parent    = _sample_parents(rng, del_cnt, N, s_het)
        which_hap = rng.integers(0, 2, size=2 * N)
        new_mut   = mutator[2 * parent + which_hap]
        inh_del   = rng.binomial(del_cnt[2 * parent] + del_cnt[2 * parent + 1], 0.5)

        geno = mutator[0::2].astype(np.int32) + mutator[1::2].astype(np.int32)
        rate = np.where(geno[parent] == 2,      phi_G,
               np.where(geno[parent] == 1,  h * phi_G, 0.0))

        mutator = new_mut
        del_cnt = inh_del + rng.poisson(f * rate)

    return lam_del, q_track
