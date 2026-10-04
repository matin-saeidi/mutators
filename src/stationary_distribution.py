"""
Stationary distribution of mutator frequencies (Eqs. 4-11 and S12-S21).

Used for the analytic curves in Figure 1 and for the starting frequencies of the
mutator simulations.

    N       population size
    phi_G   extra DNMs genome-wide transmitted by a mutator homozygote
    f       fraction of the genome under selection
    s_het   fitness cost of one deleterious allele
    s       selection coefficient of a mutator homozygote, s*
    h       dominance of the mutator (0 recessive, 0.5 semi-dominant)
    mu      mutation rate at the modifier site

Usage:
    s = selection_coefficient(phi_G=100, f=0.08, s_het=0.005)
    q, p = stationary_distribution(N=2000, s=s, h=0, mu=1.25e-7)
    mean_q = np.sum(q * p)
"""
import numpy as np
from scipy import stats
from scipy.integrate import quad


def selection_coefficient(phi_G, f, s_het, rtol=0.01):
    """s* from Eq. 5, or from the exponential form of Eq. S12 when the two
    differ by more than rtol."""
    s_linear = 2.0 * f * phi_G * s_het
    s_exponential = 1.0 - np.exp(-s_linear)
    if np.isclose(s_linear, s_exponential, rtol=rtol):
        return s_linear
    return s_exponential


def _selection_term(q, h):
    """q^2 + 2h q(1 - q)"""
    return q ** 2 * (1 - 2 * h) + 2 * h * q


def density(q, N, s, h, mu):
    """Unnormalised density between the boundaries, Eq. S14."""
    return (np.exp(-2 * N * s * _selection_term(q, h))
            * (q * (1 - q)) ** (4 * N * mu - 1))


def expected_change(q, s, h, mu):
    """Expected change in mutator frequency per generation, Eq. 4."""
    return -s * (q ** 2 * (1 - 2 * h) + h * q) * (1 - q) + mu * (1 - 2 * q)


def stationary_distribution(N, s, h, mu):
    """Probability of each frequency q = i / 2N, i = 0, ..., 2N (Eqs. S15-S20).

    Returns (q, p).
    """
    two_n = 2 * N
    half_width = 1 / (2 * two_n)
    q_interior = np.arange(1, two_n) / two_n

    mass = np.array([quad(density, x - half_width, x + half_width,
                          args=(N, s, h, mu))[0] for x in q_interior])

    # mass at q = 0 and q = 1
    q_next = np.clip(q_interior + expected_change(q_interior, s, h, mu), 0, 1)
    to_zero = stats.binom.pmf(0, two_n, q_next)
    to_fixed = stats.binom.pmf(two_n, two_n, q_next)
    leave_boundary = 1 - stats.binom.pmf(0, two_n, mu)
    mass_zero = np.sum(to_zero * mass) / leave_boundary
    mass_fixed = np.sum(to_fixed * mass) / leave_boundary

    interior = quad(density, half_width, 1 - half_width, args=(N, s, h, mu))[0]
    norm = mass_zero + mass_fixed + interior
    assert np.isclose(norm, mass_zero + mass_fixed + mass.sum())

    q = np.concatenate(([0.0], q_interior, [1.0]))
    p = np.concatenate(([mass_zero], mass, [mass_fixed])) / norm
    return q, p


def expected_delta_u(q, p, phi, h):
    """Increase in the mean mutation rate from one modifier site, Eq. 11.
    phi is per site, i.e. phi_G / G."""
    return phi * ((1 - 2 * h) * np.sum(q ** 2 * p) + 2 * h * np.sum(q * p))
