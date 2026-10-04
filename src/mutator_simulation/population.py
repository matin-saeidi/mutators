"""
The mutator simulation (Supplementary Section S3.1): N diploids with M recessive
modifier sites, each carrying some number of deleterious alleles. Run it with
simulate.py.

Each generation, parents are drawn in proportion to fitness (1 - s_het)^n_del.
A gamete inherits its parent's mutator alleles and deleterious alleles, then
gains new mutations at the modifier sites (rate mu, in both directions) and at
the f G selected sites (rate u0 + phi for each site at which the parent is
homozygous for the mutator). Starting frequencies come from the stationary
distribution (Eq. 7).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stationary_distribution import (selection_coefficient,     # noqa: E402
                                     stationary_distribution)


class Individual:
    """het, hom: modifier sites at which it is heterozygous / homozygous for the
    mutator (sites fixed for the mutator are in Population.fixed instead).
    n_del: number of deleterious alleles."""
    __slots__ = ("het", "hom", "n_del")

    def __init__(self):
        self.het = []
        self.hom = []
        self.n_del = 0


class Population:
    """N diploids with M modifier sites.

    phi_G  extra DNMs genome-wide transmitted by a homozygote for one mutator
    G      haploid genome length
    f      fraction of the genome under selection
    s_het  fitness cost of one deleterious allele
    u0     per-site mutation rate of an individual carrying no mutator
    mu     per-site mutation rate at modifier sites
    rng    numpy.random.RandomState
    """

    def __init__(self, N, M, phi_G, G, f, s_het, u0, mu, rng):
        self.N, self.M = N, M
        self.phi = phi_G / G              # increase in per-site rate, per homozygous site
        self.loci = f * G                 # selected sites per haploid genome
        self.s_het, self.u0, self.mu = s_het, u0, mu
        self.rng = rng
        self.fixed = []
        self.people = [Individual() for _ in range(N)]

        s = selection_coefficient(phi_G, f, s_het)
        q, p = stationary_distribution(N, s, h=0, mu=mu)
        q0 = rng.choice(q, size=M, p=p)
        mean_del = 2 * u0 * self.loci / s_het
        for ind in self.people:
            ind.n_del = rng.poisson(mean_del)
            for site in range(M):
                g = rng.binomial(2, q0[site])
                if g == 2:
                    ind.hom.append(site)
                elif g == 1:
                    ind.het.append(site)

    def next_generation(self):
        """Replace the population by its offspring and return the mutator
        frequency at each modifier site."""
        N, rng = self.N, self.rng
        n_del = np.fromiter((ind.n_del for ind in self.people), dtype=float, count=N)
        w = (1 - self.s_het) ** (n_del - n_del.mean())
        parents = rng.choice(N, size=2 * N, p=w / w.sum())

        hits_at_fixed = defaultdict(list)    # fixed site -> children with a mutation there
        children = []
        for i in range(N):
            gamete_a = self._gamete(self.people[parents[i]], i, hits_at_fixed)
            gamete_b = self._gamete(self.people[parents[i + N]], i, hits_at_fixed)
            children.append(self._child(gamete_a, gamete_b))
        self.people = children

        # a fixed site that mutated becomes segregating again
        for site, hit in hits_at_fixed.items():
            for i, ind in enumerate(self.people):
                (ind.het if i in hit else ind.hom).append(site)
            self.fixed.remove(site)

        return self._frequencies()

    def _gamete(self, parent, child_index, hits_at_fixed):
        rng = self.rng
        sites = list(parent.hom)
        if parent.het:
            passed = rng.binomial(1, 0.5, len(parent.het))
            sites += [parent.het[j] for j in np.where(passed)[0]]

        for site in rng.randint(0, self.M, rng.poisson(self.M * self.mu)):
            if site in sites:
                sites.remove(site)
            elif site in self.fixed:
                hits_at_fixed[site].append(child_index)
            else:
                sites.append(site)

        u = self.u0 + self.phi * (len(parent.hom) + len(self.fixed))
        n_del = rng.binomial(parent.n_del, 0.5) + rng.poisson(self.loci * u)
        return n_del, sites

    @staticmethod
    def _child(gamete_a, gamete_b):
        child = Individual()
        child.n_del = gamete_a[0] + gamete_b[0]
        sites_a = gamete_a[1]
        for site in gamete_b[1]:
            if site in sites_a:
                child.hom.append(site)
                sites_a.remove(site)
            else:
                child.het.append(site)
        child.het.extend(sites_a)
        return child

    def _frequencies(self):
        """Mutator frequency at each site; moves newly fixed sites to self.fixed."""
        het = np.fromiter((s for ind in self.people for s in ind.het), dtype=np.int64)
        hom = np.fromiter((s for ind in self.people for s in ind.hom), dtype=np.int64)
        counts = (np.bincount(het, minlength=self.M)
                  + 2 * np.bincount(hom, minlength=self.M))
        newly_fixed = np.flatnonzero(counts == 2 * self.N)
        counts[self.fixed] = 2 * self.N
        for site in newly_fixed:
            for ind in self.people:
                ind.hom.remove(site)
            self.fixed.append(int(site))
        return counts / (2 * self.N)
