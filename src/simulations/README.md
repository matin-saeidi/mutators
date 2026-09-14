# Forward simulators

Wright-Fisher forward simulations of a mutator allele. Four binaries, built here
with `make`:

| binary | model | used for |
| --- | --- | --- |
| `single_site_simulator` | single site, back mutation on | MPG, POLD1, POLE |
| `single_site_simulator_no_back_mut` | single site, back mutation off | the back-mutation comparison |
| `comp_het_lineage_tracker` | compound heterozygote with lineage tracking, under a demographic history | XPC, MUTYH |
| `comp_het_lineage_tracker_constN` | the same at constant `Ne` | the XPC mutation-rate comparison |

```bash
make          # -> bin/
make clean
```

Needs a C++11 compiler and the Boost headers (`boost/random`, header-only).
Built and run with Boost 1.66.

## Invocation

```
single_site_simulator  mutU sel DOM RUNS mut_uncert dem_uncert dem_model Ne seed num_gens [eur|afr|sas]
comp_het_lineage_tracker  RUNS sel DOM mutU ufactor seed allow_back_mut [burn_in] [eur|afr|sas] [burnin_Ne]
comp_het_lineage_tracker_constN  RUNS N G sel DOM mutU ufactor seed allow_back_mut
```

`DOM` is the dominance of the mutator allele itself and `sel` its selection
coefficient in a homozygote. The workflows in `workflow/` supply every argument.

## The two single-site binaries

Both are built from `single_site_simulator.cpp`, selected at compile time:

```
-DBACK_MUT_RATIO=1    back-mutation rate equal to the forward rate
-DBACK_MUT_RATIO=0    back mutation off
```

The macro is spelled `BACK_MUT_RATIO` on the command line and aliased to `ratio`
after the includes, because `std::ratio` is a standard template used by
`<chrono>` and a `-Dratio=` would be seen by the standard headers.

Setting the ratio to zero also takes three guarded branches, described at those
points in the source: the reciprocal of the back rate is not formed, no
exponential waiting time is drawn for a back mutation while the derived allele
is fixed, and the Poisson-conditioned-on-at-least-one draw is skipped. All three
guards are inactive when the ratio is positive, which was checked against the
two separate sources these runs originally used: at `-DBACK_MUT_RATIO=0` the
preprocessed output is byte-identical to the back-mutation-free source, and at
`-DBACK_MUT_RATIO=1` the `-O3` assembly is byte-identical to the baseline
source.

## Demographic histories

`eur`, `afr` and `sas` select the history, compiled in as a pair of `Ne[]` /
`T[]` arrays. The simulation runs forward in time; the arrays are tabulated in
generations before the present, so the generation counter starts in the past and
decreases to 0.

Dump the active history as a table:

```bash
./bin/single_site_simulator --dump-demography sas
```

This steps through the whole history one generation at a time through the real
`popsize()` and checks that every epoch is entered once, at `T[i-1]`, returning
`Ne[i]`. `data/demographic_models/*.tsv` is generated from this command by
`tools/dump_demographic_models.sh`.

The binaries assert `taujump > T[0]` at startup.

| | eur | afr | sas |
| --- | --- | --- | --- |
| epochs | 63 | 50 | 54 |
| `taujump` | 56 000 | 68 000 | 72 000 |
| ancestral Ne | 14 448 | 18 543 | 19 663 |
| oldest breakpoint `T[0]` | 55 940 | 67 220 | 71 728 |
| present-day Ne | 31 498 556 | 2 763 103 | 4 837 454 |

The gnomAD genetic ancestry group written `nfe` in `data/` is the history called
`eur` here.

## Seeding

Both simulators seed from the wall clock:

```c++
gent.seed(time(NULL) + pid_seed);
```

so the `seed` argument is an offset on the current time, not a seed. Re-running
a command with identical arguments gives different replicates, and a resubmitted
chunk does not repeat its original draw. The workflows record the offset used
for every job in `params.txt`.

## Provenance

These are modified versions, with two separate ancestries:

* `single_site_simulator.cpp` descends from the single-site autosomal simulator
  of Yuval Simons (Simons et al. 2014), modified by Eduardo Amorim (Amorim et
  al. 2017), then by Zach Fuller, and then substantially here.
* `comp_het_lineage_tracker.cpp` and `comp_het_lineage_tracker_constN.cpp`
  descend from the forward simulators of Ipsita Agarwal, at
  <https://github.com/agarwal-i/loss-of-function-fitness-effects>.
* `population.{cpp,h}` and `BRand.{cpp,hpp}` are shared support code from the
  same lineage. Both upstream trees carried a copy of `population.cpp`; they
  differ only in whitespace, comments and dead commented-out code, so one copy
  is shipped.

Full citations are in the paper's bibliography.
