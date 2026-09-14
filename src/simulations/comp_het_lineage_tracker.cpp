// ============================================================================
// Compound-het simulation with FULL lineage tracking (survivors only).
//
// MULTI-DEMOGRAPHY VERSION of lineage_tracker_with_demography.cpp.
// Identical model; the demographic history is now selectable at runtime.
//
// - Every forward mutation spawns a new Lineage {id, birth_gen, copies=1}.
// - Back mutations are allocated proportionally across existing lineages
//   using a sequential binomial (multinomial) split.
// - Wright-Fisher reproduction (selection+drift) happens via population::populate_from().
//   After that, we redistribute the NEW derived total K_next among lineages
//   using the same sequential binomial splitter (proportional to their shares).
// - Lineages with copies==0 are removed immediately.
// - At the end of each run (gen==0), we print ONLY surviving lineages with:
//     lineage_id, birth_gen, age (generations), copies, freq.
//
// Notes:
// * The simulation runs forward in time.  `gen` is labelled in generations
//   before the present, so it starts at burn_in (or taujump) and decreases to 0.
// * Age at present is birth_gen generations old.
// * Selection and dominance act via population.{prob(), populate_from()}.
// * There is NO jumping in this simulator: every generation is simulated
//   explicitly, so taujump only serves as the default value of initgen when
//   no burn_in argument is supplied.
//
// Inputs:
//   ./sim RUNS sel DOM mutU ufactor pid_seed allow_back_mutation [burn_in] [eur|afr|sas] [burnin_Ne]
//     RUNS                  : number of runs
//     sel, DOM              : selection coefficient (s), dominance (h)
//     mutU                  : per-copy forward mutation rate per gen
//     ufactor               : back-mutation factor; back rate = ufactor * mutU
//     pid_seed              : seed offset
//     allow_back_mutation   : 0 (off) or 1 (on)
//     burn_in   (optional)  : starting generation (default = taujump of the
//                             chosen population).  When > 0, the population is
//                             held at burnin_Ne for every gen > T[0].
//     eur|afr|sas (optional): demographic history (default eur).  May also be
//                             given in place of burn_in.
//     burnin_Ne (optional)  : Ne used during the burn-in (default 20000, the
//                             value hard-coded in the original code).
//
// Compile:
//   g++ -O2 -std=c++11 lineage_tracker_with_demography_multidem.cpp population.cpp BRand.cpp -o comp_het_lineage_tracker
// ============================================================================

#include <chrono>
#include <random>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <stdint.h>
#include <map>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cctype>
#include <algorithm>   // std::max, std::remove_if
#include <numeric>     // std::accumulate
#include <vector>
#include <iomanip>

#include "population.h"
#include "BRand.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

boost::mt19937 gent;  // global RNG engine

#define ratio 1

// --------------------------------------------------------------------
// Demographic histories (time runs forward).
// Convention: epoch i has size Ne[i] and covers T[i] < gen <= T[i-1];
// Ne[0] is the ancestral size, used for every gen > T[0].
// --------------------------------------------------------------------

// --- Non-Finnish European: Schiffels & Durbin, then Schraiber et al. ------
const int Ne_eur[] = {
  14448,14068,14068,14464,14464,15208,15208,16256,16256,17618,17618,19347,
  19347,21534,21534,24236,24236,27367,27367,30416,30416,32060,32060,31284,
  29404,26686,23261,18990,16490,16490,12958,12958,9827,9827,7477,7477,
  5791,5791,4670,4670,3841,3841,3372,3372,3287,3359,3570,4095,
  4713,5661,7540,11375,
  // --- switch to Joshua G. Schraiber demography at T = 448 ---
  15887, 79693, 290702, 721018, 1368726, 2166404, 4110830, 5096386,
  23368593, 31498556, 31498556
};

const int T_eur[] = {
  55940,51395,47457,43984,40877,38067,35501,33141,30956,28922,27018,25231,
  23545,21951,20439,19000,17628,16318,15063,13859,12702,11590,10517,9482,
  8483,7516,6580,5672,5520,5156,4817,4500,4203,3922,3656,3404,
  3165,2936,2718,2509,2308,2116,1930,1752,1579,1413,1252,1096,
  945,798,656,517,
  // --- Joshua G. Schraiber demography recent Ne change generations  ---
  448, 283, 179, 113, 71, 45, 28, 18, 11, 7, 0
};

// --- African -------------------------------------------------------------
const int Ne_afr[] = {
  18543,13957,13957,13579,13579,13998,13998,14790,14790,15927,15927,17473,
  17473,19524,19524,22146,22146,25269,25269,28538,28538,31332,31332,33012,
  33012,33256,32900,32311,31233,28721,28248,24560,24560,20871,20871,17828,
  17828,15152,15152,13476,13476,12864,12639,12501,13031,13547,13981,58563,
  271461,2763103
};

const int T_afr[] = {
  67220,60705,55383,50883,46984,43546,40470,37688,35148,32811,30648,28633,
  26749,24980,23311,21733,20235,18811,17453,16155,14913,13721,12576,11474,
  10412,9388,8398,7441,6514,5616,5274,4917,4578,4256,3949,3655,
  3374,3104,2845,2596,2356,2124,1900,1684,1474,1271,1200,292,
  127,0
};

// --- South Asian ---------------------------------------------------------
const int Ne_sas[] = {
  19663,14648,14648,14142,14142,14461,14461,15121,15121,16063,16063,17324,
  17324,18986,18986,21161,21161,23901,23901,27019,27019,29934,29934,31627,
  31627,31291,29878,27598,24499,20346,18325,18325,14256,14256,10706,10706,
  8148,8148,6399,6399,5340,5340,4510,4510,3973,3973,3758,3691,
  3683,3971,4360,49350,337083,4837454
};

const int T_sas[] = {
  71728,64777,59098,54296,50136,46467,43185,40216,37505,35012,32703,30554,
  28544,26655,24875,23190,21592,20073,18623,17239,15913,14641,13419,12244,
  11111,10018,8962,7940,6951,5993,5739,5361,5008,4679,4369,4077,
  3801,3539,3290,3053,2826,2608,2400,2200,2007,1821,1642,1469,
  1302,1139,1054,274,87,0
};

// --- Active demography, selected at runtime ------------------------------
const int *Ne = Ne_eur;
const int *T  = T_eur;
int NDEM      = (int)(sizeof(Ne_eur) / sizeof(Ne_eur[0]));
int taujump   = 56000;          // only the default initgen in this simulator
string POPLABEL = "eur";

int idx = 0;                      // demographic index tracker
int Ne_zero = 20000;              // burn-in Ne for the very ancient period
bool use_burnin_override = false; // only true if user provides burn_in

static bool is_pop_label(const string& sarg)
{
    string p;
    for (size_t i = 0; i < sarg.size(); ++i) p += (char)tolower((unsigned char)sarg[i]);
    return (p == "eur" || p == "nfe" || p == "eu" || p == "afr" || p == "af"
            || p == "sas" || p == "sa");
}

static bool select_demography(const string& pop)
{
    string p;
    for (size_t i = 0; i < pop.size(); ++i) p += (char)tolower((unsigned char)pop[i]);

    if (p == "eur" || p == "nfe" || p == "eu") {
        Ne = Ne_eur;
        T  = T_eur;
        NDEM = (int)(sizeof(Ne_eur) / sizeof(Ne_eur[0]));
        taujump = 56000;
        POPLABEL = "eur";
    } else if (p == "afr" || p == "af") {
        Ne = Ne_afr;
        T  = T_afr;
        NDEM = (int)(sizeof(Ne_afr) / sizeof(Ne_afr[0]));
        taujump = 68000;
        POPLABEL = "afr";
    } else if (p == "sas" || p == "sa") {
        Ne = Ne_sas;
        T  = T_sas;
        NDEM = (int)(sizeof(Ne_sas) / sizeof(Ne_sas[0]));
        taujump = 72000;
        POPLABEL = "sas";
    } else {
        return false;
    }

    if ((int)(sizeof(Ne_eur)/sizeof(Ne_eur[0])) != (int)(sizeof(T_eur)/sizeof(T_eur[0]))) {
        cerr << "FATAL: Ne_eur and T_eur have different lengths\n"; exit(1);
    }
    if ((int)(sizeof(Ne_afr)/sizeof(Ne_afr[0])) != (int)(sizeof(T_afr)/sizeof(T_afr[0]))) {
        cerr << "FATAL: Ne_afr and T_afr have different lengths\n"; exit(1);
    }
    if ((int)(sizeof(Ne_sas)/sizeof(Ne_sas[0])) != (int)(sizeof(T_sas)/sizeof(T_sas[0]))) {
        cerr << "FATAL: Ne_sas and T_sas have different lengths\n"; exit(1);
    }
    for (int i = 1; i < NDEM; ++i) {
        if (T[i] >= T[i-1]) {
            cerr << "FATAL: T is not strictly decreasing at index " << i
                 << " (" << T[i-1] << " -> " << T[i] << ")\n";
            exit(1);
        }
    }
    if (T[NDEM-1] != 0) {
        cerr << "FATAL: last entry of T must be 0 (got " << T[NDEM-1] << ")\n";
        exit(1);
    }
    return true;
}

// --------------------------------------------------------------------
// Lineage bookkeeping
// --------------------------------------------------------------------
struct Lineage {
    int id;          // unique lineage ID
    int birth_gen;   // generation when it was created (remember: counting down)
    int copies;      // number of derived allele copies
};

int next_lineage_id = 0;   // per-run counter for unique IDs

// --------------------------------------------------------------------
// Simulation parameters
// --------------------------------------------------------------------
double sel, DOM, mutU, ufactor;
int RUNS, pid_seed;
bool allow_back_mutation = true;

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

// Demographic size at generation `gen` (advances idx when crossing T[idx])
inline int popsize(int gen, int /*i*/) {
    // If burn-in was requested and we're older than the first demography
    // breakpoint, force Ne = Ne_zero (default 20000) until we reach gen <= T[0].
    if (use_burnin_override && gen > T[0]) {
        return Ne_zero;
    }

    // Normal demography schedule.  Epochs are indexed in generations before the
    // present, so a decreasing `gen` advances forward through them.
    // NDEM is the number of epochs of whichever demography was selected
    // (63 for eur, 50 for afr).  A while-loop instead of a single `if` so
    // that a call which skips several epochs still lands on the right one.
    while (idx < NDEM - 1 && gen <= T[idx]) {
        idx++;
    }

    return Ne[idx];
}

// Poisson draw (0-safe)
inline int boost_poi(double l) {
    if (l <= 0.0) return 0;
    boost::random::poisson_distribution<> dist(l);
    return dist(gent);
}

// Binomial draw (guarded)
inline int draw_binom(int n, double p) {
    if (n <= 0)   return 0;
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;
    std::binomial_distribution<> dist(n, p);
    return dist(gent);
}

// Sequential binomial (multinomial) splitter:
// Split `total` events across lanes with integer weights.
// Ensures sum(out) == total; each lane i gets Bin(rem_total, w_i / rem_weight).
static std::vector<int> split_by_weights(const std::vector<int>& weights, int total) {
    std::vector<int> out(weights.size(), 0);
    long long rem_total  = total;
    long long rem_weight = std::accumulate(weights.begin(), weights.end(), 0LL);

    for (size_t i = 0; i < weights.size(); ++i) {
        if (rem_total <= 0 || rem_weight <= 0) break;

        double p = double(weights[i]) / double(rem_weight);
        int take = (i + 1 < weights.size())
                   ? draw_binom(int(rem_total), p)
                   : int(rem_total); // last lane gets the remainder exactly

        out[i]       = take;
        rem_total   -= take;
        rem_weight  -= weights[i];
    }
    return out;
}

static void usage(const char* prog) {
    cout << "Usage:\n"
         << "  " << prog << " RUNS sel DOM mutU ufactor pid_seed allow_back_mutation"
            " [burn_in] [eur|afr|sas] [burnin_Ne]\n\n"
         << "Optional:\n"
         << "  burn_in   : starting generation (default = taujump; 56000 eur, 68000 afr, 72000 sas)\n"
         << "  eur|afr|sas : demographic history (default eur); may be given in place of burn_in\n"
         << "  burnin_Ne : Ne held during the burn-in, i.e. for gen > T[0] (default 20000)\n";
}

// --------------------------------------------------------------------
// Main
// --------------------------------------------------------------------
int main(int argc, char *argv[]) {
    long long burn_in_arg = 0;
    string pop_arg = "eur";       // default keeps the original behaviour
    int burnin_Ne_arg = 20000;    // default matches the original hard-coded value

    if (argc >= 8 && argc <= 11) {
        RUNS     = atoi(argv[1]);
        sel      = atof(argv[2]);
        DOM      = atof(argv[3]);
        mutU     = atof(argv[4]);
        ufactor  = atof(argv[5]);
        pid_seed = atoi(argv[6]);
        allow_back_mutation = (atoi(argv[7]) != 0); // 0=off, 1=on

        // argv[8] may be either burn_in or the population label, so that
        // "... allow_back_mut afr" works without having to state a burn-in.
        int next = 8;
        if (argc > next) {
            if (is_pop_label(argv[next])) {
                pop_arg = argv[next];
                next++;
            } else {
                burn_in_arg = (long long)std::stod(argv[next]);
                next++;
                if (argc > next) { pop_arg = argv[next]; next++; }
            }
        }
        if (argc > next) { burnin_Ne_arg = (int)std::stod(argv[next]); next++; }
        if (argc > next) { usage(argv[0]); return 1; }

    } else {
        usage(argv[0]);
        return 1;
    }

    if (!select_demography(pop_arg)) {
        cerr << "FATAL: unknown population '" << pop_arg
             << "' (expected 'eur', 'afr' or 'sas')\n";
        return 1;
    }

    int initgen = taujump;  // default if burn-in not provided
    if (burn_in_arg > 0) {
        initgen = (int)burn_in_arg;
        use_burnin_override = true;
    }
    if (burnin_Ne_arg <= 0) {
        cerr << "FATAL: burnin_Ne must be positive (got " << burnin_Ne_arg << ")\n";
        return 1;
    }
    Ne_zero = burnin_Ne_arg;

    if (use_burnin_override && initgen <= T[0]) {
        cerr << "FATAL: burn_in (" << initgen << ") must exceed T[0] (" << T[0]
             << ") for population '" << POPLABEL << "'\n";
        return 1;
    }

    // Echo the configuration on stderr so it lands in the SLURM .err file and
    // never contaminates the .out file that the summarise scripts parse.
    cerr << "# demography=" << POPLABEL
         << " epochs=" << NDEM
         << " Ne_ancestral=" << Ne[0]
         << " T_oldest=" << T[0]
         << " Ne_present=" << Ne[NDEM-1]
         << " taujump=" << taujump
         << " initgen=" << initgen
         << " burnin_Ne=" << (use_burnin_override ? Ne_zero : Ne[0])
         << " burnin_override=" << (use_burnin_override ? 1 : 0)
         << endl;

    // Seed RNGs
    gent.seed(time(NULL) + pid_seed);
    BRand::Controller.seed(time(NULL) + pid_seed);

    // Initialize selection params in the population class
    population::initialize(sel, DOM);

    // Single-deme model
    population* pops[2];
    pops[0] = new population(popsize(initgen, 0));

    int gen;

    for (int run = 0; run < RUNS; run++) {
        gen = initgen;
        idx = 0;
        double Uvar = mutU;

        pops[0]->size = popsize(gen, 0);
        pops[0]->clear();  // ancestral fixed at start of run

        // ----- Lineages alive during this run -----
        vector<Lineage> lineages;
        next_lineage_id = 0;

        while (gen > 0) {
            int ploidy    = 2 * pops[0]->size;
            int K_before  = pops[0]->allelenum();      // total derived copies now
            int anc_copies = ploidy - K_before;        // ancestral copies now

            // (1) Forward mutations: each new derived copy spawns a new lineage
            int new_total = boost_poi((anc_copies > 0) ? Uvar * anc_copies : 0.0);

            if (new_total > 0) {
                pops[0]->mutateup(new_total);   // update genotype counts

                // create 1 lineage per new mutation (each starts with copies=1)
                for (int m = 0; m < new_total; ++m) {
                    lineages.push_back(Lineage{ next_lineage_id++, gen, 1 });
                }
            }

            // (2) Back mutations: convert derived -> ancestral, allocated across lineages
            if (allow_back_mutation && !lineages.empty()) {
                int K_after_up = pops[0]->allelenum();
                int back_total = boost_poi(ufactor * Uvar * K_after_up);
                if (back_total > K_after_up) back_total = K_after_up;

                if (back_total > 0) {
                    pops[0]->mutatedown(back_total);   // update genotype counts

                    // proportionally remove from lineages via multinomial split
                    std::vector<int> weights;
                    weights.reserve(lineages.size());
                    for (auto& L : lineages) weights.push_back(L.copies);

                    auto losses = split_by_weights(weights, back_total);

                    for (size_t j = 0; j < lineages.size(); ++j) {
                        lineages[j].copies -= losses[j];
                        if (lineages[j].copies < 0) lineages[j].copies = 0;
                    }

                    // prune dead lineages immediately
                    lineages.erase(
                        std::remove_if(lineages.begin(), lineages.end(),
                                       [](const Lineage& L){ return L.copies == 0; }),
                        lineages.end()
                    );
                }
            }

            // (3) Reproduction (Wright-Fisher with selection via population::prob())
            pops[0]->populate_from(pops[0]->prob(), popsize(gen - 1, 0));
            int K_next = pops[0]->allelenum();  // total derived copies after reproduction

            // (4) Redistribute K_next among lineages (multinomial proportional to current shares)
            if (!lineages.empty()) {
                std::vector<int> weights;
                weights.reserve(lineages.size());
                for (auto& L : lineages) weights.push_back(L.copies); // shares BEFORE reproduction

                auto new_counts = split_by_weights(weights, K_next);

                for (size_t j = 0; j < lineages.size(); ++j) {
                    lineages[j].copies = new_counts[j];
                }

                // prune dead lineages
                lineages.erase(
                    std::remove_if(lineages.begin(), lineages.end(),
                                   [](const Lineage& L){ return L.copies == 0; }),
                    lineages.end()
                );
            } else {
                // If there are no lineages, K_next should be 0. If not, something
                // created derived copies without a lineage (shouldn't happen).
                // We ignore this corner as forward mutation always creates lineages.
            }

            gen--;
        }

        // --- Output survivors for this run ---
        cout << "Run " << run << " survivors at present:" << "\n";

        int total_lineage_copies = 0;
        for (const auto& L : lineages) {
            total_lineage_copies += L.copies;
            int age = L.birth_gen;  // because present is gen=0, so age = birth_gen generations old
            double freq = double(L.copies) / (2.0 * pops[0]->size);
            cout << "  lineage_id=" << L.id
                << " age=" << age
                << " copies=" << L.copies
                << " freq=" << std::setprecision(10) << freq
                << "\n";
        }

        // total frequency across all lineages
        double total_freq = double(total_lineage_copies) / (2.0 * pops[0]->size);
        cout << "  Total derived freq = " << total_freq << "\n";

        // sanity check
        int total_population_copies = pops[0]->allelenum();
        if (total_lineage_copies != total_population_copies) {
            cerr << "ERROR: mismatch in copy counts at end of run "
                 << run << ": lineage sum=" << total_lineage_copies
                 << " vs population=" << total_population_copies << endl;
            exit(1);
        }
    }

    return 0;
}
