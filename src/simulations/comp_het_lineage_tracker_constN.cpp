// comp_het_lineage_tracker_constN.cpp
//
// Constant-N Wright–Fisher simulation with full lineage tracking.
// - Time runs forward: gen = 1..G (present is gen == G).
// - Each forward mutation creates a new lineage {id, birth_gen = gen, copies = 1}.
// - Back mutations occur per derived copy at rate ufactor * mutU,
//   and are allocated proportionally across existing lineages (multinomial via sequential binomials).
// - Reproduction with selection+drift uses population::populate_from(population::prob(), N).
// - After reproduction, lineage copy numbers are reassigned proportionally to their pre-repro shares
//   to keep identities consistent through drift/selection.
// - Lineages with copies == 0 are removed.
// - At the end of each run, print surviving lineages with:
//     lineage_id, age = G - birth_gen, copies, freq = copies / (2N),
//   plus a sanity check that total lineage copies == population derived copies.
//
// Build:
//   g++ -O2 -std=c++11 comp_het_lineage_tracker_constN.cpp population.cpp BRand.cpp -o comp_het_lineage_tracker_constN
//
// Run:
//   ./comp_het_lineage_tracker_constN RUNS N G sel DOM mutU ufactor pid_seed allow_back_mutation
//   e.g. ./comp_het_lineage_tracker_constN 1000 20000 56000 0.01 0.5 1e-8 1.0 42 1

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
#include <algorithm>
#include <numeric>
#include <vector>
#include <iomanip>

#include "population.h"
#include "BRand.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>

using std::cout;
using std::endl;
using std::vector;

boost::mt19937 gent;  // shared RNG (also used by population.cpp if needed)

// ------------------------------
// Lineage bookkeeping
// ------------------------------
struct Lineage {
    int id;         // unique lineage ID (per run)
    int birth_gen;  // generation when lineage arose (1..G)
    int copies;     // number of derived copies currently in this lineage
};

static int next_lineage_id = 0;

// ------------------------------
// Simulation parameters
// ------------------------------
int RUNS, N, G;
double sel, DOM, mutU, ufactor;
int pid_seed;
bool allow_back_mutation = true;

// ------------------------------
// RNG helpers
// ------------------------------
inline int boost_poi(double l) {
    if (l <= 0.0) return 0;
    boost::random::poisson_distribution<> dist(l);
    return dist(gent);
}

inline int draw_binom(int n, double p) {
    if (n <= 0)   return 0;
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;
    std::binomial_distribution<> dist(n, p);
    return dist(gent);
}

// ------------------------------
// Multinomial via sequential binomials:
// Split `total` events across "lanes" with integer weights,
// preserving sum(out) == total.
// ------------------------------
static std::vector<int> split_by_weights(const std::vector<int>& weights, int total) {
    std::vector<int> out(weights.size(), 0);
    long long rem_total  = total;
    long long rem_weight = std::accumulate(weights.begin(), weights.end(), 0LL);

    for (size_t i = 0; i < weights.size(); ++i) {
        if (rem_total <= 0 || rem_weight <= 0) break;

        const double p = double(weights[i]) / double(rem_weight);
        const int take = (i + 1 < weights.size())
                         ? draw_binom(int(rem_total), p)
                         : int(rem_total);  // last lane gets the remainder

        out[i]      = take;
        rem_total  -= take;
        rem_weight -= weights[i];
    }
    return out;
}

// ------------------------------
// Main
// ------------------------------
int main(int argc, char* argv[]) {
    if (argc != 10) {
        cout << "Usage: ./constN_lineages RUNS N G sel DOM mutU ufactor pid_seed allow_back_mutation\n";
        return 1;
    }

    // Parse CLI
    RUNS     = (int)atof(argv[1]);
    N        = (int)atof(argv[2]);
    G        = (int)atof(argv[3]);
    sel      = atof(argv[4]);
    DOM      = atof(argv[5]);
    mutU     = atof(argv[6]);
    ufactor  = atof(argv[7]);
    pid_seed = (int)atof(argv[8]);
    allow_back_mutation = ((int)atof(argv[9]) != 0);

    // Seed RNGs
    gent.seed(time(NULL) + pid_seed);
    BRand::Controller.seed(time(NULL) + pid_seed);

    // Initialize selection/dominance in population class
    population::initialize(sel, DOM);

    // Single-deme model (pops[1] unused here but kept for compatibility)
    population* pops[2];
    pops[0] = new population(N);
    pops[1] = new population(N);

    for (int run = 0; run < RUNS; ++run) {
        // Reset population to ancestral fixed state
        pops[0]->size = N;
        pops[0]->clear();

        // Lineages alive during this run
        vector<Lineage> lineages;
        next_lineage_id = 0;

        // Simulate forward in time: gen = 1..G (present is gen == G)
        for (int gen = 1; gen <= G; ++gen) {
            const int ploidy     = 2 * pops[0]->size;     // diploid, constant N
            const int K_before   = pops[0]->allelenum();  // total derived copies before mutations
            const int anc_copies = ploidy - K_before;     // ancestral copies before mutations

            // (1) Forward mutations on ancestral copies (per-copy rate mutU)
            const int new_total = boost_poi((anc_copies > 0) ? mutU * anc_copies : 0.0);
            if (new_total > 0) {
                // Update population state
                pops[0]->mutateup(new_total);

                // Create one lineage per new mutation (each starts with 1 copy)
                for (int m = 0; m < new_total; ++m) {
                    lineages.push_back(Lineage{ next_lineage_id++, gen, 1 });
                }
            }

            // (2) Back mutations on derived copies (per-copy rate ufactor * mutU),
            //     allocated proportionally across lineages by current copy counts.
            if (allow_back_mutation && !lineages.empty()) {
                const int K_after_up = pops[0]->allelenum();              // derived total after forward muts
                int back_total       = boost_poi(ufactor * mutU * K_after_up);
                if (back_total > K_after_up) back_total = K_after_up;     // safety cap

                if (back_total > 0) {
                    // Apply to population state
                    pops[0]->mutatedown(back_total);

                    // Proportional allocation of back mutations across lineages
                    std::vector<int> weights;
                    weights.reserve(lineages.size());
                    for (const auto &L : lineages) weights.push_back(L.copies);

                    const auto losses = split_by_weights(weights, back_total);

                    // Reduce lineage copies and prune zeros
                    for (size_t j = 0; j < lineages.size(); ++j) {
                        lineages[j].copies -= losses[j];
                        if (lineages[j].copies < 0) lineages[j].copies = 0;
                    }
                    lineages.erase(
                        std::remove_if(lineages.begin(), lineages.end(),
                                       [](const Lineage& L){ return L.copies == 0; }),
                        lineages.end()
                    );
                }
            }

            // (3) Wright–Fisher reproduction with selection/drift to constant size N
            pops[0]->populate_from(pops[0]->prob(), N);
            const int K_next = pops[0]->allelenum();  // derived total after reproduction

            // (4) Reassign K_next to lineages proportionally to their pre-repro shares
            if (!lineages.empty()) {
                std::vector<int> weights;
                weights.reserve(lineages.size());
                for (const auto &L : lineages) weights.push_back(L.copies);

                const auto new_counts = split_by_weights(weights, K_next);

                for (size_t j = 0; j < lineages.size(); ++j) {
                    lineages[j].copies = new_counts[j];
                }

                // Prune any that dropped to zero
                lineages.erase(
                    std::remove_if(lineages.begin(), lineages.end(),
                                   [](const Lineage& L){ return L.copies == 0; }),
                    lineages.end()
                );
            } else {
                // If no lineages exist, K_next should be 0; nothing to do.
            }
        } // end generations

        // ----- Output survivors at present (gen == G) -----
        cout << "Run " << run << " survivors at present:\n";

        int total_lineage_copies = 0;
        for (const auto& L : lineages) {
        total_lineage_copies += L.copies;
        const double denom = 2.0 * pops[0]->size;
        const double freq  = (denom > 0.0) ? (double(L.copies) / denom) : 0.0;
        const int age      = G - L.birth_gen+1;
        cout << "  lineage_id=" << L.id
            << " age=" << age
            << " copies=" << L.copies
            << " freq=" << std::setprecision(10) << freq
            << "\n";
    }

        const double total_freq = double(total_lineage_copies) / (2.0 * N);
        cout << "  Total derived freq = " << std::setprecision(12) << total_freq << "\n";

        // Sanity check: lineage sum must match population derived copies
        const int pop_copies = pops[0]->allelenum();
        if (pop_copies != total_lineage_copies) {
            std::cerr << "ERROR: lineage sum (" << total_lineage_copies
                      << ") != population (" << pop_copies << ")\n";
            return 2;
        }
    } // runs

    return 0;
}
