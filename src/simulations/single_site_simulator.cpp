//Modified by Eduardo Amorim (guerraamorim@gmail.com) from the original of Yuval Simons (Simons et al. 2014).
//Modified by Zach Fuller from Eduardo Amorim (Amorim et al. 2017) and Yuval Simons (Simons et al. 2014)
//
// ===========================================================================
// MULTI-DEMOGRAPHY VERSION of AutosomeSimulate_smallback.cpp
//
// Identical to AutosomeSimulate_smallback.cpp except that the demographic
// history is now selectable at runtime via an extra (optional) trailing
// argument:
//
//     ... <num_generations> [eur|afr|sas]    (default: eur)
//
//   eur : Schiffels-Durbin + Schraiber non-Finnish European history
//         (63 epochs, taujump = 56000)  -- byte-for-byte the same arrays as
//         the original AutosomeSimulate_smallback.cpp
//   afr : African history (50 epochs, taujump = 68000)
//   sas : South Asian history (54 epochs, taujump = 72000)
//
// Everything that used to be a compile-time #define / hard-coded index and
// that depends on the demography is now a runtime variable:
//
//   * taujump          (was #define 56000)  -> per-population
//   * number of epochs (was hard-coded 67 in popsize()) -> NDEM-1
//   * dem_uncert index (was hard-coded 55)  -> per-population UNCERT_IDX
//
// IMPORTANT invariant, checked at startup: taujump MUST be > T[0], i.e. the
// whole "jump over long stretches of monomorphism" phase has to happen while
// the population is still at its constant ancestral size Ne[0].  The jump
// waiting times are drawn as Exp(2*Ne[0]*mutU), which is only correct in a
// constant-size population.  eur: 56000 > 55940. afr: 68000 > 67220.
// sas: 72000 > 71728.
//
// ---------------------------------------------------------------------------
// BACK-MUTATION-FREE VARIANT (ratio = 0).
//
// `ratio` is the back-mutation rate as a multiple of the forward rate, and it
// is a COMPILE-TIME constant, so turning back mutation off needs its own
// binary. This file is AutosomeSimulate_smallback_multidem.cpp with ratio set
// to 0 and the three places that divide by, or condition on, a non-zero back
// rate guarded -- the same three guards Sara already applied to the
// single-demography AutosomeSimulate_smallback_back_mu_0.cpp:
//
//   1. basedownVAR = 1/(ratio*halfthetaVAR) is 1/0 = inf when ratio = 0, and
//      the int(-log(u)*inf) that consumes it is undefined behaviour.
//   2. In branch 3 (derived allele fixed, gen > taujump) there is no back
//      mutation to wait for, so there is no exponential jump to draw: the
//      allele simply stays fixed and time is credited down to taujump+1.
//   3. poicond1(0) does NOT return 0 -- it is Poisson CONDITIONED ON >= 1, so
//      at rate 0 it would still force a back mutation. The two calls that
//      would see rate 0 are guarded instead.
//
// boost_poi(0) does return 0 (it special-cases l == 0), so the segregating
// branch's mutatedown needs no guard.
//
// for compiling:
// g++ -O3 -std=c++11 BRand.cpp population.cpp AutosomeSimulate_smallback_multidem_back_mu_0.cpp -o single_site_simulator_no_back_mut
// ===========================================================================

#include <chrono>
#include <random>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "population.h"
#include "BRand.hpp"
#include <stdint.h>
#include <map>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
using std::setprecision;
#include <cstdlib>
using std::atoi;
using std::atof;
#include <ctime>
using std::time;
#include <vector>
using std::vector;
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
using boost::poisson_distribution;
#include <boost/random/variate_generator.hpp>
using boost::variate_generator;

#define SPLIT     1000
#define Nzero     10000
#define tau       2040
//#define SEED    2710
#define TAU       100

// Back-mutation rate as a multiple of the forward rate, fixed at compile time.
// Build with -DBACK_MUT_RATIO=1 for the baseline runs and -DBACK_MUT_RATIO=0
// for the back-mutation-free runs; the three guards further down are exact
// no-ops when the ratio is > 0, so one source serves both. See README.md.
//
// The macro is spelled BACK_MUT_RATIO on the command line and only aliased to
// `ratio` here, AFTER the includes: `ratio` is also the name of a std template
// (std::ratio, used by <chrono>), so defining it via -D would be seen by the
// standard headers and break the build.
#ifndef BACK_MUT_RATIO
#define BACK_MUT_RATIO 1
#endif
#define ratio BACK_MUT_RATIO

boost::mt19937 gent;

double lognormal();
double randnum();

int nfinal, RUNS, Ntau = 14448;
int randflag = 0;

using namespace std;

struct freqs
{
  double freq0;
  double freq1;

  freqs(const double a = 0, const double b = 0) :
    freq0(a), freq1(b) {}
};

class valueComp
{
public:
  bool operator()(const freqs& A,
                  const freqs& B) const
  {
    if (A.freq0 != B.freq0)
      return A.freq0 < B.freq0;
    else
      return A.freq1 < B.freq1;
  }
};

char* filename(char *);
char* seriesfilename(char * str);
int popsize(int demographic_model, int dem_uncert, int gen, int i);
int poi(double l);
int poicond1(double l);
int boost_poi(double l);
int boost_poicond1(double l);

int popNe;
double sel, dom, mutU, Uvar;
int idx = 0;
int site_class;
int exp_lof_num;
double mut_rate, expec_lof, obs_num;
int initgen;
int demographic_model;
int mut_uncert;
int dem_uncert;
int last_pop_size, pop_size;
int pid_seed;

double lognormal(double mutU);

//

void Print(const vector<int>& v);

void Print(const vector<int>& v)
{
  //vector<int> v;
  for (int i = 0; i < (int)v.size(); i++) {
    cout << v[i] << endl;
  }
}

// ===========================================================================
// Demographic histories.  The simulation runs FORWARD in time.  The history is
// tabulated in generations before the present, so the counter `gen` starts in
// the past and decreases to 0, which is the present.
// Convention (unchanged from the original): epoch i has size Ne[i] and covers
// generations T[i] < gen <= T[i-1].  Ne[0] is the ancestral size, used for
// every gen > T[0].
// ===========================================================================

// --- Non-Finnish European: Schiffels & Durbin, then Schraiber et al. -------
// (identical values to the original AutosomeSimulate_smallback.cpp)
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

// --- Active demography, selected at runtime -------------------------------
const int *Ne = Ne_eur;
const int *T  = T_eur;
int NDEM      = (int)(sizeof(Ne_eur) / sizeof(Ne_eur[0]));   // number of epochs
int taujump   = 56000;   // per-population; must be > T[0]
int UNCERT_IDX = 55;     // epoch at which the dem_uncert draw kicks in
string POPLABEL = "eur";

// Pick the demography.  Returns false on an unrecognised label.
bool select_demography(const string& pop)
{
  string p;
  for (size_t i = 0; i < pop.size(); ++i) p += (char)tolower((unsigned char)pop[i]);

  if (p == "eur" || p == "nfe" || p == "eu")
  {
    Ne = Ne_eur;
    T  = T_eur;
    NDEM = (int)(sizeof(Ne_eur) / sizeof(Ne_eur[0]));
    taujump = 56000;
    // idx 55 -> Ne = 721018 at T = 113: where the original code started
    // drawing the uncertain recent size from U(log10 6e5, log10 6e6).
    UNCERT_IDX = 55;
    POPLABEL = "eur";
  }
  else if (p == "afr" || p == "af")
  {
    Ne = Ne_afr;
    T  = T_afr;
    NDEM = (int)(sizeof(Ne_afr) / sizeof(Ne_afr[0]));
    taujump = 68000;
    // idx 48 -> Ne = 271461 at T = 127: the analogous point at which the
    // recent explosion starts.  Only ever used when dem_uncert == 1.
    UNCERT_IDX = 48;
    POPLABEL = "afr";
  }
  else if (p == "sas" || p == "sa")
  {
    Ne = Ne_sas;
    T  = T_sas;
    NDEM = (int)(sizeof(Ne_sas) / sizeof(Ne_sas[0]));
    taujump = 72000;
    // idx 52 -> Ne = 337083 at T = 87: the analogous point at which the
    // recent explosion starts.  Only ever used when dem_uncert == 1.
    UNCERT_IDX = 52;
    POPLABEL = "sas";
  }
  else
  {
    return false;
  }

  // Sanity checks --------------------------------------------------------
  if ((int)(sizeof(Ne_eur)/sizeof(Ne_eur[0])) != (int)(sizeof(T_eur)/sizeof(T_eur[0])))
  {
    cerr << "FATAL: Ne_eur and T_eur have different lengths\n";
    exit(1);
  }
  if ((int)(sizeof(Ne_afr)/sizeof(Ne_afr[0])) != (int)(sizeof(T_afr)/sizeof(T_afr[0])))
  {
    cerr << "FATAL: Ne_afr and T_afr have different lengths\n";
    exit(1);
  }
  if ((int)(sizeof(Ne_sas)/sizeof(Ne_sas[0])) != (int)(sizeof(T_sas)/sizeof(T_sas[0])))
  {
    cerr << "FATAL: Ne_sas and T_sas have different lengths\n";
    exit(1);
  }
  // The jumping phase assumes a constant ancestral size, so taujump has to
  // sit strictly above the oldest demographic breakpoint.
  if (taujump <= T[0])
  {
    cerr << "FATAL: taujump (" << taujump << ") must be > T[0] (" << T[0]
         << ") for population '" << POPLABEL << "'\n";
    exit(1);
  }
  // T must be strictly decreasing and end at 0 for popsize() to be correct.
  for (int i = 1; i < NDEM; ++i)
  {
    if (T[i] >= T[i-1])
    {
      cerr << "FATAL: T is not strictly decreasing at index " << i
           << " (" << T[i-1] << " -> " << T[i] << ")\n";
      exit(1);
    }
  }
  if (T[NDEM-1] != 0)
  {
    cerr << "FATAL: last entry of T must be 0 (got " << T[NDEM-1] << ")\n";
    exit(1);
  }
  if (UNCERT_IDX >= NDEM) UNCERT_IDX = NDEM - 1;

  return true;
}

// Self-check: step through the demography one generation at a time, from the
// oldest epoch to the present, using the real popsize(), and report every epoch
// actually visited.  Lets you verify the arrays and the idx bookkeeping without
// running a simulation.
//   ./single_site_simulator --dump-demography [eur|afr|sas]
int dump_demography(const string& pop)
{
    if (!select_demography(pop))
    {
        cerr << "FATAL: unknown population '" << pop << "'\n";
        return 1;
    }

    demographic_model = 1;
    dem_uncert = 0;
    idx = 0;

    cout << "# population=" << POPLABEL << " epochs=" << NDEM
         << " taujump=" << taujump << "\n";
    cout << "# taujump > T[0]? " << taujump << " > " << T[0] << " -> "
         << ((taujump > T[0]) ? "OK" : "FAIL") << "\n";
    cout << "epoch\tgen_hi\tgen_lo\tNe_expected\tNe_from_popsize\tstatus\n";

    // Burn-in / ancestral epoch: every gen > T[0] must give Ne[0].
    int got = popsize(demographic_model, dem_uncert, 250000, 0);
    cout << "0(anc)\tinf\t" << (T[0] + 1) << "\t" << Ne[0] << "\t" << got
         << "\t" << ((got == Ne[0] && idx == 0) ? "OK" : "FAIL") << "\n";

    int failures = (got == Ne[0] && idx == 0) ? 0 : 1;

    // Now step down one generation at a time and record each epoch visited.
    // Invariants checked for every epoch i:
    //   * idx advances by exactly 1 (no epoch is skipped or visited twice)
    //   * the transition into epoch i happens exactly at gen == T[i-1]
    //   * popsize() returns Ne[i] there
    int prev_idx = 0;
    int epochs_seen = 0;
    for (int gen = T[0]; gen >= 1; --gen)
    {
        int size = popsize(demographic_model, dem_uncert, gen, 0);
        if (idx != prev_idx)
        {
            bool ok = true;
            if (idx != prev_idx + 1) {
                cout << "# SKIPPED EPOCHS between " << prev_idx << " and " << idx << "\n";
                ok = false;
            }
            if (gen != T[idx - 1]) {
                cout << "# epoch " << idx << " started at gen " << gen
                     << " but T[" << (idx - 1) << "]=" << T[idx - 1] << "\n";
                ok = false;
            }
            if (size != Ne[idx]) ok = false;

            int hi = T[idx - 1];
            int lo = T[idx] + 1;   // T[NDEM-1]==0, so the last epoch ends at gen 1
            cout << idx << "\t" << hi << "\t" << lo << "\t" << Ne[idx]
                 << "\t" << size << "\t" << (ok ? "OK" : "FAIL") << "\n";
            if (!ok) failures++;
            prev_idx = idx;
            epochs_seen++;
        }
    }

    if (epochs_seen != NDEM - 1)
    {
        cout << "# visited " << epochs_seen << " epoch transitions, expected "
             << (NDEM - 1) << "\n";
        failures++;
    }

    // Final epoch reached at gen == 1 must be the present-day size.
    int last = popsize(demographic_model, dem_uncert, 1, 0);
    bool ok_last = (idx == NDEM - 1) && (last == Ne[NDEM - 1]);
    cout << "# gen=1 -> idx=" << idx << " Ne=" << last
         << " (expected idx=" << (NDEM - 1) << " Ne=" << Ne[NDEM - 1] << ") "
         << (ok_last ? "OK" : "FAIL") << "\n";
    if (!ok_last) failures++;

    cout << "# " << (failures == 0 ? "ALL CHECKS PASSED" : "FAILURES: ")
         << (failures == 0 ? "" : std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}

int main(int argc, char *argv[])
{
    //cout << last_pop_size << " " << Ne[55] << "\n";
    int initgen = 250000, stopover;
    long long num_generations = 0;  // declare here so it's visible globally
    string pop_arg = "eur";         // default keeps the original behaviour

    if (argc >= 2 && string(argv[1]) == "--dump-demography")
        return dump_demography(argc >= 3 ? argv[2] : "eur");

    //double m=0.00015;
    if (argc == 11 || argc == 12) // "./a.out Uvar sel DOM RUNS mut_uncert dem_uncert dem_model Ne seed num_gens [eur|afr]"
    {
      Uvar            = atof(argv[1]);
      //length=atof(argv[2]);
      sel             = atof(argv[2]);
      dom             = atof(argv[3]);
      RUNS            = atof(argv[4]);
      mut_uncert      = atof(argv[5]);
      dem_uncert      = atof(argv[6]);
      //Input 1 if Schiffles-Durbin demographic model should be used. Else, constant size is used
      demographic_model = atof(argv[7]);
      //Only useful if constant size population model is used
      popNe           = atof(argv[8]);
      pid_seed        = atof(argv[9]);
      num_generations = (long long)atof(argv[10]); // now this sets the global variable
      if (argc == 12) pop_arg = argv[11];
    }
    else // User may run the script with "./a.out" and the above mentioned parameters will be asked in the command line prompt.
    {
        cout << "Error: Not enough parameters enetered" << endl;
        cout << "Usage: " << argv[0]
             << " mutU sel DOM RUNS mut_uncert dem_uncert demographic_model"
                " Ne pid_seed num_generations [eur|afr|sas]" << endl;
        cout << "  [eur|afr|sas] selects the demographic history (default eur)."
             << endl;
        return 1;
    }

    if (!select_demography(pop_arg))
    {
        cerr << "FATAL: unknown population '" << pop_arg
             << "' (expected 'eur', 'afr' or 'sas')\n";
        return 1;
    }

    // Echo the demography on stderr so it lands in the SLURM .err file and
    // never contaminates the .out file that the summarise scripts parse.
    cerr << "# demography=" << POPLABEL
         << " epochs=" << NDEM
         << " Ne_ancestral=" << Ne[0]
         << " T_oldest=" << T[0]
         << " Ne_present=" << Ne[NDEM-1]
         << " taujump=" << taujump
         << " demographic_model=" << demographic_model
         << endl;

    gent.seed(time(NULL) + pid_seed);
    BRand::Controller.seed(time(NULL) + pid_seed);

    stopover = RUNS;

    std::uniform_real_distribution<double> mt_rand{0.0, 1.0};

    population::initialize(sel, dom);
    population* pops[2];
    // Query at a generation older than the first breakpoint so idx stays at 0
    // (the original called popsize(...,0,...) here, which walked idx to the end
    // of the array; harmless because idx is reset per run, but confusing).
    pops[0] = new population(popsize(demographic_model, dem_uncert, initgen, 0));
    pops[1] = new population(popsize(demographic_model, dem_uncert, initgen, 0));
    idx = 0;

    int euroflag = 0;
    //double halftheta=2*popsize(demographic_model,dem_uncert,initgen,0)*mutU;
    //double p1=1./(1.+ratio*exp((2*popsize(demographic_model,dem_uncert,initgen,0)-1)*sel)*(1-(2*dom-1)*(2*popsize(demographic_model,dem_uncert,initgen,0)-1)*sel*sel/(12*popsize(demographic_model,dem_uncert,initgen,0))) );
    //cout<<"P1:"<<halftheta<<",popsize:"<<popsize(initgen,0)<<"\n";
    //map<int,int> count[2],counts[2][2],countd[2],taucount,taucountd;
    //map<freqs,int,valueComp> joint,joints[2],jointd;

    int gen, thispop;
    //double baseup=(1./halftheta);
    //double basedown=(1./(ratio*halftheta));
    //double oneovertwoU=(1./(2*mutU));
    //double oneovertwoUratio=(1./(2*mutU*ratio));
    //int initialstate,zeroruns=0,oneruns=0;
    //int skip=0,stretch;
    time_t tt;
    struct tm *tim;

    double totfreq;
    int age = 0;
    int tauallele, before, after;
    freqs f;
    ofstream myfile, results, afrDistrib, eurDistrib, mutDistribution;

    for (int run = 0; run < RUNS; run++)
    {
        int gens_seg   = 0;
        int gens_fixed = 0;
        int gens_lost  = 0;

        if (mut_uncert == 1)
        {
          std::normal_distribution<double> norm_distribution(Uvar, Uvar / 10);
          mutU = norm_distribution(gent);
        }
        else
        {
          mutU = Uvar;
        }

        randflag = 0;
        if (demographic_model == 1)
        {
          //Burn-in period, only used under Schiffels-Durbin model
          initgen = 250000;
        }
        else
        {
          //Burn-in period, only used under constant population size model
          initgen = num_generations;
        }

        gen = initgen;
        idx = 0;  // FIX #3: reset demographic index at start of each run
        double halfthetaVAR = 2 * popsize(demographic_model, dem_uncert, initgen, 0) * mutU;
        double baseupVAR    = (1.0 / halfthetaVAR);
        double basedownVAR  = (ratio > 0 ? 1.0 / (ratio * halfthetaVAR) : 0.0);

        pops[0]->size = popsize(demographic_model, dem_uncert, gen, 0); //Initializing the population size for the beginning of each run with deleterious allele absent from the population
        pops[0]->clear();

        euroflag = 0; //A flag indicating if the African-European population split has occured (0 = it hasn't occurred)

        while (gen > 1)
        {
            if (euroflag == 1) {
                totfreq = 0.5 * (pops[0]->freq() + pops[1]->freq());
                if (totfreq == 1) {
                    pops[0]->clear();
                    pops[1]->clear();
                    totfreq = 0.0;
                }
            }

            int i = 0;
            int allele_count = pops[i]->allelenum();
            int ploidy       = 2 * pops[i]->size;

            // 1) Segregating, or recent history
            if ((allele_count > 0 && allele_count < ploidy) || (gen <= taujump))
            {
                gens_seg++;
                pops[i]->mutateup(   boost_poi(mutU * (ploidy - allele_count)) );
                pops[i]->mutatedown( boost_poi(mutU * ratio * allele_count) );
            }
            // 2) Ancestral allele fixed
            else if (allele_count == 0)
            {
                if (gen > taujump)
                {
                    int jump = int(-log(mt_rand(gent)) * baseupVAR);
                    if (gen - jump > taujump) {
                        gens_lost += jump;
                        gen       -= jump;
                        pops[i]->size = popsize(demographic_model, dem_uncert, gen, 0);
                        pops[i]->clear();
                        pops[i]->mutateup(poicond1(2 * mutU * pops[i]->size));
                        continue; // skip the gen-- at the bottom
                    } else {
                        // credit remainder down to taujump+1
                        int remainder = gen - (taujump + 1);
                        gens_lost += remainder;
                        gen = taujump + 1;
                        // no mutation here; next loop iteration will handle per-gen
                    }
                }
                else
                {
                    gens_lost++;
                    pops[i]->clear();
                    pops[i]->mutateup(poicond1(2 * mutU * pops[i]->size));
                }
            }
            // 3) Derived allele fixed
            else /* allele_count == ploidy */
            {
                if (gen > taujump)
                {
                    if (ratio == 0) {
                        // no back mutation: once fixed, fixed for good. Credit the
                        // whole stretch down to taujump+1 rather than drawing a
                        // waiting time for an event that cannot happen.
                        int remainder = gen - (taujump + 1);
                        gens_fixed += remainder;
                        gen = taujump + 1;
                    } else {
                        int jump = int(-log(mt_rand(gent)) * basedownVAR);
                        if (gen - jump > taujump) {
                            gens_fixed += jump;
                            gen        -= jump;
                            pops[i]->size = popsize(demographic_model, dem_uncert, gen, 0);
                            pops[i]->fix();
                            pops[i]->mutatedown(poicond1(2 * mutU * ratio * pops[i]->size));
                            continue; // skip the gen-- at the bottom
                        } else {
                            int remainder = gen - (taujump + 1);
                            gens_fixed += remainder;
                            gen = taujump + 1;
                        }
                    }
                }
                else
                {
                    gens_fixed++;
                    pops[i]->fix();
                    // poicond1 is conditioned on >= 1, so at rate 0 it would still
                    // force a back mutation. Skip it instead.
                    if (ratio > 0) {
                        pops[i]->mutatedown(poicond1(2 * mutU * ratio * pops[i]->size));
                    }
                }
            }

            // 4) Reproduce / advance one generation
            pops[i]->populate_from(
                pops[i]->prob(),
                popsize(demographic_model, dem_uncert, gen - 1, 0)
            );
            gen--;
        }

        cout << "Gen: " << gen
             << " (Segregating: " << gens_seg
             << ", Fixed: "      << gens_fixed
             << ", Lost: "       << gens_lost << " generations) ";
        for (int j = 2; j >= 0; j--)
          cout << pops[0]->alleleholders[j] << " ";

        cout << mutU << "\n";
    }
}

int popsize(int demographic_model, int dem_uncert, int gen, int i) //Calculates the size of population at generation gen according to the selected demographic model
{
  if (demographic_model == 1)
  {
    // FIX #2: robust advancement across multiple boundaries.  NDEM is the
    // number of epochs of whichever demography was selected (63 for eur,
    // 50 for afr) -- this used to be hard-coded as 67.
    while (idx < NDEM - 1 && gen <= T[idx]) idx += 1;

    if (dem_uncert == 0)
    {
      return Ne[idx];
    }
    else
    {
      pop_size = Ne[idx];
      if ((idx == UNCERT_IDX) && (randflag == 0))
      {
        // FIX #1: remove RNG reseeding here; just draw from the existing global gent
        std::uniform_real_distribution<double> distribution(log10(600000), log10(6000000));
        last_pop_size = pow(10, distribution(gent));
        pop_size = last_pop_size;

        randflag = 1;
      }
      else if ((idx >= UNCERT_IDX) && (randflag == 1))
      {
        pop_size = last_pop_size;
      }

      return pop_size;
    }
  }
  else
  {
    return popNe;
  }
}

double randnum(double a, double b)
{
  // FIX #1: remove internal reseeding; use global gent
  std::uniform_real_distribution<double> distribution(a, b);
  return distribution(gent);
}

double lognormal(double mutU)
{
  mutU = log10(mutU) - (0.3249 / 2) * log(10);
  // FIX #1: remove internal reseeding; use global gent
  // The following lines have to be uncommented or commented prior to compilation according to the type of mutation that is going to be simulated.
  // There are 4 types (CpGti, CpGtv, nonCpGti and nonCpGtv) and the genomic average, based on Kong et al. (2012) - See Methods for detail
  // We also apply the correction from Harpak et al. (2016) - See Methods for detail.

  //std::normal_distribution<double> distribution(-7.324836926,0.57); //CpGti
  //std::normal_distribution<double> distribution(-8.392236341,0.57); //CpGtv
  //std::normal_distribution<double> distribution(-8.583066473,0.57); //nonCpGti
  //std::normal_distribution<double> distribution(-8.798867103,0.57); //nonCpGtv
  //std::normal_distribution<double> distribution(-5.906195,0.57); //PRDM9
  std::normal_distribution<double> distribution(mutU, 0.57);

  double u = distribution(gent);
  return pow(10, u);
}

int boost_poi(double l) //Regular Poisson random variate
{
  if (l == 0.0)
  {
    return 0.0;
  }
  else
  {
    boost::random::poisson_distribution<> dist(l);
    return dist(gent);
  }
}

int boost_poicond1(double l) //Regular Poisson random variate
{
  if (l == 0.0)
  {
    return 1.0;
  }
  else
  {
    int res = 0;
    while (res < 1)
    {
      boost::random::poisson_distribution<> dist(l);
      res = dist(gent);
    }
    return res;
  }
}

int poicond1(double l) //Poisson random variate conditional on the result being at least 1
{
  double expl = exp(-l);
  double p    = expl + BRand::Controller.nextClosed() * (1 - expl);
  int k       = 1;
  while (p >= expl)
  {
    k++;
    p = p * BRand::Controller.nextClosed();
  }
  return k - 1;
}
