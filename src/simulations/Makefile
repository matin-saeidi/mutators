# Forward simulators for mutator-allele frequency distributions.
#
#   make            build all four binaries into ./bin
#   make clean      remove them
#
# The two single-site binaries are built from ONE source, single_site_simulator.cpp,
# differing only in -DBACK_MUT_RATIO (the back-mutation rate as a multiple of the forward
# rate). See README.md for why that is exactly equivalent to the two separate
# sources these runs originally used.

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++11
BIN       = bin
COMMON    = BRand.cpp population.cpp

.PHONY: all clean
all: $(BIN)/single_site_simulator \
     $(BIN)/single_site_simulator_no_back_mut \
     $(BIN)/comp_het_lineage_tracker \
     $(BIN)/comp_het_lineage_tracker_constN

$(BIN):
	mkdir -p $(BIN)

# Baseline: back-mutation rate equal to the forward rate.
$(BIN)/single_site_simulator: single_site_simulator.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) -DBACK_MUT_RATIO=1 $(COMMON) $< -o $@

# Back mutation switched off, for the back-mutation-on/off comparison.
$(BIN)/single_site_simulator_no_back_mut: single_site_simulator.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) -DBACK_MUT_RATIO=0 $(COMMON) $< -o $@

# Compound-het genes (XPC, MUTYH) under a demographic history.
$(BIN)/comp_het_lineage_tracker: comp_het_lineage_tracker.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON) $< -o $@

# Compound-het genes at constant Ne, for the XPC mutation-rate comparison.
$(BIN)/comp_het_lineage_tracker_constN: comp_het_lineage_tracker_constN.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(COMMON) $< -o $@

clean:
	rm -rf $(BIN)
