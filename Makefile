# Convenience targets. See README.md for what each stage actually does.
#
#   make sims       build the four C++ simulators into src/simulations/bin
#   make figures    redraw every figure from the simulation outputs in results/
#   make models     regenerate data/demographic_models/ from the simulators
#   make clean      remove build products (never touches results/)

PYTHON ?= python3
SIMDIR  = src/simulations
FIGS    = $(wildcard figures/fig_*.py)

.PHONY: all sims figures models clean

all: sims

sims:
	$(MAKE) -C $(SIMDIR)

# Every figure script reads $MUTATORS_RESULTS (default ./results) and writes to
# $MUTATORS_FIGURES (default ./figures/output).
figures:
	@for f in $(FIGS); do \
	  echo "--- $$f"; \
	  MPLBACKEND=Agg $(PYTHON) $$f || exit 1; \
	done
	@echo "--- figures/fig_scenario_comparison.py --scenario backmut"
	@MPLBACKEND=Agg $(PYTHON) figures/fig_scenario_comparison.py --scenario backmut

# The demographic histories are compiled into the simulators; these files are a
# dump of them, not a separate source of truth. Regenerating them is how you pick
# up a change to the Ne/T arrays in src/simulations/.
models: sims
	@./tools/dump_demographic_models.sh

clean:
	$(MAKE) -C $(SIMDIR) clean
	rm -rf figures/output
