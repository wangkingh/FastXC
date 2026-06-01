# Unified developer build entry for FastXC.
#
# Typical WSL/Linux workflow:
#   make install        # build native backends, stage them into the Python package, install fastxc
#   make doctor-source  # run doctor directly from the source tree

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
BUILD ?= release
MODE ?= par

NATIVE_DIR := native

.PHONY: all install develop python-dev python-dev-no-deps \
	native native-full \
	stage-binaries doctor doctor-source clean-native veryclean-native help

all: install ## Build native binaries and install Python console commands

install: native stage-binaries python-dev ## Full editable install: fastxc command + all native binaries

develop: install ## Alias for install

python-dev: ## Install Python package in editable mode, creating the fastxc command
	$(PIP) install -e .

python-dev-no-deps: ## Editable install without dependency resolution
	$(PIP) install -e . --no-deps

native: ## Build all native backends: sac2spec, xc_fast, ncf_pws, ncf_tfpws
	$(MAKE) -C "$(NATIVE_DIR)" BUILD=$(BUILD) MODE=$(MODE) all

native-full: native ## Alias for native; kept for explicit full-build workflows

stage-binaries: ## Copy bin native binaries into fastxc/bin/<platform> for packaging
	$(PYTHON) -m fastxc.devtools.stage_binaries

doctor-source: ## Run doctor from the source tree without requiring installation
	$(PYTHON) -m fastxc.cli doctor

doctor: ## Run doctor through the installed fastxc command
	fastxc doctor

clean-native: ## Remove native object files and generated test fixtures
	$(MAKE) -C "$(NATIVE_DIR)" clean

veryclean-native: ## Remove native object files, fixtures, and built native binaries
	$(MAKE) -C "$(NATIVE_DIR)" veryclean

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m [BUILD=release|debug] [MODE=par|seq]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
