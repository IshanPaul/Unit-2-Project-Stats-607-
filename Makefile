# Makefile for Simulation Study Pipeline

PY=python3
VENV=.venv
PYBIN=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Detect number of CPU cores (macOS or Linux)
NCORES=$(shell (sysctl -n hw.ncpu 2>/dev/null || nproc))

# Default target
all: simulate analyze figures

# Environment setup
venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run full simulation (serial)
simulate:
	$(PYBIN) -m src.simulation --mode large --save

# Run full simulation (parallel)
large:
	$(PYBIN) -m src.simulation --mode large --save --n_jobs $(NCORES)

# Analyze results
analyze:
	$(PYBIN) -m src.analyze

# Generate visualizations
figures:
	$(PYBIN) -m src.figures

# Run tests
test:
	$(PYBIN) -m tests.test_basic

# Clean all generated files
clean:
	rm -rf $(VENV) results/raw/*.csv results/figures/*.png __pycache__ .pytest_cache

.PHONY: all simulate large analyze figures test clean
