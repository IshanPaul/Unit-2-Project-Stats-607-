# ======================================================
# Makefile — Full workflow: Lasso simulation + analysis + plots + lambda search + heatmaps + tests
# ======================================================

PYTHON = python
SRC = src
RESULTS = results
RAW = $(RESULTS)/raw
FIGURES = $(RESULTS)/figures
TESTS = tests
SCRIPTS = scripts

# ------------------------------------------------------
# Default target: install → simulate → analyze → figures → lambda_search → heatmaps → test
# ------------------------------------------------------
all: install simulate analyze figures lambda_search heatmaps test

# ------------------------------------------------------
# Step 0: Install dependencies
# ------------------------------------------------------
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# ------------------------------------------------------
# Step 1: Run main simulation
# ------------------------------------------------------
simulate:
	@echo "▶ Running main simulation pipeline..."
	$(PYTHON) $(SRC)/simulation.py --save $(RAW)

# ------------------------------------------------------
# Step 2: Aggregate and analyze results
# ------------------------------------------------------
analyze:
	@echo "📊 Aggregating and analyzing results..."
	$(PYTHON) $(SRC)/figures.py --analyze $(RAW) --save $(RESULTS)

# ------------------------------------------------------
# Step 3: Generate standard plots
# ------------------------------------------------------
figures:
	@echo "📈 Generating plots..."
	$(PYTHON) $(SRC)/figures.py --plot $(RESULTS)/summary.csv --out $(FIGURES)

# ------------------------------------------------------
# Step 4: Focused lambda-factor search
# ------------------------------------------------------
lambda_search:
	@echo "🔍 Running focused lambda-factor search..."
	$(PYTHON) $(SCRIPTS)/focused_search_lam.py

# ------------------------------------------------------
# Step 5: Heatmap sweeps (lambda vs sigma, lambda vs k)
# ------------------------------------------------------
heatmaps:
	@echo "🔥 Running heatmap sweeps (lambda, sigma) and (lambda, k)..."
	$(PYTHON) $(SCRIPTS)/heatmaps.py

# ------------------------------------------------------
# Step 6: Run unit tests
# ------------------------------------------------------
test:
	@echo "🧪 Running tests..."
	pytest -q $(TESTS)

# ------------------------------------------------------
# Step 7: Clean generated results
# ------------------------------------------------------
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf $(RAW)/*.csv $(RAW)/*.pkl \
	       $(RESULTS)/summary.csv \
	       $(FIGURES)/*.png \
	       $(RESULTS)/analysis/*.csv \
	       $(RESULTS)/analysis/*.png

# ------------------------------------------------------
# Utility: run only tests (quick check)
# ------------------------------------------------------
check:
	pytest -q $(TESTS)

.PHONY: all install simulate analyze figures lambda_search heatmaps test clean check
