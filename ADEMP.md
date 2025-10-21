# ======================================================
# Makefile — Full workflow: Lasso simulation + analysis + plots + lambda search + tests
# ======================================================

PYTHON = python
SRC = src
RESULTS = results
RAW = $(RESULTS)/raw
FIGURES = $(RESULTS)/figures
TESTS = tests
SCRIPTS = scripts

# ------------------------------------------------------
# Default target: install → simulate → analyze → figures → lambda search → test
# ------------------------------------------------------
all: install simulate analyze figures lambda_search test

# ------------------------------------------------------
# Step 0: Install dependencies
# ------------------------------------------------------
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# ------------------------------------------------------
# Step 1: Run simulation pipeline
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
# Step 3: Generate plots and figures
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
# Step 5: Run unit tests
# ------------------------------------------------------
test:
	@echo "🧪 Running tests..."
	pytest -q $(TESTS)

# ------------------------------------------------------
# Step 6: Clean generated results
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

.PHONY: all install simulate analyze figures lambda_search test clean check
Unit-2-Project-Stats-607-/
├── Makefile
├── README.md
├── ADEMP.md
├── requirements.txt
├── src/
│   ├── dgps.py
│   ├── methods.py
│   ├── simulation.py
│   ├── figures.py
│   ├── __init__.py
│
├── scripts/
│   └── focused_search_lam.py
│
├── tests/
│   └── test_basic.py
│
├── results/
│   ├── raw/
│   ├── figures/
│   └── analysis/
