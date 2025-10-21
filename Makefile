# Makefile — Run entire Lasso simulation study

PYTHON = python
SRC = src
RESULTS = results
RAW = $(RESULTS)/raw
FIGURES = $(RESULTS)/figures

# Default target: run everything
all: simulate analyze figures

# Step 1: Run simulation
simulate:
	@echo "▶ Running simulation pipeline..."
	$(PYTHON) $(SRC)/simulation.py --save $(RAW)

# Step 2: Analyze aggregated results
analyze:
	@echo "▶ Aggregating and analyzing results..."
	$(PYTHON) $(SRC)/figures.py --analyze $(RAW) --save $(RESULTS)

# Step 3: Generate figures
figures:
	@echo "▶ Generating plots..."
	$(PYTHON) $(SRC)/figures.py --plot $(RESULTS)/summary.csv --out $(FIGURES)

# Step 4: Run tests
test:
	pytest -q

# Step 5: Clean up
clean:
	@echo "🧹 Cleaning up results..."
	rm -rf $(RAW)/*.csv $(RAW)/*.pkl $(RESULTS)/summary.csv $(FIGURES)/*.png

# Step 6: Install dependencies
install:
	pip install -r requirements.txt

.PHONY: all simulate analyze figures test clean install
