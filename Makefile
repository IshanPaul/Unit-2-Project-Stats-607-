# Corrected Makefile for Unit 2/3 Simulation Study
# High-Performance Lasso Support Recovery Simulation

.PHONY: help venv install clean simulate analyze figures all \
        profile complexity benchmark parallel stability-check \
        test test-regression baseline-run optimized-run compare \
        docs view-profile

# Default target
.DEFAULT_GOAL := help

##@ Setup Targets

venv: ## Create virtual environment
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed"

##@ Unit 2 Targets (Baseline)

simulate: ## Run baseline simulation (serial)
	@echo "Running baseline simulation..."
	time python3 -m src.simulation --mode large --n_jobs 1 --save
	@echo "Baseline simulation complete"

analyze: ## Run analysis on results
	@echo "Analyzing results..."
	python3 -m src.analyze
	@echo "Analysis complete"

figures: ## Generate all figures
	@echo "Generating figures..."
	python3 -m src.figures
	@echo "Figures saved to results/figures/"

all: simulate analyze figures ## Run complete pipeline
	@echo "Complete pipeline finished"

##@ Unit 3 Targets (Profiling & Optimization)touc

view-baseline-profile: ## View profiling results
	@if [ -f profile.prof ]; then \
		echo "Opening profile visualization..."; \
		python3 -m snakeviz baseline.prof; \
	else \
		echo "No profile found. Run 'make profile' first"; \
	fi

view-optimized-profile: ## View profiling results
	@if [ -f profile2.prof ]; then \
		echo "Opening profile visualization..."; \
		python3 -m snakeviz profile2.prof; \
	else \
		echo "No profile found. Run 'make profile' first"; \
	fi


complexity: ## Analyze computational complexity (timing vs n)
	@echo "Running complexity analysis..."
	python3 scripts/complexity_analysis.py
	@echo "Complexity plots saved to results/figures/complexity_*.png"

baseline: ## Baseline performance
	@echo "Running benchmark comparison..."
	@echo ""
	@echo "=== Baseline (parallel) ==="
	python3 -m cProfile -o baseline.prof -m scripts.simulation --mode large --n_jobs 10
	@echo ""
	@echo "Benchmark results saved to results/baseline.prof"

parallel: ## Run optimized version with full parallelization
	@echo "Running optimized parallel simulation..."
	@echo "Using all available cores"
	time python3 -m src.simulation --mode large --n_jobs 6 --batch_size 5 --save
	@echo "Parallel simulation complete"

stability-check: ## Check for numerical warnings and convergence issues
	@echo "Running stability checks across parameter space..."
	python3 scripts/stability_check.py
	@echo "Stability report saved to results/stability_report.txt"

##@ Testing Targets

test: ## Run all unit tests
	@echo "Running unit tests..."
	pytest tests/ -v

test-regression: ## Verify optimizations preserve correctness
	@echo "Running regression tests..."
	python3 tests/test_regression.py
	@echo ""
	@echo "✓ All regression tests passed"
	@echo "Optimizations preserve correctness"

##@ Utility Targets

baseline-run: ## Quick baseline run (small scale for testing)
	@echo "Running small baseline test..."
	python3 -m scripts.simulation --mode small --n_jobs 1

optimized-run: ## Quick optimized run (small scale for testing)
	@echo "Running small optimized test..."
	python3 -m src.simulation --mode small --n_jobs 4 --n_reps 100

compare: ## New results (fixed indentation & corrected cProfile call)
	@echo "Running optimized parallel simulation..."
	@echo "Using all available cores"
	python3 -m cProfile -o profile_new.prof -m src.simulation --mode large --n_jobs 6 --batch_size 5 --save
	@echo "Parallel simulation complete"

clean: ## Remove generated files and virtual environment
	rm -rf venv/
	rm -rf results/raw/*.csv
	rm -rf results/figures/*.png
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf .pytest_cache/
	rm -f profile.prof
	rm -f *.log
	@echo "Cleaned up generated files"

##@ Documentation Targets

docs: ## Generate documentation
	@echo "Documentation available in:"
	@echo "  - docs/BASELINE.md"
	@echo "  - docs/OPTIMIZATION.md"
	@echo "  - docs/ADEMP.md"
	@echo "  - README.md"


##@ Help

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
