.PHONY: help lint test format clean install coverage coverage-xml matrix-check matrix-update matrix-verify

PYTHON := python3
PYTHON_SRC := src

help:
	@echo "nvidia-inst - Cross-distribution Nvidia driver installer"
	@echo ""
	@echo "Available targets:"
	@echo "  lint            - Run ruff (Python linter)"
	@echo "  test            - Run pytest"
	@echo "  coverage        - Run pytest with coverage report"
	@echo "  coverage-xml    - Generate XML coverage report for CI"
	@echo "  format          - Run black (Python formatter)"
	@echo "  matrix-check    - Check matrix status"
	@echo "  matrix-update   - Update compatibility matrix"
	@echo "  matrix-verify   - Verify matrix data integrity"
	@echo "  clean           - Clean up cache and temporary files"
	@echo "  install         - Install package"

lint:
	@echo "Running Python linter..."
	@$(PYTHON) -m ruff check $(PYTHON_SRC)

test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v

coverage:
	@echo "Running tests with coverage..."
	@$(PYTHON) -m pytest tests/ -v --cov=nvidia_inst --cov-report=term-missing

coverage-xml:
	@echo "Generating XML coverage report..."
	@$(PYTHON) -m pytest tests/ --cov=nvidia_inst --cov-report=xml
	@echo "Report saved to coverage.xml"

format:
	@echo "Formatting Python code..."
	@$(PYTHON) -m black $(PYTHON_SRC)

clean:
	@echo "Cleaning up..."
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf ~/.cache/nvidia-inst/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

matrix-check:
	@echo "Checking compatibility matrix..."
	@$(PYTHON) scripts/update-matrix.py --check

matrix-update:
	@echo "Updating compatibility matrix..."
	@$(PYTHON) scripts/update-matrix.py

matrix-verify:
	@echo "Verifying compatibility matrix..."
	@$(PYTHON) scripts/update-matrix.py --verify

install:
	@echo "Installing nvidia-inst..."
	@pip install -e .
