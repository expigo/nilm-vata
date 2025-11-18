.PHONY: help install install-dev install-dl install-all test validate demo benchmark showcase clean format lint

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies using uv
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"

install-dl:  ## Install deep learning dependencies (PyTorch)
	uv pip install -e ".[deep-learning]"

install-datasets:  ## Install dataset dependencies
	uv pip install -e ".[datasets]"

install-nilmtk:  ## Install optional NILMTK dependencies
	uv pip install -e ".[nilmtk]"

install-all:  ## Install all optional dependencies
	uv pip install -e ".[all]"

sync:  ## Sync dependencies with uv
	uv pip sync

test:  ## Run the test algorithm script
	python test_algorithms.py

validate:  ## Run validation script
	python validate.py

demo:  ## Run real-time demo
	python demo_realtime.py

benchmark:  ## Run algorithm benchmark comparison
	python benchmark.py

showcase:  ## Showcase all available algorithms
	python showcase_algorithms.py

clean:  ## Clean generated files
	rm -rf *.png
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__ src/*/__pycache__
	rm -rf build dist *.egg-info
	rm -rf models/*.pkl models/*.joblib

format:  ## Format code with black
	black src/ *.py

lint:  ## Lint code with ruff
	ruff check src/ *.py

setup:  ## Initial setup with uv
	@echo "Setting up NILM VATA with uv..."
	uv pip install -e .
	@echo ""
	@echo "Setup complete! Run 'make validate' to test the installation."
