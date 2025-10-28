.PHONY: help install install-dev test lint format clean docker-build docker-run deploy-render

help:
	@echo "RAW Fusion Diffusion - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run all tests"
	@echo "  make lint             Run linting checks"
	@echo "  make format           Format code with black and isort"
	@echo "  make type-check       Run type checking with mypy"
	@echo ""
	@echo "Training:"
	@echo "  make train-quick      Quick training test"
	@echo "  make train-full       Full training pipeline"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-test      Test Docker build"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-render    Deploy to Render (requires render CLI)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean build artifacts"
	@echo "  make clean-all        Clean everything including data"

install:
	pip install --upgrade pip
	pip install -r requirements-full.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements-full.txt
	pip install -e .[dev]

install-minimal:
	pip install --upgrade pip
	pip install -r requirements-minimal.txt

test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

format:
	black . --line-length=100
	isort . --profile black

type-check:
	mypy . --ignore-missing-imports --no-strict-optional

train-quick:
	python quick_train.py

train-full:
	python scripts/train_orchestrator.py --config configs/training_config.yaml --data_dir ./data/train --output_dir ./outputs

train-vae:
	python training/train_diffusion.py --stage vae --config configs/training_config.yaml --data_path ./data/train

train-flow:
	python training/train_diffusion.py --stage optical_flow --config configs/training_config.yaml --data_path ./data/train

evaluate:
	python evaluation/comprehensive_eval.py --models ./outputs --data ./data/val

docker-build:
	docker build -t raw-fusion-diffusion:latest .

docker-run:
	docker run -p 8000:8000 --env MODEL_PATH=/models raw-fusion-diffusion:latest

docker-test:
	docker build -t raw-fusion-diffusion:test . && \
	docker run --rm raw-fusion-diffusion:test python -c "import torch; print(f'PyTorch: {torch.__version__}')"

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

serve-local:
	python quick_deploy.py --local

serve-api:
	uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

serve-ui:
	python web/gradio_interface.py --models ./outputs --port 7860

deploy-render:
	@echo "Deploying to Render..."
	@echo "1. Push code to GitHub"
	@echo "2. Connect repo to Render"
	@echo "3. Render will use render.yaml for configuration"
	@echo "See DEPLOYMENT.md for detailed instructions"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -rf .mypy_cache/

clean-outputs:
	rm -rf outputs/
	rm -rf checkpoints/

clean-all: clean clean-outputs
	rm -rf data/processed/
	rm -rf logs/

benchmark:
	python -m inference.realtime_pipeline --benchmark

export-coreml:
	python deployment/coreml_converter.py --checkpoint ./outputs/consistency_final.pt --output ./ios_models

export-onnx:
	python export/model_export.py --format onnx --checkpoint ./outputs/consistency_final.pt --output ./exports/

docs-build:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

ci-local:
	@echo "Running CI checks locally..."
	make lint
	make type-check
	make test-fast
	@echo "✓ All CI checks passed!"

setup-git-hooks:
	@echo "Setting up git hooks..."
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make format" >> .git/hooks/pre-commit
	@echo "make lint" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Git hooks installed"

.DEFAULT_GOAL := help
