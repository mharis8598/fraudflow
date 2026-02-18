.PHONY: install train test lint serve dashboard docker-up docker-down clean help

# ─── Setup ────────────────────────────────────────────
install:  ## Install all dependencies
	pip install -r requirements.txt

# ─── ML Pipeline ──────────────────────────────────────
train:  ## Train the model with MLflow tracking
	python -m src.train

# ─── Testing ──────────────────────────────────────────
test:  ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

lint:  ## Lint code with Ruff
	ruff check src/ api/ tests/

# ─── Serving ──────────────────────────────────────────
serve:  ## Start FastAPI server locally
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

mlflow-ui:  ## Start MLflow tracking UI
	mlflow ui --port 5000

# ─── Docker ───────────────────────────────────────────
docker-up:  ## Start all services with Docker Compose
	docker-compose up --build -d

docker-down:  ## Stop all Docker services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

# ─── Utilities ────────────────────────────────────────
clean:  ## Remove artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache mlruns/
	rm -rf src/__pycache__ api/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
