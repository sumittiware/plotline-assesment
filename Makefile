.DEFAULT_GOAL := help

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
UVICORN    := $(VENV)/bin/uvicorn
PYTEST     := $(VENV)/bin/pytest

API_HOST   ?= 0.0.0.0
API_PORT   ?= 8000

.PHONY: help setup venv db-rebuild ingest bootstrap run test eval-live \
        clean clean-index clean-pyc clean-venv clean-all

help: ## Show this list of targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv (.venv) if it doesn't exist yet
	@test -d $(VENV) || python3 -m venv $(VENV)

setup: venv ## Fetch dependencies (venv + pip install -r requirements.txt)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example -- fill in GOOGLE_API_KEY.")

db-rebuild: ## Rebuild user_activity_summary/user_feature_adoption from raw events in data.sqlite
	$(PYTHON) -m src.data_access.db

ingest: ## Chunk + embed /guidelines into the FAISS index (data/guidelines_index)
	$(PYTHON) -m src.rag.ingest

bootstrap: setup db-rebuild ingest ## First-time setup: deps + derived tables + RAG index, all in one shot

run: ## Start the FastAPI server (uvicorn, auto-reload) on $(API_HOST):$(API_PORT)
	$(UVICORN) src.main:app --host $(API_HOST) --port $(API_PORT) --reload

test: ## Run the full test suite (deterministic only -- no API key needed)
	$(PYTHON) -m pytest tests/ -v

eval-live: ## Run the golden-set eval fixtures against the REAL Gemini API (costs quota, needs GOOGLE_API_KEY)
	RUN_LIVE_EVAL=1 $(PYTHON) -m pytest tests/test_eval_goldenset.py -v -k live

clean-index: ## Delete the persisted RAG/FAISS index (forces a rebuild on next `make ingest`)
	rm -rf data/guidelines_index

clean-pyc: ## Remove __pycache__ / .pytest_cache
	find . -type d -name '__pycache__' -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf .pytest_cache

clean: clean-index clean-pyc ## clean-index + clean-pyc (does not touch .venv or data.sqlite)

clean-venv: ## Remove the virtualenv
	rm -rf $(VENV)

clean-all: clean clean-venv ## clean + clean-venv (fully reset generated state)
