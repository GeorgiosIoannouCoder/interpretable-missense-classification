.PHONY: help install data preprocess features embeddings train eval external figures report supplementary reproduce clean test lint

PYTHON ?= python3
ACCELERATE ?= accelerate launch

help:
	@echo "Targets:"
	@echo "  install       Install pinned Python dependencies"
	@echo "  data          Phase 1: download ClinVar, UniProt, idmapping, AlphaMissense, CADD"
	@echo "  preprocess    Phases 2-4: filter ClinVar, map to UniProt, gene-disjoint splits"
	@echo "  features      Phase 5: extract handcrafted features"
	@echo "  embeddings    Phase 6: extract ESM-2 residue embeddings (resumable)"
	@echo "  train         Phases 7-8: train baselines and the ESM-2 MLP head"
	@echo "  eval          Phase 9: compute metrics + bootstrap CIs + efficiency table"
	@echo "  external      Phase 10: AlphaMissense and CADD comparison on test variants"
	@echo "  figures       Phase 11: generate all figures and tables"
	@echo "  report        Phase 12: build the final LaTeX report PDF"
	@echo "  supplementary Phase 12: build supplementary_team19_gi2100_vsj7589.zip"
	@echo "  reproduce     Run the full pipeline end-to-end"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff and black --check"
	@echo "  clean         Remove caches and generated artifacts (keeps data/raw)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu126
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/01_download.py

preprocess:
	$(PYTHON) scripts/02_preprocess_clinvar.py
	$(PYTHON) scripts/03_map_to_proteins.py
	$(PYTHON) scripts/04_make_splits.py

features:
	$(PYTHON) scripts/05_extract_features.py

embeddings:
	$(PYTHON) scripts/06_extract_esm2_embeddings.py

train:
	$(PYTHON) scripts/07_train_baseline.py
	$(ACCELERATE) scripts/08_train_esm2_head.py
	$(ACCELERATE) scripts/12_train_combined.py

eval:
	$(PYTHON) scripts/09_evaluate.py

external:
	$(PYTHON) scripts/10_compare_external.py

figures:
	$(PYTHON) scripts/11_make_figures.py

report:
	$(MAKE) -C reports/paper

supplementary:
	bash scripts/build_supplementary.sh

reproduce: install data preprocess features embeddings train eval external figures report supplementary

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

clean:
	rm -rf data/processed data/embeddings checkpoints runs logs reports/figures/cache
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
