#!/usr/bin/env bash
#
# Phase 12: build the supplementary materials ZIP.
#
# Produces ``supplementary_team19_gi2100_vsj7589.zip`` at the repository
# root, containing the source code, scripts, configs, pinned requirements,
# reproduction guide, generated figures and tables, and the data download
# instructions. Raw and processed data files are *not* bundled (per the
# instructions, an external download path is acceptable when data is large).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="supplementary_team19_gi2100_vsj7589.zip"
rm -f "$OUT"

INCLUDE=(
  "src/imc"
  "scripts"
  "configs"
  "tests"
  "reports/figures"
  "reports/tables"
  "reports/paper/main.tex"
  "reports/paper/references.bib"
  "reports/paper/Makefile"
  "data/README.md"
  "Makefile"
  "README.md"
  "REPRODUCE.md"
  "requirements.txt"
  "pyproject.toml"
  "conftest.py"
  "LICENSE"
  ".gitignore"
)

zip -r "$OUT" "${INCLUDE[@]}" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.ipynb_checkpoints*" \
  -x "*.DS_Store" >/dev/null

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
