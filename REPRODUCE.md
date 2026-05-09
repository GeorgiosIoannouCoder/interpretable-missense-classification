# Reproduction guide

This file accompanies the *Comparing Interpretable Sequence Features and Protein Language Models for Missense Variant Classification* final project (Spring 2026 AI in Genomics).

## 1. Environment

- Linux (tested on Ubuntu 22, kernel 6.8) with CUDA 12.
- 1 NVIDIA GPU recommended (the project was developed on a GH200 96 GB). The training and embedding code transparently uses multiple GPUs without code changes when launched with `accelerate launch --num_processes=N`.
- Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install torch from the CUDA 12.6 PyTorch index so it matches the GH200's
# 12.8 driver. The PyPI default (torch+cu130) requires a newer driver.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 2. Data

All data sources are public. The download script fetches them under `data/raw/` and `data/external/`:

- **ClinVar** `variant_summary.txt.gz` from NCBI's ClinVar FTP. We pin the release date used into `data/processed/clinvar_release.txt`.
- **UniProt** human reference proteome **UP000005640** FASTA (Reviewed / Swiss-Prot only is filtered downstream).
- **UniProt** RefSeq -> UniProt id-mapping (`HUMAN_9606_idmapping_selected.tab.gz`).
- **AlphaMissense** `AlphaMissense_hg38.tsv.gz` (~1.6 GB, all human missense scores).
- **CADD** `whole_genome_SNVs.tsv.gz` (~80 GB) and the small `.tbi` tabix index. The bulk file is downloaded once at the start of Phase 1 in the background; queries against it during Phase 10 are fast local tabix lookups.

```bash
make data        # Phase 1
```

The CADD download is the slowest (~hours depending on network). It runs in the background while Phases 2-9 execute.

## 3. Pipeline

```bash
make preprocess  # Phases 2-4: filter ClinVar, map to UniProt, gene-disjoint splits
make features    # Phase 5: handcrafted features
make embeddings  # Phase 6: ESM-2 (650M) residue embeddings (resumable, sharded)
make train       # Phases 7-8 + 13: baselines, ESM-2 MLP head, combined head
make eval        # Phase 9: metrics + bootstrap CIs + efficiency table
make external    # Phase 10: AlphaMissense + CADD comparison on test variants
make figures     # Figures + tables (incl. calibration appendix, RF ablation, RF vs ESM-2 disagreement summary)
make report      # Phase 12: LaTeX report PDF
make supplementary  # Phase 12: build supplementary ZIP
```

Or end-to-end:

```bash
make reproduce
```

## 4. Resumability (SSH-drop safe)

- **ESM-2 embedding extraction:** sharded into chunks of ~1000 proteins. Each completed shard is recorded in `data/embeddings/esm2_650M/manifest.json`. Re-running `make embeddings` skips completed shards.
- **ESM-2 / combined MLP heads:** `last.pt` checkpoints live under `checkpoints/esm2_head/` and `checkpoints/combined_head/` with the same resume cadence.
- **sklearn baselines:** persisted with `joblib`; deterministic so re-running is cheap.

## 5. Multi-GPU

Single-GPU is the default. To use N GPUs:

```bash
accelerate launch --num_processes=N scripts/08_train_esm2_head.py
WORLD_SIZE=N python3 scripts/06_extract_esm2_embeddings.py  # shards split across ranks
```

## 6. Submission artifacts

- `final_report_team19_gi2100_vsj7589.pdf` (built by `make report`)
- `supplementary_team19_gi2100_vsj7589.zip` (built by `make supplementary`)

Both are uploaded to the team's group on the Brightspace assignment.
