# interpretable-missense-classification

Comparing interpretable sequence features and protein language model embeddings for human missense variant pathogenicity classification.

## Project summary

Final project for the Spring 2026 *AI in Genomics* course.

We build binary classifiers that predict whether a human missense variant in ClinVar is pathogenic / likely pathogenic versus benign / likely benign. The goal is **not** to beat state-of-the-art predictors, but to study:

1. How much pathogenicity signal is captured by simple, interpretable handcrafted sequence features (logistic regression / random forest on reference and alternate amino acid identities, BLOSUM62 substitution score, local sequence-window composition, normalized residue position).
2. Whether residue-level embeddings from the pretrained protein language model **ESM-2 (650M parameters)** improve cross-gene generalization beyond those features.

We evaluate models with a **gene-held-out** split (entire genes are held out of training so test variants come from genes the model has never seen) using AUROC, AUPRC, F1, and confusion matrices. As reference points (not targets to beat) we compare against the established pathogenicity predictors **AlphaMissense** and **CADD** on the same held-out test variants.

Authors: Georgios Ioannou (gi2100), Vedant Jagtap (vsj7589).

## Repository layout

```
src/imc/        Python package
  data/         ClinVar / UniProt / RefSeq parsing, mapping, splits
  features/     Handcrafted features and ESM-2 embedding extraction
  models/       Baseline (LR / RF) and ESM-2 MLP head
  training/     Resumable training loops, checkpoint helpers
  eval/         Metrics, bootstrap CIs, external-tool comparison
  viz/          Plots and tables
  utils/        Logging, seeding, IO
scripts/        Numbered orchestration scripts (01_download.py ... 13_ablation_features.py)
configs/        YAML configs for data / baseline / esm2 runs
reports/
  figures/      Generated figures
  tables/       Generated tables (CSV)
  paper/        LaTeX source of the final report
tests/          Unit / smoke tests
data/           (gitignored) raw, processed, and embedding caches
checkpoints/    (gitignored) training checkpoints
```

## Hardware used

A single NVIDIA GH200 96 GB (CUDA 12, Ubuntu 22). The training and embedding-extraction code is written so that it transparently uses multiple GPUs without code changes when launched via `accelerate launch --num_processes=N`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install      # installs Torch from the CUDA 12.6 index, then requirements.txt

make data         # Phase 1: download ClinVar, UniProt, idmapping, AlphaMissense, CADD
make preprocess   # Phases 2-4: filter, map, split
make features     # Phase 5: handcrafted features
make embeddings   # Phase 6: ESM-2 residue embeddings (resumable)
make train        # Phases 7-8 + combined head: baselines + ESM-2 head + combined head
make eval         # Phase 9: metrics + bootstrap CIs + efficiency + test_predictions
make external     # Phase 10: AlphaMissense + CADD on test variants
make figures      # Phase 11: figures, tables, ablation, disagreement summary
make report       # Phase 12: build the LaTeX report PDF
make supplementary  # Phase 12: ZIP for submission (code + tables/figures + configs)

# end-to-end:
make reproduce
```

See [`REPRODUCE.md`](REPRODUCE.md) for the long-form reproduction guide.

## Submission artifacts

- Final report PDF: `reports/paper/final_report_team19_gi2100_vsj7589.pdf`
- Supplementary ZIP: `supplementary_team19_gi2100_vsj7589.zip`

## License

MIT (see [`LICENSE`](LICENSE)).
