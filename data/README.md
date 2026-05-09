# Data layout

This directory holds all data the pipeline reads. Its contents are **not** committed to git (see the project's `.gitignore`); they are reproducible from the download script.

## Subdirectories

- `raw/` - primary downloaded files (ClinVar, UniProt FASTA, UniProt id-mapping).
- `external/` - third-party precomputed scores (AlphaMissense, CADD, CADD `.tbi`).
- `processed/` - outputs of Phases 2-4 (filtered ClinVar, UniProt-mapped variants, splits, dataset stats, `clinvar_release.txt`).
- `embeddings/` - Phase 6 ESM-2 residue embeddings, sharded by `data/embeddings/esm2_650M/shard_*.npz` plus a `manifest.json`.

## Re-creating the data

```bash
make data
make preprocess
make features
make embeddings
```

See [`../REPRODUCE.md`](../REPRODUCE.md) for the full reproduction guide.

## Sources

| File | Source | Approx size |
|---|---|---|
| `variant_summary.txt.gz` | NCBI ClinVar FTP | ~120 MB compressed |
| `UP000005640_9606.fasta.gz` | UniProt UP000005640 | ~10 MB compressed |
| `HUMAN_9606_idmapping_selected.tab.gz` | UniProt id-mapping | ~150 MB compressed |
| `AlphaMissense_hg38.tsv.gz` | Google DeepMind AlphaMissense release | ~1.6 GB |
| `whole_genome_SNVs.tsv.gz` | UW CADD v1.7 GRCh38 | ~80 GB |
| `whole_genome_SNVs.tsv.gz.tbi` | UW CADD v1.7 GRCh38 | ~5 MB |
