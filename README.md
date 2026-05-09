<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![GitHub][github-shield]][github-url]

# Interpretable Missense Classification

<br />
<div align="center">

Comparing interpretable sequence features and protein language model embeddings for human missense variant pathogenicity classification.

<p align="center">
  <a href="https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification"><strong>Explore the repository »</strong></a>
</p>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#repository-layout">Repository layout</a></li>
    <li><a href="#hardware-used">Hardware used</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#submission-artifacts">Submission artifacts</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

We build binary classifiers that predict whether a human missense variant in ClinVar is pathogenic / likely pathogenic versus benign / likely benign. The goal is **not** to beat state-of-the-art predictors, but to study:

1. How much pathogenicity signal is captured by simple, interpretable handcrafted sequence features (logistic regression / random forest on reference and alternate amino acid identities, BLOSUM62 substitution score, local sequence-window composition, normalized residue position).
2. Whether residue-level embeddings from the pretrained protein language model **ESM-2 (650M parameters)** improve cross-gene generalization beyond those features.

We evaluate models with a **gene-held-out** split (entire genes are held out of training so test variants come from genes the model has never seen) using AUROC, AUPRC, F1, and confusion matrices. As reference points (not targets to beat) we compare against the established pathogenicity predictors **AlphaMissense** and **CADD** on the same held-out test variants.

Authors: Georgios Ioannou (gi2100), Vedant Jagtap (vsj7589).

### Built With

[![Python][Python]][Python-url]
[![PyTorch][PyTorch]][PyTorch-url]
[![Hugging Face][HuggingFace]][HuggingFace-url]
[![scikit-learn][scikitlearn]][scikitlearn-url]
[![Pandas][Pandas]][Pandas-url]
[![NumPy][Numpy]][Numpy-url]
[![Biopython][Biopython]][Biopython-url]
[![Matplotlib][Matplotlib]][Matplotlib-url]
[![Seaborn][Seaborn]][Seaborn-url]
[![PyYAML][PyYAML]][PyYAML-url]
[![pytest][pytest]][pytest-url]
[![Ruff][Ruff]][Ruff-url]
[![Black][Black]][Black-url]

Additional libraries in [`requirements.txt`](requirements.txt) include **Accelerate**, **Transformers**, **pysam**, **PyArrow**, **SciPy**, **h5py**, **UMAP-learn**, **tqdm**, **Joblib**, and **Requests**.

<p align="right"><a href="#readme-top">Back to top</a></p>

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

<p align="right"><a href="#readme-top">Back to top</a></p>

## Hardware used

A single NVIDIA GH200 96 GB (CUDA 12, Ubuntu 22). The training and embedding-extraction code is written so that it transparently uses multiple GPUs without code changes when launched via `accelerate launch --num_processes=N`.

<p align="right"><a href="#readme-top">Back to top</a></p>

## Getting Started

**To reproduce experiments locally, follow these steps.**

### Prerequisites

Create and activate a Python virtual environment (example: `.venv`). The `Makefile` target `install` installs PyTorch from the CUDA 12.6 index, then the rest of [`requirements.txt`](requirements.txt).

### Setup

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

<p align="right"><a href="#readme-top">Back to top</a></p>

## Submission artifacts

- Final report PDF: [`reports/paper/final_report_team19_gi2100_vsj7589.pdf`](https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/blob/main/reports/paper/final_report_team19_gi2100_vsj7589.pdf)
- Supplementary ZIP: [`supplementary_team19_gi2100_vsj7589.zip`](https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/blob/main/supplementary_team19_gi2100_vsj7589.zip)

<p align="right"><a href="#readme-top">Back to top</a></p>

## License

MIT (see [`LICENSE`](LICENSE)).

<p align="right"><a href="#readme-top">Back to top</a></p>

## Contact

Georgios Ioannou - [@LinkedIn](https://linkedin.com/in/georgiosioannoucoder)
Vedant Jagtap (vsj7589)

Project link: [https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification](https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification)

<p align="right"><a href="#readme-top">Back to top</a></p>

[contributors-shield]: https://img.shields.io/github/contributors/GeorgiosIoannouCoder/interpretable-missense-classification.svg?style=for-the-badge
[contributors-url]: https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/graphs/contributors

[license-shield]: https://img.shields.io/github/license/GeorgiosIoannouCoder/interpretable-missense-classification.svg?style=for-the-badge
[license-url]: https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/blob/main/LICENSE

[forks-shield]: https://img.shields.io/github/forks/GeorgiosIoannouCoder/interpretable-missense-classification.svg?style=for-the-badge
[forks-url]: https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/network/members

[stars-shield]: https://img.shields.io/github/stars/GeorgiosIoannouCoder/interpretable-missense-classification.svg?style=for-the-badge
[stars-url]: https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/stargazers

[issues-shield]: https://img.shields.io/github/issues/GeorgiosIoannouCoder/interpretable-missense-classification.svg?style=for-the-badge
[issues-url]: https://github.com/GeorgiosIoannouCoder/interpretable-missense-classification/issues

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=0077B5
[linkedin-url]: https://linkedin.com/in/georgiosioannoucoder

[github-shield]: https://img.shields.io/badge/-GitHub-black.svg?style=for-the-badge&logo=github&colorB=000
[github-url]: https://github.com/GeorgiosIoannouCoder/

[Python]: https://img.shields.io/badge/python-FFDE57?style=for-the-badge&logo=python&logoColor=4584B6
[Python-url]: https://www.python.org/

[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/

[HuggingFace]: https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000
[HuggingFace-url]: https://huggingface.co/

[scikitlearn]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[scikitlearn-url]: https://scikit-learn.org/stable/

[Pandas]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[Numpy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/

[Biopython]: https://img.shields.io/badge/Biopython-198CFF?style=for-the-badge
[Biopython-url]: https://biopython.org/

[Matplotlib]: https://img.shields.io/badge/matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/

[Seaborn]: https://img.shields.io/badge/seaborn-7db0bc?style=for-the-badge&logo=seaborn&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/

[PyYAML]: https://img.shields.io/badge/PyYAML-166FAD?style=for-the-badge&logo=yaml&logoColor=white
[PyYAML-url]: https://pyyaml.org/

[pytest]: https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white
[pytest-url]: https://pytest.org/

[Ruff]: https://img.shields.io/badge/Ruff-261230?style=for-the-badge&logo=ruff&logoColor=d7ff64
[Ruff-url]: https://docs.astral.sh/ruff/

[Black]: https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge
[Black-url]: https://black.readthedocs.io/
