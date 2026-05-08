"""ESM-2 (650M) residue-level embedding extraction with sharded, resumable, batched output.

Strategy
--------
We minimize the number of ESM-2 forward passes:

* **Short proteins** (``len(seq) <= max_residues``) are forwarded **once**
  and embeddings for **all** their variants are read off the same hidden
  state.
* **Long proteins** (``len(seq) > max_residues``) are tiled with overlapping
  windows of ``max_residues`` residues at stride ``max_residues // 2``.
  Each tile is forwarded **once**; each variant is then assigned to the
  tile that places it closest to the tile's center, and its embedding is
  read off that tile.
* Tasks (one per short protein OR per tile of a long protein) are sorted
  by length and batched into mini-batches with right-padding so that
  similar-length tasks share GPU work. Padding tokens are present in the
  output but ignored when reading variant positions.

Sharding & resume
-----------------
Tasks are partitioned into shards of approximately ``shard_size`` tasks.
Each shard writes one ``.npz`` and one entry in ``manifest.json``; a re-run
skips shards already in the manifest. Multi-GPU is supported by partitioning
shards across ranks: rank ``r`` of ``WORLD_SIZE`` ranks processes shards
where ``shard_id % WORLD_SIZE == r``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from imc.data.uniprot import UniprotEntry
from imc.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class ESM2Config:
    """Hyperparameters for ESM-2 residue-embedding extraction."""

    model_id: str = "facebook/esm2_t33_650M_UR50D"
    hidden_dim: int = 1280
    max_residues: int = 1022
    shard_size: int = 1000
    dtype: str = "float16"
    batch_size: int = 8


@dataclass
class _Task:
    """A single forward-pass unit (one short protein OR one tile of a long protein)."""

    seq: str
    aa_offset: int
    variants: list[tuple[str, int]] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Sub-sequence length in residues (no special tokens)."""
        return len(self.seq)


def _torch_dtype(name: str) -> torch.dtype:
    """Map a string dtype name to a ``torch.dtype``."""
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _build_tasks(
    variants: pd.DataFrame,
    swissprot: dict[str, UniprotEntry],
    max_residues: int,
) -> list[_Task]:
    """Build the global task list (one per short protein or per long-protein tile)."""
    by_acc = {acc: g for acc, g in variants.groupby("uniprot_acc")}
    tasks: list[_Task] = []
    for acc, vrows in by_acc.items():
        entry = swissprot.get(acc)
        if entry is None:
            continue
        seq = entry.sequence
        n = len(seq)
        if n <= max_residues:
            task = _Task(seq=seq, aa_offset=0)
            for _, row in vrows.iterrows():
                pos = int(row["position_aa"])
                if 1 <= pos <= n:
                    task.variants.append((str(row["variation_id"]), pos))
            if task.variants:
                tasks.append(task)
            continue

        stride = max_residues // 2
        tile_starts: list[int] = []
        s = 0
        while True:
            tile_starts.append(s)
            if s + max_residues >= n:
                break
            s += stride
        if tile_starts[-1] + max_residues < n:
            tile_starts.append(max(0, n - max_residues))

        tile_objs: list[_Task] = []
        for ts in tile_starts:
            te = min(n, ts + max_residues)
            tile_objs.append(_Task(seq=seq[ts:te], aa_offset=ts))

        for _, row in vrows.iterrows():
            pos = int(row["position_aa"])
            if pos < 1 or pos > n:
                continue
            best_idx = -1
            best_score = float("-inf")
            for i, t in enumerate(tile_objs):
                te = t.aa_offset + t.length
                if t.aa_offset + 1 <= pos <= te:
                    center = t.aa_offset + t.length / 2.0
                    score = -abs(pos - center)
                    if score > best_score:
                        best_score = score
                        best_idx = i
            if best_idx >= 0:
                t = tile_objs[best_idx]
                within = pos - t.aa_offset
                t.variants.append((str(row["variation_id"]), within))

        for t in tile_objs:
            if t.variants:
                tasks.append(t)
    LOG.info("Built %d tasks for %d proteins", len(tasks), len(by_acc))
    return tasks


def _shard_tasks(tasks: list[_Task], shard_size: int) -> list[list[_Task]]:
    """Partition tasks into shards. Tasks are kept in the input order."""
    shards: list[list[_Task]] = []
    for i in range(0, len(tasks), shard_size):
        shards.append(tasks[i : i + shard_size])
    return shards


def _read_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _write_manifest_atomic(manifest_path: Path, manifest: dict[str, dict[str, object]]) -> None:
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(manifest_path)


def _process_shard(
    shard: list[_Task],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[np.ndarray]]:
    """Run all tasks in a shard through the model and collect variant embeddings."""
    indexed = sorted(enumerate(shard), key=lambda p: -p[1].length)
    var_ids: list[str] = []
    embs: list[np.ndarray] = []

    for batch_start in range(0, len(indexed), batch_size):
        batch = indexed[batch_start : batch_start + batch_size]
        seqs = [t.seq for _, t in batch]
        toks = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            out = model(**toks)
        h = out.last_hidden_state  # [B, seq_len_max+2, hidden]
        for i, (_, task) in enumerate(batch):
            for vid, pos_in_tile in task.variants:
                if 1 <= pos_in_tile <= task.length:
                    vec = h[i, pos_in_tile].detach().to(torch.float32).cpu().numpy()
                    var_ids.append(vid)
                    embs.append(vec)
    return var_ids, embs


def extract_embeddings(
    variants: pd.DataFrame,
    swissprot: dict[str, UniprotEntry],
    cfg: ESM2Config,
    out_dir: str | Path,
    *,
    device: str | None = None,
    progress: bool = True,
) -> Path:
    """Extract per-variant ESM-2 residue embeddings, sharded, batched, resumable.

    Parameters
    ----------
    variants : pandas.DataFrame
        Mapped variants with at least ``variation_id, uniprot_acc,
        position_aa`` columns.
    swissprot : dict[str, UniprotEntry]
        UniProt sequences keyed by accession.
    cfg : ESM2Config
        Extraction configuration.
    out_dir : str or Path
        Directory for shard ``.npz`` files and ``manifest.json``.
    device : str or None
        Torch device string (default: ``cuda`` if available, else ``cpu``).
    progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    Path
        Path to ``manifest.json`` after this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _torch_dtype(cfg.dtype)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    LOG.info(
        "Extracting ESM-2 embeddings: model=%s device=%s dtype=%s batch=%d world_size=%d rank=%d",
        cfg.model_id, device, cfg.dtype, cfg.batch_size, world_size, rank,
    )

    tasks = _build_tasks(variants, swissprot, cfg.max_residues)
    shards = _shard_tasks(tasks, cfg.shard_size)
    LOG.info("Total tasks: %d -> %d shards (size %d)", len(tasks), len(shards), cfg.shard_size)

    manifest = _read_manifest(manifest_path)
    todo = []
    for sid, shard in enumerate(shards):
        if str(sid) in manifest:
            continue
        if sid % world_size != rank:
            continue
        todo.append((sid, shard))
    LOG.info("Rank %d will process %d shards (skipping %d already done)", rank, len(todo), len(manifest))
    if not todo:
        return manifest_path

    LOG.info("Loading %s ...", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModel.from_pretrained(cfg.model_id, torch_dtype=torch_dtype).to(device)
    model.eval()

    bar = tqdm(todo, desc=f"rank{rank}/shards", disable=not progress)
    for sid, shard in bar:
        var_ids, embs = _process_shard(
            shard=shard,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=cfg.batch_size,
        )
        if not var_ids:
            LOG.warning("Shard %d produced no embeddings; skipping write.", sid)
            continue
        emb = np.stack(embs).astype(np.float32)
        ids = np.array(var_ids, dtype=object)
        shard_path = out_dir / f"shard_{sid:05d}.npz"
        np.savez_compressed(shard_path, variation_id=ids, embedding=emb)
        manifest[str(sid)] = {
            "shard_path": shard_path.name,
            "n_variants": int(len(var_ids)),
            "rank": rank,
        }
        _write_manifest_atomic(manifest_path, manifest)
        bar.set_postfix(last_n=len(var_ids))
        LOG.info("Wrote shard %d -> %s (n=%d)", sid, shard_path.name, len(var_ids))

    return manifest_path


def consolidate_shards(out_dir: str | Path) -> Path:
    """Concatenate all completed shards into a single ``embeddings.parquet``."""
    out_dir = Path(out_dir)
    shards = sorted(out_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shards found in {out_dir}")
    var_ids: list[np.ndarray] = []
    embs: list[np.ndarray] = []
    for s in shards:
        z = np.load(s, allow_pickle=True)
        var_ids.append(z["variation_id"])
        embs.append(z["embedding"])
    var_id = np.concatenate(var_ids)
    emb = np.vstack(embs)
    LOG.info("Consolidated %d shards -> %d variants x %d dims", len(shards), emb.shape[0], emb.shape[1])

    df = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df.insert(0, "variation_id", var_id.astype(str))
    out_parquet = out_dir / "embeddings.parquet"
    df.to_parquet(out_parquet, index=False)
    LOG.info("Wrote %s (size: %.1f MB)", out_parquet, out_parquet.stat().st_size / 1e6)
    return out_parquet
