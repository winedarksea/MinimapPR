#!/usr/bin/env python3
"""Download a Hugging Face drone-audio dataset into the WAV drop-dir layout.

Streams the parquet shards of a HF audio dataset (default
``geronimobasso/drone-audio-detection-samples``) and writes per-class WAV files
under ``--out-dir/<label>/`` for :mod:`scripts.train_drone_head`. Never loads the
full dataset (6.8 GB) into RAM: parquet is read in batches via pyarrow, and audio
bytes are validated with the stdlib ``wave`` module (no soundfile dependency).

Deterministic subsample: rows are ranked per class by ``sha1(original_path)`` and
the first ``--max-per-class`` are kept. This both caps the dominant ``drone`` class
and rebalances toward the minority negative class. Idempotent — existing files are
skipped.

Usage::

    python scripts/fetch_drone_audio.py --repo geronimobasso/drone-audio-detection-samples \\
      [--revision <pin>] --out-dir drone_dataset/wav --max-per-class 4000 --seed 1234 \\
      --label-map "drone=drone,no drone=ambient,no_drone=ambient" [--dry-run]

Requires the ``train`` extra: ``pip install -e '.[train]'`` (huggingface-hub, pyarrow).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import wave
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("fetch_drone_audio")

DEFAULT_REPO = "geronimobasso/drone-audio-detection-samples"
DEFAULT_LABEL_MAP = "drone=drone,no drone=ambient,no_drone=ambient"

# Candidate column names (verify actual schema with --dry-run first).
_AUDIO_COLUMNS = ("audio", "wav", "waveform", "file")
_LABEL_COLUMNS = ("label", "class", "target", "category")
_PATH_COLUMNS = ("path", "file", "filename", "orig_path", "audio_path")


def _parse_label_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _, value = pair.partition("=")
        key, value = key.strip().lower(), value.strip().lower()
        if key and value:
            out[key] = value
    return out


def _download_parquets(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["**/*.parquet"],
    )
    return Path(local)


def _parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def _extract_audio_bytes(value) -> bytes | None:
    """Pull WAV bytes out of a HF audio cell (dict with 'bytes', or raw bytes)."""
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def _extract_orig_path(value, fallback: str) -> str:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path:
            return path
    if isinstance(value, str) and value:
        return value
    return fallback


def _valid_wav(data: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getnframes() > 0
    except (wave.Error, OSError, EOFError):
        return False


def _pick_column(available: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in available}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _dry_run(parquets: list[Path]) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    if not parquets:
        logger.error("No parquet files found")
        return 1
    pf = pq.ParquetFile(str(parquets[0]))
    logger.info("Schema of %s:\n%s", parquets[0].name, pf.schema_arrow)
    label_col = _pick_column(pf.schema_arrow.names, _LABEL_COLUMNS)
    if label_col:
        values: set[str] = set()
        for batch in pf.iter_batches(columns=[label_col], batch_size=2048):
            values.update(str(v) for v in batch.column(0).to_pylist())
            if len(values) > 50:
                break
        logger.info("Distinct '%s' values (up to 50): %s", label_col, sorted(values)[:50])
    else:
        logger.warning("No label column among %s", _LABEL_COLUMNS)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("drone_dataset/wav"))
    parser.add_argument("--max-per-class", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1234)  # reserved; ranking is hash-deterministic
    parser.add_argument("--label-map", default=DEFAULT_LABEL_MAP)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("Downloading parquet shards from %s ...", args.repo)
    root = _download_parquets(args.repo, args.revision)
    parquets = _parquet_files(root)
    logger.info("Found %d parquet file(s)", len(parquets))

    if args.dry_run:
        return _dry_run(parquets)

    import pyarrow.parquet as pq  # noqa: PLC0415

    label_map = _parse_label_map(args.label_map)

    # Pass 1: collect (class, sort_hash, parquet_index, row_index, orig_path) so we
    # can deterministically pick the lowest-hash rows per class without holding
    # audio bytes in memory.
    candidates: dict[str, list[tuple[str, int, int, str]]] = {}
    audio_col = label_col = path_col = None
    for p_idx, parquet in enumerate(parquets):
        pf = pq.ParquetFile(str(parquet))
        names = pf.schema_arrow.names
        if audio_col is None:
            audio_col = _pick_column(names, _AUDIO_COLUMNS)
            label_col = _pick_column(names, _LABEL_COLUMNS)
            path_col = _pick_column(names, _PATH_COLUMNS) or audio_col
            if audio_col is None or label_col is None:
                logger.error(
                    "Could not locate audio/label columns in %s; run --dry-run to inspect schema",
                    names,
                )
                return 1
        cols = [c for c in {audio_col, label_col, path_col} if c]
        row_base = 0
        for batch in pf.iter_batches(columns=cols, batch_size=1024):
            table = batch.to_pydict()
            n = len(table[label_col])
            for i in range(n):
                raw_label = str(table[label_col][i]).strip().lower()
                mapped = label_map.get(raw_label, raw_label)
                orig = _extract_orig_path(
                    table[path_col][i] if path_col in table else None,
                    fallback=f"{parquet.name}:{row_base + i}",
                )
                sort_hash = hashlib.sha1(orig.encode()).hexdigest()
                candidates.setdefault(mapped, []).append((sort_hash, p_idx, row_base + i, orig))
            row_base += n

    # Deterministic per-class selection: sort by hash, take first max-per-class.
    selected: dict[int, dict[int, tuple[str, str]]] = {}  # parquet_idx -> {row_idx: (label, orig)}
    for label, rows in candidates.items():
        rows.sort(key=lambda r: r[0])
        keep = rows if args.max_per_class <= 0 else rows[: args.max_per_class]
        logger.info("Class %-10s: %d available -> keeping %d", label, len(rows), len(keep))
        for _, p_idx, row_idx, orig in keep:
            selected.setdefault(p_idx, {})[row_idx] = (label, orig)

    # Pass 2: re-read only the selected rows and write WAVs.
    written = {label: 0 for label in candidates}
    skipped_existing = 0
    invalid = 0
    for p_idx, parquet in enumerate(parquets):
        wanted = selected.get(p_idx)
        if not wanted:
            continue
        pf = pq.ParquetFile(str(parquet))
        cols = [c for c in {audio_col, path_col} if c]
        row_base = 0
        for batch in pf.iter_batches(columns=cols, batch_size=1024):
            table = batch.to_pydict()
            n = len(table[audio_col])
            for i in range(n):
                row_idx = row_base + i
                if row_idx not in wanted:
                    continue
                label, orig = wanted[row_idx]
                data = _extract_audio_bytes(table[audio_col][i])
                if data is None or not _valid_wav(data):
                    invalid += 1
                    continue
                out_dir = args.out_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)
                name = f"hf-{hashlib.sha1(orig.encode()).hexdigest()[:12]}.wav"
                dest = out_dir / name
                if dest.exists():
                    skipped_existing += 1
                    continue
                tmp = dest.with_suffix(".wav.tmp")
                tmp.write_bytes(data)
                tmp.replace(dest)
                written[label] += 1
            row_base += n

    logger.info("Done. Written per class: %s", written)
    logger.info("Skipped existing: %d; invalid audio: %d", skipped_existing, invalid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
