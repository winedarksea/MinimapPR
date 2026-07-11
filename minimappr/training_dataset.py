"""Managed, operator-curated audio examples for future model training."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote


class TrainingDatasetError(RuntimeError):
    """Raised when a reviewed detection cannot be materialized safely."""


def _example_paths(dataset_dir: Path, detection_id: str) -> tuple[Path, Path]:
    # IDs originate from the server, but escaping keeps the managed directory
    # boundary intact if an external producer supplies an unusual ID.
    filename = quote(detection_id, safe="-_.")
    return dataset_dir / f"{filename}.wav", dataset_dir / f"{filename}.json"


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with source.open("rb") as source_file, temporary.open("wb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_example(
    *,
    dataset_dir: Path,
    detection_id: str,
    source_audio_path: Path,
    manifest: dict,
) -> tuple[str, str]:
    if not source_audio_path.is_file():
        raise TrainingDatasetError("Detection audio is unavailable; it may have expired")
    audio_path, manifest_path = _example_paths(dataset_dir, detection_id)
    _atomic_copy(source_audio_path, audio_path)
    try:
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise
    return str(audio_path), str(manifest_path)


async def materialize_training_example(
    *,
    dataset_dir: Path,
    detection_id: str,
    source_audio_path: Path,
    manifest: dict,
) -> tuple[str, str]:
    return await asyncio.to_thread(
        _materialize_example,
        dataset_dir=dataset_dir,
        detection_id=detection_id,
        source_audio_path=source_audio_path,
        manifest=manifest,
    )


def _delete_example_files(dataset_dir: Path, example: dict) -> None:
    root = dataset_dir.resolve()
    for raw_path in (
        example.get("audio_path"),
        example.get("manifest_path"),
        example.get("embedding_path"),
    ):
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if path.is_relative_to(root):
            path.unlink(missing_ok=True)


async def delete_training_example_files(dataset_dir: Path, example: dict) -> None:
    await asyncio.to_thread(_delete_example_files, dataset_dir, example)
