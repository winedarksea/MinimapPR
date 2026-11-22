#!/usr/bin/env python3
"""Download BirdNET+ V3.0.1 ONNX model files from Zenodo and verify checksums.

Source: Zenodo record 24362980 ("BirdNET+ V3.0 model files", version 3.0.1,
2026-11-19, DOI 10.5281/zenodo.24362980). Weights and labels are Apache 2.0;
ACCEPTABLE_USE.txt asks derived apps not to use BirdNET branding and not to be
used for covert surveillance of people (guidance, not a licence term).

Files land in a gitignored directory (default ``data/models/birdnet_v3/``).
Downloads are streamed to ``<name>.part`` and only renamed after the MD5 in the
pinned manifest below matches, so a truncated or tampered file never reaches the
classifier path. Model files are treated as data: nothing in them is executed.

Usage::

    python scripts/fetch_birdnet_v3.py                    # pruned FP16 + labels + licence
    python scripts/fetch_birdnet_v3.py --include-full-fp16 --include-taxonomy
    python scripts/fetch_birdnet_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("fetch_birdnet_v3")

ZENODO_RECORD_ID = "24362980"
ZENODO_FILE_URL_TEMPLATE = "https://zenodo.org/api/records/{record}/files/{name}/content"
DEFAULT_OUTPUT_DIRECTORY = Path("data/models/birdnet_v3")
GENERIC_USER_AGENT = "model-fetch-client/1.0"
DOWNLOAD_CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class PinnedZenodoFile:
    name: str
    md5_hex: str
    size_bytes: int
    always_fetched: bool


# Pinned from the Zenodo API listing of record 24362980 (fetched 2026-11-21).
PINNED_FILES: tuple[PinnedZenodoFile, ...] = (
    PinnedZenodoFile("BirdNET+_V3.0_Global_12K_FP16_pruned.onnx", "6b42d8905f25f856fb8ad91544138914", 72317224, True),
    PinnedZenodoFile("BirdNET+_V3.0_Global_12K_Labels.csv", "bb25847af8a581e1382081382f01e762", 8007144, True),
    PinnedZenodoFile("MODEL_LICENSE.txt", "77b299ba50d3119cd28c5a2b6fc0ff4b", 11836, True),
    PinnedZenodoFile("ACCEPTABLE_USE.txt", "92f52349d7a08f74167f096702bac26a", 2986, True),
    PinnedZenodoFile("BirdNET+_V3.0_Global_12K_FP16.onnx", "3f284bdeae7ffb38718c391839b58b60", 173871766, False),
    # Taxonomy v0.5 (synonyms) - only needed to rebuild the v2.4 -> V3 label map.
    PinnedZenodoFile("BirdNET+_Taxonomy_v0.5-Nov2026.zip", "0310312e852326e17e45e5b2dcf64655", 69831066, False),
)
OPTIONAL_FILE_FLAGS: dict[str, str] = {
    "BirdNET+_V3.0_Global_12K_FP16.onnx": "include_full_fp16",
    "BirdNET+_Taxonomy_v0.5-Nov2026.zip": "include_taxonomy",
}


def compute_file_md5_hex(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches_pin(path: Path, pinned: PinnedZenodoFile) -> bool:
    return path.is_file() and path.stat().st_size == pinned.size_bytes and compute_file_md5_hex(path) == pinned.md5_hex


def download_pinned_file(pinned: PinnedZenodoFile, output_directory: Path) -> None:
    destination = output_directory / pinned.name
    if file_matches_pin(destination, pinned):
        logger.info("ok (cached)  %s", pinned.name)
        return
    partial_path = destination.with_name(destination.name + ".part")
    url = ZENODO_FILE_URL_TEMPLATE.format(record=ZENODO_RECORD_ID, name=urllib.request.quote(pinned.name))
    request = urllib.request.Request(url, headers={"User-Agent": GENERIC_USER_AGENT})
    logger.info("downloading %s (%.1f MB)", pinned.name, pinned.size_bytes / 1e6)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, partial_path.open("wb") as out:
        for chunk in iter(lambda: response.read(DOWNLOAD_CHUNK_BYTES), b""):
            out.write(chunk)
    if not file_matches_pin(partial_path, pinned):
        actual = compute_file_md5_hex(partial_path)
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {pinned.name}: expected {pinned.md5_hex}, got {actual}")
    partial_path.replace(destination)
    logger.info("ok (verified) %s md5=%s", pinned.name, pinned.md5_hex)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--include-full-fp16", action="store_true", help="also fetch the 174 MB unpruned FP16 model")
    parser.add_argument("--include-taxonomy", action="store_true", help="also fetch the 70 MB taxonomy zip (label-map rebuilds)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        f for f in PINNED_FILES if f.always_fetched or getattr(args, OPTIONAL_FILE_FLAGS.get(f.name, ""), False)
    ]
    failures = 0
    for pinned in selected:
        if args.verify_only:
            ok = file_matches_pin(args.output_dir / pinned.name, pinned)
            logger.info("%s %s", "ok     " if ok else "MISSING", pinned.name)
            failures += 0 if ok else 1
            continue
        try:
            download_pinned_file(pinned, args.output_dir)
        except (OSError, RuntimeError) as error:
            logger.error("failed %s: %s", pinned.name, error)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
