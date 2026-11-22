#!/usr/bin/env python3
"""Fetch the out-of-git field fixtures listed in tests/fixtures/field_bundles.toml.

usage:
  python scripts/fetch_field_fixtures.py [fetch] [--only NAME ...] [--cache-dir DIR]
  python scripts/fetch_field_fixtures.py list
  python scripts/fetch_field_fixtures.py register FILE --name N --source URL \
      --description TEXT --licence TEXT [--extract]

Every item is verified (size, then sha256) before it is moved into the cache, so a partial or
tampered download never looks valid to `require_field_bundle`. Sources: https:// (generic
User-Agent, no personal data), ssh://user@host:port/abs/path (rsync over ssh — the Bluefin
tunnel), file:///abs/path. Cache: --cache-dir, else $MINIMAPPR_FIELD_FIXTURE_CACHE, else
~/.cache/minimappr/field_fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLES_MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "field_bundles.toml"
CACHE_ENV = "MINIMAPPR_FIELD_FIXTURE_CACHE"
DEFAULT_CACHE = Path.home() / ".cache" / "minimappr" / "field_fixtures"
USER_AGENT = "minimappr-fixture-fetch/0.1"
REQUIRED_KEYS = ("name", "sha256", "size_bytes", "source", "cache_path", "description", "licence")
HTTP_TIMEOUT_SECONDS = 300


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundles(manifest_path: Path = BUNDLES_MANIFEST_PATH) -> list[dict]:
    bundles = tomllib.loads(manifest_path.read_text(encoding="utf-8")).get("bundle", [])
    names = set()
    for bundle in bundles:
        missing = [k for k in REQUIRED_KEYS if k not in bundle]
        if missing:
            raise ValueError(f"bundle {bundle.get('name', '?')!r} is missing {missing}")
        if bundle["name"] in names:
            raise ValueError(f"duplicate bundle name {bundle['name']!r}")
        if Path(bundle["cache_path"]).is_absolute() or ".." in Path(bundle["cache_path"]).parts:
            raise ValueError(f"bundle {bundle['name']!r}: cache_path must stay inside the cache")
        names.add(bundle["name"])
    return bundles


def download_to(source: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "https":
        request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response, \
                destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    elif parsed.scheme == "ssh":
        port = parsed.port or 22
        remote = f"{parsed.username}@{parsed.hostname}:{parsed.path}"
        subprocess.run(
            ["rsync", "-t", "--partial", "-e", f"ssh -p {port}", remote, str(destination)],
            check=True,
        )
    elif parsed.scheme == "file":
        shutil.copyfile(parsed.path, destination)
    else:
        raise ValueError(f"unsupported source scheme in {source!r}")


def verify(path: Path, bundle: dict) -> str | None:
    """None when the file matches the manifest, else a reason string."""
    size = path.stat().st_size
    if size != int(bundle["size_bytes"]):
        return f"size {size} != manifest {bundle['size_bytes']}"
    actual = sha256_of_file(path)
    if actual != bundle["sha256"]:
        return f"sha256 {actual} != manifest {bundle['sha256']}"
    return None


def extract_archive(archive: Path) -> Path:
    target = archive.parent / archive.name.split(".tar")[0]
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return target


def fetch(bundles: list[dict], cache_dir: Path, only: list[str] | None) -> int:
    known = {b["name"] for b in bundles}
    unknown = sorted(set(only or []) - known)
    if unknown:
        print(f"unknown bundle names: {unknown}", file=sys.stderr)
        return 2
    failures = 0
    for bundle in bundles:
        if only and bundle["name"] not in only:
            continue
        destination = cache_dir / bundle["cache_path"]
        if destination.exists() and verify(destination, bundle) is None:
            print(f"ok      {bundle['name']} (cached)")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
                staged = Path(staging) / destination.name
                try:
                    download_to(bundle["source"], staged)
                except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                    print(f"FAILED  {bundle['name']}: fetch error: {exc}", file=sys.stderr)
                    failures += 1
                    continue
                reason = verify(staged, bundle)
                if reason:
                    print(f"FAILED  {bundle['name']}: {reason}", file=sys.stderr)
                    failures += 1
                    continue
                os.replace(staged, destination)
            print(f"fetched {bundle['name']} -> {destination}")
        if bundle.get("extract"):
            print(f"        extracted to {extract_archive(destination)}")
    return 1 if failures else 0


def register(args: argparse.Namespace) -> int:
    path = Path(args.file)
    lines = [
        "[[bundle]]",
        f'name = "{args.name}"',
        f'sha256 = "{sha256_of_file(path)}"',
        f"size_bytes = {path.stat().st_size}",
        f'source = "{args.source}"',
        f'cache_path = "{args.cache_path or path.name}"',
    ]
    if args.extract:
        lines.append("extract = true")
    lines += [f'description = "{args.description}"', f'licence = "{args.licence}"',
              "speech_checked = false"]
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    fetch_parser = sub.add_parser("fetch", help="fetch and verify bundles (default)")
    fetch_parser.add_argument("--only", nargs="+", default=None)
    sub.add_parser("list", help="list manifest entries and cache status")
    reg = sub.add_parser("register", help="print a [[bundle]] table for a local file")
    reg.add_argument("file")
    reg.add_argument("--name", required=True)
    reg.add_argument("--source", required=True)
    reg.add_argument("--description", required=True)
    reg.add_argument("--licence", required=True)
    reg.add_argument("--cache-path", default=None)
    reg.add_argument("--extract", action="store_true")
    # Bare `--only` without the subcommand is the common case; accept it.
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.command == "register":
        return register(args)
    cache_dir = args.cache_dir or Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE))
    bundles = load_bundles()
    if args.command == "list":
        for bundle in bundles:
            local = cache_dir / bundle["cache_path"]
            state = "missing" if not local.exists() else (verify(local, bundle) or "ok")
            print(f"{bundle['name']:40s} {int(bundle['size_bytes']) / 1e6:9.2f} MB  {state}")
        if not bundles:
            print(f"no bundles listed in {BUNDLES_MANIFEST_PATH}")
        return 0
    if not bundles:
        print(f"no bundles listed in {BUNDLES_MANIFEST_PATH}; nothing to fetch")
        return 0
    return fetch(bundles, cache_dir, args.only)


if __name__ == "__main__":
    sys.exit(main())
