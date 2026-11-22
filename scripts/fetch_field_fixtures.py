#!/usr/bin/env python3
"""Fetch the out-of-git field fixtures listed in tests/fixtures/field_bundles.toml.

usage:
  python scripts/fetch_field_fixtures.py [fetch] [--only NAME ...] [--cache-dir DIR]
  python scripts/fetch_field_fixtures.py list [--cache-dir DIR]
  python scripts/fetch_field_fixtures.py register FILE --name N --source URL \
      --description TEXT --licence TEXT [--extract]

Every item is verified (size, then sha256) before it is moved into the cache, so a partial or
tampered download never looks valid to `require_field_bundle`. Sources: https:// (generic
User-Agent, no personal data), ssh://user@host:port/abs/path (rsync over ssh — the Bluefin
tunnel), file:///abs/path. Cache: --cache-dir, else $MINIMAPPR_FIELD_FIXTURE_CACHE, else
~/.cache/minimappr/field_fixtures. tests/field_audio_fixtures.py imports this module, so the
manifest validation and verification rules live only here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
HASH_READ_BLOCK_BYTES = 1 << 20
DEFAULT_SSH_PORT = 22
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")
# ssh user/host/path are handed to rsync, which forwards host to ssh and the path to the
# remote shell; a leading "-" would become an option and shell metacharacters would be
# expanded remotely (macOS openrsync has no --protect-args), so only a conservative
# character set is accepted.
SSH_USER_OR_HOST_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
SSH_REMOTE_PATH_PATTERN = re.compile(r"/[A-Za-z0-9._/+-]+")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    return cache_dir or Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE))


def extracted_dir_for(archive: Path) -> Path:
    """Directory an ``extract = true`` archive unpacks into: the archive path minus .tar*."""
    for suffix in ARCHIVE_SUFFIXES:
        if archive.name.endswith(suffix) and len(archive.name) > len(suffix):
            return archive.parent / archive.name[: -len(suffix)]
    raise ValueError(f"{archive.name!r} is not a tar archive ({', '.join(ARCHIVE_SUFFIXES)})")


def load_bundles(manifest_path: Path = BUNDLES_MANIFEST_PATH) -> list[dict]:
    bundles = tomllib.loads(manifest_path.read_text(encoding="utf-8")).get("bundle", [])
    names = set()
    for bundle in bundles:
        label = bundle.get("name", "?")
        missing = [key for key in REQUIRED_KEYS if key not in bundle]
        if missing:
            raise ValueError(f"bundle {label!r} is missing {missing}")
        if bundle["name"] in names:
            raise ValueError(f"duplicate bundle name {bundle['name']!r}")
        if not SHA256_HEX_PATTERN.fullmatch(str(bundle["sha256"])):
            raise ValueError(f"bundle {label!r}: sha256 must be 64 lowercase hex chars")
        cache_path = Path(bundle["cache_path"])
        if cache_path.is_absolute() or ".." in cache_path.parts or not cache_path.parts:
            raise ValueError(f"bundle {label!r}: cache_path must stay inside the cache")
        if bundle.get("extract"):
            extracted_dir_for(cache_path)
        names.add(bundle["name"])
    return bundles


def rsync_remote_spec(source: str) -> tuple[str, int]:
    """(``user@host:/path``, port) for an ssh:// source, rejecting option/shell injection."""
    parsed = urllib.parse.urlparse(source)
    user, host, path = parsed.username, parsed.hostname, urllib.parse.unquote(parsed.path)
    if not (user and SSH_USER_OR_HOST_PATTERN.fullmatch(user)):
        raise ValueError("ssh source needs a plain user name (ssh://user@host:port/abs/path)")
    if not (host and SSH_USER_OR_HOST_PATTERN.fullmatch(host)):
        raise ValueError(f"ssh source host {host!r} has unsupported characters")
    if not SSH_REMOTE_PATH_PATTERN.fullmatch(path):
        raise ValueError(f"ssh source path {path!r} must be absolute with [A-Za-z0-9._/+-] only")
    return f"{user}@{host}:{path}", int(parsed.port or DEFAULT_SSH_PORT)


class _LimitedReader:
    """Caps how many bytes ``shutil.copyfileobj`` can pull from an HTTP response."""

    def __init__(self, stream, limit_bytes: int) -> None:
        self._stream, self._remaining = stream, limit_bytes

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        size = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._stream.read(size)
        self._remaining -= len(chunk)
        return chunk


def download_to(source: str, destination: Path, expected_size_bytes: int) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "https":
        request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response, \
                destination.open("wb") as out:
            # One byte past the manifest size is enough for verify() to reject an oversized
            # body without streaming it all to disk.
            shutil.copyfileobj(_LimitedReader(response, expected_size_bytes + 1), out)
    elif parsed.scheme == "ssh":
        remote, port = rsync_remote_spec(source)
        subprocess.run(
            ["rsync", "-t", "-e", f"ssh -p {port}", "--", remote, str(destination)],
            check=True,
        )
    elif parsed.scheme == "file":
        shutil.copyfile(urllib.parse.unquote(parsed.path), destination)
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
    """Unpack a verified archive via a staging dir + rename, replacing any older copy, so an
    interrupted extraction never leaves a half-populated directory that looks complete."""
    target = extracted_dir_for(archive)
    with tempfile.TemporaryDirectory(dir=archive.parent) as staging:
        staged = Path(staging) / target.name
        staged.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(staged, filter="data")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staged, target)
    return target


def fetch(bundles: list[dict], cache_dir: Path, only: list[str] | None) -> int:
    known = {bundle["name"] for bundle in bundles}
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
                    download_to(bundle["source"], staged, int(bundle["size_bytes"]))
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
    # json.dumps yields a valid TOML basic string (quotes, backslashes, unicode escaped).
    lines = [
        "[[bundle]]",
        f"name = {json.dumps(args.name)}",
        f'sha256 = "{sha256_of_file(path)}"',
        f"size_bytes = {path.stat().st_size}",
        f"source = {json.dumps(args.source)}",
        f"cache_path = {json.dumps(args.cache_path or path.name)}",
    ]
    if args.extract:
        lines.append("extract = true")
    lines += [f"description = {json.dumps(args.description)}",
              f"licence = {json.dumps(args.licence)}", "speech_checked = false"]
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Options accepted both before and after the subcommand. SUPPRESS stops a subparser's
    # default from overwriting a value given before the subcommand name.
    cache_option = argparse.ArgumentParser(add_help=False)
    cache_option.add_argument("--cache-dir", type=Path, default=argparse.SUPPRESS)
    only_option = argparse.ArgumentParser(add_help=False)
    only_option.add_argument("--only", nargs="+", default=argparse.SUPPRESS, metavar="NAME")

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], parents=[cache_option, only_option]
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("fetch", parents=[cache_option, only_option],
                   help="fetch and verify bundles (default)")
    sub.add_parser("list", parents=[cache_option], help="list manifest entries and cache status")
    reg = sub.add_parser("register", help="print a [[bundle]] table for a local file")
    reg.add_argument("file")
    reg.add_argument("--name", required=True)
    reg.add_argument("--source", required=True)
    reg.add_argument("--description", required=True)
    reg.add_argument("--licence", required=True)
    reg.add_argument("--cache-path", default=None)
    reg.add_argument("--extract", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        return register(args)
    cache_dir = resolve_cache_dir(getattr(args, "cache_dir", None))
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
    return fetch(bundles, cache_dir, getattr(args, "only", None))


if __name__ == "__main__":
    sys.exit(main())
