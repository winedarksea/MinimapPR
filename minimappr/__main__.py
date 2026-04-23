"""Allow running MinimapPR with ``python -m minimappr``."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import uvicorn

from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.storage.db import Storage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minimappr")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Start the MinimapPR server")

    cleanup_parser = subparsers.add_parser("cleanup", help="Run cleanup and retention maintenance commands")
    cleanup_subparsers = cleanup_parser.add_subparsers(dest="cleanup_mode", required=True)

    full_parser = cleanup_subparsers.add_parser("full", help="Delete the local database and managed file stores")
    full_parser.add_argument("--yes", action="store_true", help="Confirm destructive cleanup")
    full_parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without mutating")
    full_parser.add_argument("--db-path", type=Path, help="Override the configured database path")
    full_parser.add_argument("--snippet-dir", type=Path, help="Override the configured snippet directory")
    full_parser.add_argument("--artifact-dir", type=Path, help="Override the configured artifact directory")

    partial_parser = cleanup_subparsers.add_parser(
        "partial",
        help="Prune old snippets and artifacts using the retention policy",
    )
    partial_parser.add_argument("--policy", type=Path, help="Override the configured retention policy path")
    partial_parser.add_argument("--now-ns", type=int, help="Use a fixed timestamp for deterministic cleanup runs")
    partial_parser.add_argument("--dry-run", action="store_true", help="Report deletions without mutating storage")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in {None, "serve"}:
        _run_server()
        return

    if args.command == "cleanup":
        settings = Settings.from_env()
        asyncio.run(_run_cleanup(args=args, settings=settings, parser=parser))
        return

    parser.error(f"Unsupported command: {args.command}")


def _run_server() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "minimappr.main:app",
        host=settings.host,
        port=settings.port,
        # High-frequency ingest frames would flood the access log; application
        # code emits periodic summaries instead via IngestProcessor.
        access_log=False,
    )


async def _run_cleanup(*, args: argparse.Namespace, settings: Settings, parser: argparse.ArgumentParser) -> None:
    if args.cleanup_mode == "full":
        if not args.yes and not args.dry_run:
            parser.error("`minimappr cleanup full` requires --yes unless --dry-run is used")
        cleanup_service = CleanupService(
            settings=replace(
                settings,
                db_path=args.db_path or settings.db_path,
                snippet_dir=args.snippet_dir or settings.snippet_dir,
                large_artifact_dir=args.artifact_dir or settings.large_artifact_dir,
            )
        )
        summary = await cleanup_service.run_full_cleanup(dry_run=bool(args.dry_run))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.cleanup_mode == "partial":
        storage = Storage(settings.db_path)
        await storage.initialize()
        try:
            cleanup_service = CleanupService(settings=settings, storage=storage)
            summary = await cleanup_service.run_partial_cleanup(
                now_ns=args.now_ns,
                policy_path=args.policy,
                dry_run=bool(args.dry_run),
            )
        finally:
            await storage.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    parser.error(f"Unsupported cleanup mode: {args.cleanup_mode}")


if __name__ == "__main__":
    main()
