"""Allow running MinimapPR with ``python -m minimappr``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import uvicorn

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None  # type: ignore[assignment]

from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.storage.db import Storage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minimappr")
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        default=False,
        help="Do not automatically start the Rust ingest sidecar process",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the MinimapPR server")
    serve_parser.add_argument(
        "--no-sidecar",
        action="store_true",
        default=False,
        help="Do not automatically start the Rust ingest sidecar process",
    )
    subparsers.add_parser("api", help="Start only the API/UI process")
    subparsers.add_parser("ingest", help="Start only the Python DSP/ingest process")

    cleanup_parser = subparsers.add_parser("cleanup", help="Run cleanup and retention maintenance commands")
    cleanup_subparsers = cleanup_parser.add_subparsers(dest="cleanup_mode", required=True)

    full_parser = cleanup_subparsers.add_parser("full", help="Delete the local database and managed file stores")
    full_parser.add_argument("--yes", action="store_true", help="Confirm destructive cleanup")
    full_parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without mutating")
    full_parser.add_argument("--db-path", type=Path, help="Override the configured database path")
    full_parser.add_argument("--snippet-dir", type=Path, help="Override the configured snippet directory")
    full_parser.add_argument("--artifact-dir", type=Path, help="Override the configured artifact directory")
    full_parser.add_argument("--training-dataset-dir", type=Path, help="Override the configured training dataset directory")

    partial_parser = cleanup_subparsers.add_parser(
        "partial",
        help="Prune old snippets and artifacts using the retention policy",
    )
    partial_parser.add_argument("--policy", type=Path, help="Override the configured retention policy path")
    partial_parser.add_argument("--now-ns", type=int, help="Use a fixed timestamp for deterministic cleanup runs")
    partial_parser.add_argument("--dry-run", action="store_true", help="Report deletions without mutating storage")

    return parser


def _set_proc_title(role: str | None = None) -> None:
    if _setproctitle is None:
        return
    title = "minimappr" if role is None else f"minimappr: {role}"
    _setproctitle.setproctitle(title)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in {None, "serve"}:
        _set_proc_title()
        _run_server(no_sidecar=getattr(args, "no_sidecar", False))
        return

    if args.command == "api":
        _set_proc_title("api")
        os.environ["MINIMAPPR_PROCESS_ROLE"] = "api"
        _run_server(no_sidecar=False)
        return

    if args.command == "ingest":
        _set_proc_title("ingest")
        os.environ["MINIMAPPR_PROCESS_ROLE"] = "ingest"
        os.environ["MINIMAPPR_INGEST_BACKEND"] = "python"
        os.environ["MINIMAPPR_DIRECT_INGEST_ENABLED"] = "true"
        os.environ["MINIMAPPR_INGEST_SIDECAR_ENABLED"] = "false"
        _run_ingest_server()
        return

    if args.command == "cleanup":
        settings = Settings.from_env()
        asyncio.run(_run_cleanup(args=args, settings=settings, parser=parser))
        return

    parser.error(f"Unsupported command: {args.command}")


def _run_server(*, no_sidecar: bool = False) -> None:
    if no_sidecar:
        os.environ["MINIMAPPR_INGEST_SIDECAR_ENABLED"] = "false"
    settings = Settings.from_env()
    if _should_supervise_python_ingest(settings):
        _run_split_python_server(settings)
        return
    uvicorn.run(
        "minimappr.main:app",
        host=settings.host,
        port=settings.port,
        # High-frequency ingest frames would flood the access log; application
        # code emits periodic summaries instead via IngestProcessor.
        access_log=False,
    )


def _should_supervise_python_ingest(settings: Settings) -> bool:
    """Return True when one terminal should supervise API + Python ingest."""
    return (
        settings.process_role == "combined"
        and settings.ingest_backend == "python"
        and settings.ingest_port != settings.port
        and os.getenv("MINIMAPPR_INGEST_PORT") is not None
    )


def _run_split_python_server(settings: Settings) -> None:
    ingest_process = _start_python_ingest_process(settings)
    previous_process_role = os.environ.get("MINIMAPPR_PROCESS_ROLE")
    previous_direct_ingest = os.environ.get("MINIMAPPR_DIRECT_INGEST_ENABLED")
    try:
        _wait_for_python_ingest_ready(ingest_process, settings=settings)
        os.environ["MINIMAPPR_PROCESS_ROLE"] = "api"
        os.environ["MINIMAPPR_INGEST_BACKEND"] = "python"
        os.environ["MINIMAPPR_DIRECT_INGEST_ENABLED"] = "false"
        uvicorn.run(
            "minimappr.main:app",
            host=settings.host,
            port=settings.port,
            access_log=False,
        )
    finally:
        _restore_env("MINIMAPPR_PROCESS_ROLE", previous_process_role)
        _restore_env("MINIMAPPR_DIRECT_INGEST_ENABLED", previous_direct_ingest)
        _stop_python_ingest_process(ingest_process)


def _start_python_ingest_process(settings: Settings) -> subprocess.Popen:
    env = {
        **os.environ,
        "MINIMAPPR_PROCESS_ROLE": "ingest",
        "MINIMAPPR_INGEST_BACKEND": "python",
        "MINIMAPPR_DIRECT_INGEST_ENABLED": "true",
        "MINIMAPPR_INGEST_SIDECAR_ENABLED": "false",
        "MINIMAPPR_INGEST_PORT": str(settings.ingest_port),
        "MINIMAPPR_INGEST_HOST": settings.ingest_host,
    }
    print(
        "Starting MinimapPR Python ingest process on "
        f"{settings.ingest_host}:{settings.ingest_port}",
        flush=True,
    )
    return subprocess.Popen([sys.executable, "-m", "minimappr", "ingest"], env=env)


def _wait_for_python_ingest_ready(
    process: subprocess.Popen,
    *,
    settings: Settings,
    timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 0.2,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_host = settings.ingest_host
    if health_host in {"0.0.0.0", "::", ""}:
        health_host = "127.0.0.1"
    health_url = f"http://{health_host}:{settings.ingest_port}/health"
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Python ingest process exited before readiness check completed "
                f"(code {process.returncode})"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if getattr(response, "status", None) == 200:
                    return
        except (OSError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            _stop_python_ingest_process(process)
            raise RuntimeError(
                f"Python ingest process did not become ready at {health_url} "
                f"within {timeout_seconds:.1f}s"
            )
        time.sleep(poll_interval_seconds)


def _stop_python_ingest_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print("Stopping MinimapPR Python ingest process", flush=True)
    process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _restore_env(key: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous_value


def _run_ingest_server() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "minimappr.main:app",
        host=settings.ingest_host,
        port=settings.ingest_port,
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
                training_dataset_dir=args.training_dataset_dir or settings.training_dataset_dir,
            )
        )
        summary = await cleanup_service.run_full_cleanup(dry_run=bool(args.dry_run))
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["db_removed"]:
            # db_path is relative by default, so the overwhelmingly likely cause
            # is a wrong working directory or a MINIMAPPR_DB_PATH that the server
            # has and this shell does not. Silently reporting success there leaves
            # the operator believing a live database was wiped when it was not.
            print(
                f"warning: no database found at {summary['db_path']} - nothing was wiped there. "
                "Re-run from the server's working directory or pass --db-path.",
                file=sys.stderr,
            )
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
