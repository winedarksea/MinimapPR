#!/usr/bin/env python3
"""Run a local end-to-end soak test against a temporary MinimapPR server."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minimappr.sim.soak import (
    SoakConfig,
    SoakExpectations,
    detection_breakdown,
    evaluate_soak,
    poll_fusion_health,
    stream_four_point_nodes,
    wait_for_server,
)


async def run_soak(config: SoakConfig, expectations: SoakExpectations) -> int:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.snippet_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-shm", "-wal"):
        (Path(str(config.db_path) + suffix)).unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MINIMAPPR_DB_PATH": str(config.db_path),
            "MINIMAPPR_SNIPPET_DIR": str(config.snippet_dir),
            "MINIMAPPR_TRIGGER_RMS": str(config.trigger_rms),
            "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": str(config.trigger_cooldown_seconds),
            "MINIMAPPR_FUSION_WORKER_COUNT": str(config.worker_count),
            "MINIMAPPR_SNIPPET_RETENTION_SECONDS": "0",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "minimappr.main:app",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    server = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    base_url = f"http://{config.host}:{config.port}"
    poll_stop = asyncio.Event()
    poll_task: asyncio.Task | None = None
    try:
        await wait_for_server(base_url, timeout_seconds=30.0)
        async with httpx.AsyncClient(timeout=4.0) as client:
            poll_task = asyncio.create_task(
                poll_fusion_health(
                    client=client,
                    base_url=base_url,
                    poll_interval_seconds=config.poll_interval_seconds,
                    stop=poll_stop,
                )
            )
            ingest_errors = await stream_four_point_nodes(
                client=client,
                base_url=base_url,
                duration_seconds=config.duration_seconds,
                sample_rate_hz=config.sample_rate_hz,
                frame_size=config.frame_size,
            )

            await asyncio.sleep(2.0)
            final_status = (await client.get(f"{base_url}/api/v1/fusion/status")).json()
            poll_stop.set()
            poll_samples, health_errors = await poll_task

        detections_total, full_3d_detections, localization_methods = detection_breakdown(config.db_path)
        errors = evaluate_soak(
            final_status=final_status,
            poll_samples=poll_samples,
            health_errors=health_errors,
            ingestion_errors=ingest_errors,
            detections_total=detections_total,
            full_3d_detections=full_3d_detections,
            expectations=expectations,
        )

        print(f"duration_seconds={config.duration_seconds:.1f}")
        print(f"poll_samples={len(poll_samples)}")
        print(f"detections_total={detections_total}")
        print(f"full_3d_detections={full_3d_detections}")
        print(f"localization_methods={localization_methods}")
        print(f"final_metrics={final_status.get('metrics', {})}")

        if errors:
            print("SOAK_RESULT=FAIL")
            for err in errors:
                print(f"error: {err}")
            return 1
        print("SOAK_RESULT=PASS")
        return 0
    finally:
        if poll_task is not None and not poll_task.done():
            poll_stop.set()
            with contextlib.suppress(Exception):
                await poll_task
        if server.returncode is None:
            server.terminate()
            try:
                await asyncio.wait_for(server.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                server.kill()
                await server.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local MinimapPR soak test")
    parser.add_argument("--duration", type=float, default=300.0, help="Soak duration in seconds")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for temporary uvicorn instance")
    parser.add_argument("--port", type=int, default=8099, help="Bind port for temporary uvicorn instance")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Synthetic ingest sample rate")
    parser.add_argument("--frame-size", type=int, default=1024, help="Synthetic ingest frame size")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Fusion status poll cadence")
    parser.add_argument("--trigger-rms", type=float, default=0.01, help="Server trigger RMS for soak run")
    parser.add_argument("--trigger-cooldown", type=float, default=0.2, help="Server trigger cooldown seconds")
    parser.add_argument("--workers", type=int, default=1, help="Server fusion worker count")
    parser.add_argument("--min-detections", type=int, default=25, help="Minimum detections expected")
    parser.add_argument("--min-full3d", type=int, default=5, help="Minimum full_3d detections expected")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite DB path for the temporary run (default: tmpdir/minimappr_soak_<ts>.db)",
    )
    parser.add_argument(
        "--snippet-dir",
        type=Path,
        default=None,
        help="Snippet dir for the temporary run (default: sibling tmpdir path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.db_path is None:
        ts = int(time.time())
        base = Path(tempfile.gettempdir())
        db_path = base / f"minimappr_soak_{ts}.db"
    else:
        db_path = args.db_path
    snippet_dir = args.snippet_dir or db_path.with_suffix("").with_name(f"{db_path.stem}_snippets")

    config = SoakConfig(
        duration_seconds=max(1.0, float(args.duration)),
        host=args.host,
        port=args.port,
        sample_rate_hz=max(8_000, int(args.sample_rate)),
        frame_size=max(128, int(args.frame_size)),
        poll_interval_seconds=max(0.25, float(args.poll_interval)),
        db_path=db_path,
        snippet_dir=snippet_dir,
        trigger_rms=max(1e-6, float(args.trigger_rms)),
        trigger_cooldown_seconds=max(0.0, float(args.trigger_cooldown)),
        worker_count=max(1, int(args.workers)),
    )
    expectations = SoakExpectations(
        min_detections=max(1, int(args.min_detections)),
        min_full_3d_detections=max(0, int(args.min_full3d)),
    )
    raise SystemExit(asyncio.run(run_soak(config, expectations)))


if __name__ == "__main__":
    main()
