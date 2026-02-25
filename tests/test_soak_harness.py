from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _run_soak_subprocess(
    *,
    repo_root: Path,
    duration_seconds: float,
    tmp_path: Path,
    min_detections: int,
    min_full3d: int,
) -> subprocess.CompletedProcess[str]:
    port = _free_port()
    db_path = tmp_path / f"soak_{int(duration_seconds)}.db"
    snippet_dir = tmp_path / f"soak_{int(duration_seconds)}_snippets"
    cmd = [
        sys.executable,
        "scripts/run_soak.py",
        "--duration",
        str(duration_seconds),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--db-path",
        str(db_path),
        "--snippet-dir",
        str(snippet_dir),
        "--min-detections",
        str(min_detections),
        "--min-full3d",
        str(min_full3d),
        "--trigger-rms",
        "0.008",
        "--trigger-cooldown",
        "0.15",
    ]
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=max(180, int(duration_seconds * 2.0)),
        check=False,
    )


def test_soak_harness_short_run_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = _run_soak_subprocess(
        repo_root=repo_root,
        duration_seconds=8.0,
        tmp_path=tmp_path,
        min_detections=6,
        min_full3d=2,
    )
    if result.returncode != 0:
        pytest.fail(
            f"short soak failed with rc={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    assert "SOAK_RESULT=PASS" in result.stdout


@pytest.mark.skipif(
    os.getenv("MINIMAPPR_ENABLE_5MIN_SOAK") != "1",
    reason="Set MINIMAPPR_ENABLE_5MIN_SOAK=1 to run the full 5-minute soak test.",
)
def test_soak_harness_five_minute_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = _run_soak_subprocess(
        repo_root=repo_root,
        duration_seconds=300.0,
        tmp_path=tmp_path,
        min_detections=120,
        min_full3d=40,
    )
    if result.returncode != 0:
        pytest.fail(
            f"five-minute soak failed with rc={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    assert "SOAK_RESULT=PASS" in result.stdout
