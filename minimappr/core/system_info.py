"""Stdlib-only system diagnostics (no psutil dependency)."""

from __future__ import annotations

import os
import platform
import resource
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any


def collect(db_path: Path | str | None = None, start_ns: int | None = None) -> dict[str, Any]:
    """Runtime facts suitable for a system/diagnostics endpoint.

    Deliberately uses only the stdlib — runs identically on Linux and macOS dev boxes.
    """
    now_ns = time.time_ns()
    out: dict[str, Any] = {
        "now_ns": now_ns,
        "uptime_ns": now_ns - int(start_ns) if start_ns else None,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "process": {
            "pid": os.getpid(),
            "threads": threading.active_count(),
        },
    }

    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes; Linux reports kilobytes. Normalize to bytes using a heuristic.
        maxrss_bytes = ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024
        out["process"]["cpu_user_s"] = float(ru.ru_utime)
        out["process"]["cpu_system_s"] = float(ru.ru_stime)
        out["process"]["max_rss_bytes"] = int(maxrss_bytes)
    except Exception:
        pass

    try:
        la1, la5, la15 = os.getloadavg()
        out["load"] = {"1m": la1, "5m": la5, "15m": la15}
    except (AttributeError, OSError):
        out["load"] = None

    try:
        out["cpu_count"] = os.cpu_count()
    except Exception:
        out["cpu_count"] = None

    def _disk(path: Path | str) -> dict[str, int | str] | None:
        try:
            du = shutil.disk_usage(str(path))
            return {"path": str(path), "total": du.total, "used": du.used, "free": du.free}
        except Exception:
            return None

    disks: dict[str, Any] = {"cwd": _disk(Path.cwd())}
    if db_path:
        db_parent = Path(db_path).expanduser().parent
        if db_parent.exists():
            disks["db"] = _disk(db_parent)
    out["disk"] = disks

    return out
