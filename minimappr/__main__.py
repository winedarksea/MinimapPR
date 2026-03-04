"""Allow running MinimapPR with ``python -m minimappr``."""

from __future__ import annotations

import uvicorn

from minimappr.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "minimappr.main:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
