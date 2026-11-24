"""Blocking JSON request to the ingest sidecar that reports failure instead of hiding it.

Callers run it via ``asyncio.to_thread`` so the event loop never waits on the
sidecar. Unlike the status-poll helper in ``main.py`` (which maps every failure
to ``{}`` because a missing counter is harmless), commands such as a geometry
push or an audio override must know whether the sidecar actually took them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SidecarRequestError(RuntimeError):
    """The sidecar did not accept the request.

    ``status_code`` is None for transport errors; ``response_body`` is the
    decoded JSON error object when the sidecar sent one (for example the
    applied revision on a 409 stale-revision rejection).
    """

    def __init__(self, message: str, *, status_code: int | None = None, response_body: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def request_sidecar_json(
    base_url: str,
    path: str,
    *,
    body: dict | None,
    timeout_seconds: float,
) -> dict:
    """GET (``body`` None) or POST JSON; return the decoded object reply or raise."""
    url = f"{base_url.rstrip('/')}{path}"
    if body is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(body, allow_nan=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_reply = response.read()
    except urllib.error.HTTPError as exc:
        raw_detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
        try:
            decoded_detail = json.loads(raw_detail) if raw_detail else None
        except ValueError:
            decoded_detail = None
        raise SidecarRequestError(
            f"{path} returned HTTP {exc.code}: {raw_detail[:200]}".rstrip(": "),
            status_code=exc.code,
            response_body=decoded_detail if isinstance(decoded_detail, dict) else None,
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise SidecarRequestError(f"{path} unreachable: {exc}") from exc
    try:
        reply = json.loads(raw_reply.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SidecarRequestError(f"{path} returned a non-JSON reply") from exc
    if not isinstance(reply, dict):
        raise SidecarRequestError(f"{path} returned {type(reply).__name__}, expected a JSON object")
    return reply
