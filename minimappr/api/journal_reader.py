"""Handle reader for persisted derived sidecar payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class JournalHandleReadError(RuntimeError):
    """Raised when a journal handle cannot be reopened or verified."""


@dataclass(frozen=True, slots=True)
class JournalPayloadHandle:
    journal_epoch: int
    segment_id: str
    stream_key: str
    payload_offset_bytes: int
    payload_length_bytes: int
    sample_index_start: int | None
    sample_count: int | None
    integrity_hash: str
    segment_path: Path

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, journal_streams_dir: Path | None = None) -> "JournalPayloadHandle":
        stream_key = str(payload["stream_key"])
        segment_id = str(payload["segment_id"])
        segment_path_value = payload.get("segment_path")
        if segment_path_value is None:
            if journal_streams_dir is None:
                raise JournalHandleReadError("journal handle omitted segment_path and no journal_streams_dir was supplied")
            segment_path = journal_streams_dir / stream_key / "segments" / f"{segment_id}.bin"
        else:
            segment_path = Path(str(segment_path_value))
        return cls(
            journal_epoch=int(payload["journal_epoch"]),
            segment_id=segment_id,
            stream_key=stream_key,
            payload_offset_bytes=int(
                payload["payload_offset_bytes"]
                if "payload_offset_bytes" in payload
                else payload["body_offset_bytes"]
            ),
            payload_length_bytes=int(
                payload["payload_length_bytes"]
                if "payload_length_bytes" in payload
                else payload["body_length_bytes"]
            ),
            sample_index_start=_optional_int(payload.get("sample_index_start")),
            sample_count=_optional_int(payload.get("sample_count")),
            integrity_hash=str(payload.get("integrity_hash") or ""),
            segment_path=segment_path,
        )

    def read_bytes(self) -> bytes:
        if self.payload_offset_bytes < 0 or self.payload_length_bytes < 0:
            raise JournalHandleReadError("journal handle byte range must be non-negative")
        if self.payload_length_bytes == 0:
            payload = b""
            _verify_integrity_hash(payload, self.integrity_hash)
            return payload
        try:
            segment_size = self.segment_path.stat().st_size
        except FileNotFoundError as exc:
            raise JournalHandleReadError(f"journal segment {self.segment_path} is missing or evicted") from exc
        end_offset = self.payload_offset_bytes + self.payload_length_bytes
        if end_offset > segment_size:
            raise JournalHandleReadError(
                f"journal handle range {self.payload_offset_bytes}..{end_offset} exceeds segment size {segment_size}"
            )
        with self.segment_path.open("rb") as handle:
            handle.seek(self.payload_offset_bytes)
            payload = handle.read(self.payload_length_bytes)
        _verify_integrity_hash(payload, self.integrity_hash)
        return payload


def _verify_integrity_hash(payload: bytes, integrity_hash: str) -> None:
    if not integrity_hash:
        return
    expected = integrity_hash.removeprefix("sha256:").lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise JournalHandleReadError(f"journal payload integrity hash mismatch: expected {expected}, got {actual}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
