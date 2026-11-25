"""A session artifact directory read as the bundle zip it would export (plan item E9).

``write_bundle_zip`` copies the artifact directory's files into the zip under
the same relative paths and adds the ground-truth payload. Validation only
reads members, so it can read them straight from the directory: no capture
WAV is re-deflated on every validation request, and no half-written zip is
ever touched. Only the members the zip would hold are visible, so a manifest
path outside them reads as missing, exactly as in the zip.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO

from minimappr.calibration.frame_timing import FRAME_TIMING_BUNDLE_DIRECTORY

GROUND_TRUTH_MEMBER = "ground_truth.json"


class ArtifactDirectoryBundleMembers:
    """The ``namelist`` / ``read`` / ``open`` subset of ``zipfile.ZipFile`` that
    bundle validation uses, over a session artifact directory."""

    def __init__(self, artifact_dir: Path, ground_truth_payload: dict) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._ground_truth_bytes = json.dumps(ground_truth_payload).encode("utf-8")
        self._member_paths: dict[str, Path] = {}
        for member_name in ("manifest.json", "detections.json"):
            if (self._artifact_dir / member_name).is_file():
                self._member_paths[member_name] = self._artifact_dir / member_name
        for directory, pattern in (("audio", "*.wav"), (FRAME_TIMING_BUNDLE_DIRECTORY, "*.json")):
            if (self._artifact_dir / directory).is_dir():
                for path in sorted((self._artifact_dir / directory).glob(pattern)):
                    self._member_paths[f"{directory}/{path.name}"] = path

    def namelist(self) -> list[str]:
        return [*self._member_paths, GROUND_TRUTH_MEMBER]

    def read(self, member_name: str) -> bytes:
        if member_name == GROUND_TRUTH_MEMBER:
            return self._ground_truth_bytes
        return self._member_path(member_name).read_bytes()

    def open(self, member_name: str) -> BinaryIO:
        if member_name == GROUND_TRUTH_MEMBER:
            return io.BytesIO(self._ground_truth_bytes)
        return self._member_path(member_name).open("rb")

    def _member_path(self, member_name: str) -> Path:
        if member_name not in self._member_paths:
            raise KeyError(f"There is no item named {member_name!r} in the bundle")
        return self._member_paths[member_name]


__all__ = ["ArtifactDirectoryBundleMembers"]
