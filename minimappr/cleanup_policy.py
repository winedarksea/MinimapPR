"""Policy models for operator-managed snippet and artifact cleanup."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _normalize_label(label: str | None) -> str:
    return (label or "").strip().lower()


@dataclass(slots=True)
class LabelCleanupOverride:
    snippet_max_age_seconds: int | None = None
    artifact_max_age_seconds: int | None = None
    keep_snippets: bool = False
    keep_artifacts: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelCleanupOverride":
        snippet_max_age_seconds = payload.get("snippet_max_age_seconds")
        artifact_max_age_seconds = payload.get("artifact_max_age_seconds")
        keep_snippets = bool(payload.get("keep_snippets", False))
        keep_artifacts = bool(payload.get("keep_artifacts", False))
        if snippet_max_age_seconds is not None and int(snippet_max_age_seconds) < 0:
            raise ValueError("snippet_max_age_seconds must be >= 0 when provided")
        if artifact_max_age_seconds is not None and int(artifact_max_age_seconds) < 0:
            raise ValueError("artifact_max_age_seconds must be >= 0 when provided")
        return cls(
            snippet_max_age_seconds=int(snippet_max_age_seconds) if snippet_max_age_seconds is not None else None,
            artifact_max_age_seconds=int(artifact_max_age_seconds) if artifact_max_age_seconds is not None else None,
            keep_snippets=keep_snippets,
            keep_artifacts=keep_artifacts,
        )


@dataclass(slots=True)
class CleanupPolicy:
    default_snippet_max_age_seconds: int
    default_artifact_max_age_seconds: int
    keep_labels: dict[str, LabelCleanupOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_snippet_max_age_seconds < 0:
            raise ValueError("default_snippet_max_age_seconds must be >= 0")
        if self.default_artifact_max_age_seconds < 0:
            raise ValueError("default_artifact_max_age_seconds must be >= 0")
        normalized: dict[str, LabelCleanupOverride] = {}
        for raw_label, override in self.keep_labels.items():
            label = _normalize_label(raw_label)
            if not label:
                continue
            normalized[label] = override
        self.keep_labels = normalized

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_snippet_max_age_seconds: int,
        default_artifact_max_age_seconds: int,
    ) -> "CleanupPolicy":
        snippet_age = int(payload.get("default_snippet_max_age_seconds", default_snippet_max_age_seconds))
        artifact_age = int(payload.get("default_artifact_max_age_seconds", default_artifact_max_age_seconds))
        keep_labels_raw = payload.get("keep_labels", {})
        if not isinstance(keep_labels_raw, dict):
            raise ValueError("keep_labels must be an object when provided")
        keep_labels = {
            _normalize_label(label): LabelCleanupOverride.from_dict(value if isinstance(value, dict) else {})
            for label, value in keep_labels_raw.items()
            if _normalize_label(label)
        }
        return cls(
            default_snippet_max_age_seconds=snippet_age,
            default_artifact_max_age_seconds=artifact_age,
            keep_labels=keep_labels,
        )

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        default_snippet_max_age_seconds: int,
        default_artifact_max_age_seconds: int,
    ) -> "CleanupPolicy":
        if not path.exists():
            return cls(
                default_snippet_max_age_seconds=default_snippet_max_age_seconds,
                default_artifact_max_age_seconds=default_artifact_max_age_seconds,
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Cleanup policy at {path} must be a JSON object")
        return cls.from_dict(
            payload,
            default_snippet_max_age_seconds=default_snippet_max_age_seconds,
            default_artifact_max_age_seconds=default_artifact_max_age_seconds,
        )

    def override_for_label(self, label: str | None) -> LabelCleanupOverride | None:
        return self.keep_labels.get(_normalize_label(label))

    def snippet_max_age_seconds_for_label(self, label: str | None) -> int | None:
        override = self.override_for_label(label)
        if override is None:
            return self.default_snippet_max_age_seconds
        if override.keep_snippets:
            return None
        return (
            override.snippet_max_age_seconds
            if override.snippet_max_age_seconds is not None
            else self.default_snippet_max_age_seconds
        )

    def artifact_max_age_seconds_for_label(self, label: str | None) -> int | None:
        override = self.override_for_label(label)
        if override is None:
            return self.default_artifact_max_age_seconds
        if override.keep_artifacts:
            return None
        return (
            override.artifact_max_age_seconds
            if override.artifact_max_age_seconds is not None
            else self.default_artifact_max_age_seconds
        )
