"""Taxonomy provider utilities for label/category/IFF mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimappr.interfaces import TaxonomyProvider


DEFAULT_CATEGORY_TO_IFF = {
    "wildlife": "friendly",
    "security": "hostile",
    "human": "unknown",
    "vehicle": "unknown",
    "unknown": "unknown",
}


def _heuristic_category_for_name(label: str) -> str:
    value = label.strip().lower()
    if any(token in value for token in ("bird", "wild", "animal", "canid", "feline", "dog", "cat", "hawk", "coyote")):
        return "wildlife"
    if any(token in value for token in ("speech", "voice", "shout", "scream", "human", "talk")):
        return "human"
    if any(token in value for token in ("engine", "machine", "car", "truck", "vehicle", "drone", "aircraft")):
        return "vehicle"
    if any(token in value for token in ("gun", "glass", "alarm", "fire", "explosion", "impulse", "sir")):
        return "security"
    return "unknown"


class RuntimeTaxonomyProvider(TaxonomyProvider):
    def __init__(
        self,
        *,
        label_to_category: dict[str, str] | None = None,
        category_to_iff: dict[str, str] | None = None,
        label_aliases: dict[str, str] | None = None,
    ) -> None:
        self._label_aliases = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (label_aliases or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._label_to_category = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (label_to_category or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._category_to_iff = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (category_to_iff or DEFAULT_CATEGORY_TO_IFF).items()
            if str(key).strip() and str(value).strip()
        }

    @classmethod
    def from_config_file(cls, path: Path | None) -> "RuntimeTaxonomyProvider":
        if path is None or not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls(
            label_to_category=_extract_map(raw.get("label_to_category")),
            category_to_iff=_extract_map(raw.get("category_to_iff")),
            label_aliases=_extract_map(raw.get("label_aliases")),
        )

    def merge_labels(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            name = str(row.get("name") or "").strip().lower()
            category = str(row.get("category") or "").strip().lower()
            if not name or not category:
                continue
            self._label_to_category[name] = category

    def register_label(self, label: str, category: str) -> None:
        if not label.strip():
            return
        self._label_to_category[label.strip().lower()] = category.strip().lower() or "unknown"

    def canonical_label(self, label: str) -> str:
        """Resolve alias labels to their canonical form (identity when unmapped).

        Matching is label-level: canonicalization normalizes the *incoming*
        label before dedupe queries; rows already stored under a different
        alias are not re-keyed.
        """
        return self._label_aliases.get(label.strip().lower(), label)

    def category_for_label(self, label: str) -> str:
        key = label.strip().lower()
        if key in self._label_to_category:
            return self._label_to_category[key]
        return _heuristic_category_for_name(label)

    def iff_for_category(self, category: str) -> str:
        key = category.strip().lower() or "unknown"
        return self._category_to_iff.get(key, "unknown")


def _extract_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for key, item in value.items():
        name = str(key).strip().lower()
        mapped = str(item).strip().lower()
        if name and mapped:
            output[name] = mapped
    return output


_default_provider = RuntimeTaxonomyProvider()


def label_category_for_name(label: str) -> str:
    """Backward-compatible helper."""
    return _default_provider.category_for_label(label)


def iff_for_category(category: str) -> str:
    """Backward-compatible helper."""
    return _default_provider.iff_for_category(category)
