"""Taxonomy provider utilities for label/category/IFF mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from minimappr.interfaces import TaxonomyProvider

_logger = logging.getLogger(__name__)


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
    # "siren" spelled out: the former "sir" prefix matched any label containing
    # those three letters (e.g. a node named "sirith") and mislabelled it security.
    if any(token in value for token in ("gun", "glass", "alarm", "fire", "explosion", "impulse", "siren")):
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
        """Load the taxonomy map, degrading loudly rather than silently.

        Every failure here downgrades the whole system's labels to ``unknown``
        (and therefore ``iff_category="unknown"``), which is invisible in the
        output — it looks exactly like a classifier that never recognises
        anything. Each degradation path is logged at WARNING so the cause is
        discoverable from the logs instead of by inspecting detections.
        """
        if path is None:
            return cls()
        if not path.exists():
            # Not the same degradation as the malformed cases below: with no
            # explicit map, categories still resolve through the built-in name
            # heuristics and DEFAULT_CATEGORY_TO_IFF, which is why live
            # detections still carry real categories (e.g. "wildlife" ->
            # "friendly"). Only site-specific overrides are missing.
            _logger.info(
                "taxonomy config not found at %s; using built-in category "
                "defaults and name heuristics. Create it to override them",
                path,
            )
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning(
                "taxonomy config at %s could not be read (%s); all labels will "
                "resolve to category/IFF 'unknown'",
                path,
                exc,
            )
            return cls()
        if not isinstance(raw, dict):
            _logger.warning(
                "taxonomy config at %s must be a JSON object, got %s; all labels "
                "will resolve to category/IFF 'unknown'",
                path,
                type(raw).__name__,
            )
            return cls()
        label_to_category = _extract_map(raw.get("label_to_category"))
        if not label_to_category:
            _logger.warning(
                "taxonomy config at %s contains no usable 'label_to_category' "
                "entries; labels will fall back to name heuristics",
                path,
            )
        return cls(
            label_to_category=label_to_category,
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
