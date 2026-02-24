"""Label taxonomy helpers for flat Phase 1 runtime behavior."""

from __future__ import annotations


def label_category_for_name(label: str) -> str:
    value = label.strip().lower()
    if any(token in value for token in ("bird", "wild", "animal", "canid", "feline")):
        return "wildlife"
    if any(token in value for token in ("speech", "voice", "shout", "scream", "human")):
        return "human"
    if any(token in value for token in ("engine", "machine", "car", "truck", "vehicle", "drone")):
        return "vehicle"
    if any(token in value for token in ("gun", "glass", "alarm", "fire", "explosion", "impulse")):
        return "security"
    return "unknown"


def iff_for_category(category: str) -> str:
    value = category.strip().lower()
    if value in {"wildlife"}:
        return "friendly"
    if value in {"security"}:
        return "hostile"
    return "unknown"
