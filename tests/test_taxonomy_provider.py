from __future__ import annotations

import json
from pathlib import Path

from minimappr.core.taxonomy import RuntimeTaxonomyProvider


def test_taxonomy_provider_loads_file_and_merges_labels(tmp_path: Path) -> None:
    config = tmp_path / "taxonomy.json"
    config.write_text(
        json.dumps(
            {
                "label_to_category": {
                    "fox": "wildlife",
                    "generator_hum": "vehicle",
                },
                "category_to_iff": {
                    "wildlife": "friendly",
                    "vehicle": "unknown",
                    "security": "hostile",
                },
            }
        ),
        encoding="utf-8",
    )

    provider = RuntimeTaxonomyProvider.from_config_file(config)
    assert provider.category_for_label("fox") == "wildlife"
    assert provider.iff_for_category("wildlife") == "friendly"
    assert provider.category_for_label("gunshot") == "security"
    assert provider.iff_for_category("security") == "hostile"

    provider.merge_labels([{"name": "glass_break", "category": "security"}])
    assert provider.category_for_label("glass_break") == "security"


def test_coyote_heuristic_maps_to_wildlife() -> None:
    provider = RuntimeTaxonomyProvider()
    assert provider.category_for_label("coyote") == "wildlife"
    assert provider.iff_for_category("wildlife") == "friendly"


def test_missing_taxonomy_file_still_resolves_categories(tmp_path: Path, caplog) -> None:
    """A missing config is not the same degradation as a malformed one.

    The old WARNING claimed every label would resolve to 'unknown', which sent a
    production investigation down the wrong path: built-in name heuristics and
    DEFAULT_CATEGORY_TO_IFF still resolve real categories without the file.
    """
    import logging

    missing = tmp_path / "absent-taxonomy.json"
    with caplog.at_level(logging.WARNING, logger="minimappr.core.taxonomy"):
        provider = RuntimeTaxonomyProvider.from_config_file(missing)

    assert provider.category_for_label("coyote") == "wildlife"
    assert provider.iff_for_category("wildlife") == "friendly"
    # Absence of the optional override file is not a warning-level condition.
    assert caplog.records == []


def test_abstention_label_cannot_be_pinned_at_runtime() -> None:
    """"unknown" is the absence of a label and must never learn a category.

    A single BirdNET abstention resolving to "wildlife" used to be registered
    against the label name, after which the pinned mapping won the lookup ahead
    of the name heuristics and *every* later abstention — from every model —
    read as wildlife.
    """
    provider = RuntimeTaxonomyProvider()

    provider.register_label("unknown", "wildlife")

    assert provider.category_for_label("unknown") == "unknown"
    assert provider.iff_for_category("unknown") == "unknown"


def test_abstention_label_is_not_reloaded_from_persisted_labels() -> None:
    """A DB poisoned by the old behaviour self-heals on restart.

    ``merge_labels`` replays the labels table into the provider at startup, so
    without this guard the bad row outlived the code fix.
    """
    provider = RuntimeTaxonomyProvider()

    provider.merge_labels(
        [
            {"name": "unknown", "category": "wildlife"},
            {"name": "great horned owl", "category": "wildlife"},
        ]
    )

    assert provider.category_for_label("unknown") == "unknown"
    # Real labels still merge normally.
    assert provider.category_for_label("great horned owl") == "wildlife"


def test_explicit_config_may_still_map_the_abstention_label(tmp_path: Path) -> None:
    """Only *learned* mappings are refused; operator config stays authoritative."""
    config = tmp_path / "taxonomy.json"
    config.write_text(
        json.dumps({"label_to_category": {"unknown": "security"}}), encoding="utf-8"
    )

    provider = RuntimeTaxonomyProvider.from_config_file(config)

    assert provider.category_for_label("unknown") == "security"
