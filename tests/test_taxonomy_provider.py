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
