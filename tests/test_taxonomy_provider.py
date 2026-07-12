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
