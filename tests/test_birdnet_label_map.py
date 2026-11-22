"""v2.4 -> BirdNET+ V3.0 label-mapping table and builder."""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from minimappr.classifiers.birdnet_v3_labels import (
    DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
    MATCH_METHOD_UNMAPPED,
    BirdNETV3Label,
    build_v24_to_v3_mapping_rows,
    load_birdnet_v3_labels,
    load_v24_to_v3_label_map,
    write_v24_to_v3_label_map,
)

V3_LABELS_PATH = Path(__file__).resolve().parent.parent / "data/models/birdnet_v3/BirdNET+_V3.0_Global_12K_Labels.csv"
V24_LABEL_COUNT = 6522
# Ceiling, not a target: rebuilding the table must not silently lose matches.
MAX_UNMAPPED_V24_LABELS = 20


def _v3(index: int, scientific: str, common: str, group: str = "Aves") -> BirdNETV3Label:
    return BirdNETV3Label(index, f"BN{index:05d}", scientific, common, group)


def test_builder_match_tiers_in_order():
    v3_labels = [
        _v3(0, "Haemorhous mexicanus", "House Finch"),
        _v3(1, "Astur gentilis", "Eurasian Goshawk"),
        _v3(2, "Canis familiaris", "Domestic Dog", "Mammalia"),
        _v3(3, "Grus virgo", "Demoiselle Crane"),
    ]
    rows = build_v24_to_v3_mapping_rows(
        [
            "Haemorhous mexicanus_House Finch",
            "Accipiter gentilis_Northern Goshawk",
            "Dog_Dog",
            "Anthropoides virgo_Demoiselle Crane",
            "Noise_Noise",
        ],
        v3_labels,
        {"accipiter gentilis": "BN00001"},
    )
    assert [(row.v3_index, row.match_method) for row in rows] == [
        (0, "scientific_name"),
        (1, "taxonomy_alias"),
        (2, "override"),
        (3, "common_name_en"),
        (None, MATCH_METHOD_UNMAPPED),
    ]


def test_ambiguous_common_name_does_not_match():
    v3_labels = [_v3(0, "A a", "Gray Bird"), _v3(1, "B b", "Gray Bird")]
    (row,) = build_v24_to_v3_mapping_rows(["C c_Gray Bird"], v3_labels)
    assert row.match_method == MATCH_METHOD_UNMAPPED


def test_write_then_load_round_trip(tmp_path):
    rows = build_v24_to_v3_mapping_rows(["Haemorhous mexicanus_House Finch", "Noise_Noise"], [_v3(7, "Haemorhous mexicanus", "House Finch")])
    path = tmp_path / "map.csv"
    write_v24_to_v3_label_map(rows, path)
    loaded = load_v24_to_v3_label_map(path)
    assert loaded["Haemorhous mexicanus_House Finch"].v3_index == 7
    assert not loaded["Noise_Noise"].is_mapped


def test_committed_table_covers_every_v24_label():
    mapping = load_v24_to_v3_label_map(DEFAULT_V24_TO_V3_LABEL_MAP_PATH)
    assert len(mapping) == V24_LABEL_COUNT
    counts = collections.Counter(row.match_method for row in mapping.values())
    assert counts[MATCH_METHOD_UNMAPPED] <= MAX_UNMAPPED_V24_LABELS
    assert mapping["Haemorhous mexicanus_House Finch"].v3_common_name_en == "House Finch"
    assert mapping["Turdus migratorius_American Robin"].match_method == "scientific_name"
    # Genus move resolved through the taxonomy synonym list.
    assert mapping["Corvus monedula_Eurasian Jackdaw"].v3_scientific_name == "Coloeus monedula"
    assert mapping["Human vocal_Human vocal"].v3_scientific_name == "Homo sapiens"
    assert not mapping["Noise_Noise"].is_mapped


def test_committed_table_targets_exist_in_v3_labels():
    if not V3_LABELS_PATH.is_file():
        pytest.skip(f"V3 labels not fetched ({V3_LABELS_PATH}); run: python scripts/fetch_birdnet_v3.py")
    v3_by_index = {label.index: label for label in load_birdnet_v3_labels(V3_LABELS_PATH)}
    for row in load_v24_to_v3_label_map().values():
        if row.is_mapped:
            target = v3_by_index[row.v3_index]
            assert (target.birdnet_id, target.scientific_name) == (row.v3_birdnet_id, row.v3_scientific_name)
