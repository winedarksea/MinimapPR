"""Exhaustive BirdNET label -> taxonomy category table (S-gate review).

Every BirdNET species must resolve to wildlife regardless of substring
heuristics ("scarlet tanager" -> "car"); the models' non-species sound classes
must keep a deliberate non-wildlife category; YAMNet must be untouched.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from minimappr.classifiers.base import ClassificationResult
from minimappr.classifiers.birdnet_v3_labels import (
    BIRDNET_V3_ANIMAL_TAXON_GROUPS,
    DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
    load_birdnet_v3_labels,
)
from minimappr.classifiers.yamnet import _parse_class_map_csv
from minimappr.classifiers.yamnet_model import yamnet_model_asset_directory
from minimappr.config import DEFAULT_BIRDNET_V3_LABELS_PATH
from minimappr.core.birdnet_label_categories import (
    BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES,
    BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES,
    correct_persisted_birdnet_species_label_categories,
    known_birdnet_animal_species_labels,
)
from minimappr.core.classification import _resolve_label_category
from minimappr.core.taxonomy import RuntimeTaxonomyProvider
from minimappr.storage.db import Storage

# The V3 labels CSV is a downloaded model asset (not in the repo); point this at
# it to run the full-file V3 check outside a provisioned checkout.
BIRDNET_V3_LABELS_ENVIRONMENT_VARIABLE = "MINIMAPPR_TEST_BIRDNET_V3_LABELS_CSV"

# The V3 non-animal rows (Anthropogenic / Geophony / Music) as shipped in
# BirdNET+_V3.0_Global_12K_Labels.csv; pinned so the table is tested without it.
BIRDNET_V3_NON_ANIMAL_ROWS = (
    ("Gun", "Anthropogenic"), ("Chainsaw", "Anthropogenic"), ("Aircraft", "Anthropogenic"),
    ("Boat", "Anthropogenic"), ("Explosion", "Anthropogenic"), ("Engine", "Anthropogenic"),
    ("Fire", "Geophony"), ("Fireworks", "Anthropogenic"), ("Hammer", "Anthropogenic"),
    ("Motor vehicle", "Anthropogenic"), ("Music", "Music"), ("Power tool", "Anthropogenic"),
    ("Train", "Anthropogenic"), ("Sawing", "Anthropogenic"), ("Siren", "Anthropogenic"),
    ("Vehicle horn", "Anthropogenic"),
)


def _v24_rows() -> list[dict[str, str]]:
    with DEFAULT_V24_TO_V3_LABEL_MAP_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _v24_result(row: dict[str, str]) -> ClassificationResult:
    # Mirrors classifiers/birdnet.py: label = lower-cased common name.
    return ClassificationResult(
        label=row["v24_common_name"].strip().lower(),
        confidence=0.9,
        features={"model": "birdnet_v2m4", "raw_species": row["v24_label"]},
    )


def _v3_result(common_name: str, scientific_name: str, taxon_group: str) -> ClassificationResult:
    # Mirrors classifiers/birdnet_v3.py result features.
    return ClassificationResult(
        label=common_name.strip().lower(),
        confidence=0.9,
        features={
            "model": "birdnet_v3",
            "raw_species": f"{scientific_name}_{common_name}",
            "scientific_name": scientific_name,
            "taxon_group": taxon_group,
        },
    )


def test_every_birdnet_v24_label_resolves_to_its_intended_category() -> None:
    provider = RuntimeTaxonomyProvider()
    rows = _v24_rows()
    assert len(rows) == 6522
    misfiled_species: list[tuple[str, str]] = []
    non_species_categories: dict[str, str] = {}
    for row in rows:
        result = _v24_result(row)
        category = _resolve_label_category(result, provider)
        if result.label in BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES:
            non_species_categories[result.label] = category
        elif category != "wildlife":
            misfiled_species.append((result.label, category))
    assert misfiled_species == []
    assert non_species_categories == BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES
    assert non_species_categories["gun"] == "security"
    assert non_species_categories["human vocal"] == "human"
    assert non_species_categories["engine"] == "vehicle"


def test_birdnet_v24_non_species_table_matches_the_label_file() -> None:
    # Every v2.4 "Name_Name" row is either a table entry or one of two crickets.
    repeated_name_rows = {
        row["v24_common_name"].strip().lower()
        for row in _v24_rows()
        if row["v24_scientific_name"].strip().lower() == row["v24_common_name"].strip().lower()
    }
    assert repeated_name_rows - set(BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES) == {
        "gryllus assimilis",
        "miogryllus saussurei",
    }
    assert set(BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES) <= repeated_name_rows


def test_birdnet_v3_mapped_species_and_sound_sources_resolve_correctly() -> None:
    provider = RuntimeTaxonomyProvider()
    misfiled_species = [
        (row["v3_common_name_en"], category)
        for row in _v24_rows()
        if row["v3_common_name_en"] and row["v24_common_name"].strip().lower() not in BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES
        for category in [
            _resolve_label_category(_v3_result(row["v3_common_name_en"], row["v3_scientific_name"], "Aves"), provider)
        ]
        if category != "wildlife"
    ]
    assert misfiled_species == []
    sound_source_categories = {
        name.lower(): _resolve_label_category(_v3_result(name, name, group), provider)
        for name, group in BIRDNET_V3_NON_ANIMAL_ROWS
    }
    assert sound_source_categories == BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES
    assert "wildlife" not in sound_source_categories.values()
    assert _resolve_label_category(_v3_result("Human", "Homo sapiens", "Mammalia"), provider) == "human"
    assert _resolve_label_category(_v3_result("Domestic Dog", "Canis familiaris", "Mammalia"), provider) == "wildlife"


def _birdnet_v3_labels_csv_path() -> Path:
    override = os.environ.get(BIRDNET_V3_LABELS_ENVIRONMENT_VARIABLE)
    return Path(override) if override else DEFAULT_BIRDNET_V3_LABELS_PATH


@pytest.mark.skipif(not _birdnet_v3_labels_csv_path().exists(), reason="BirdNET V3 labels CSV not provisioned")
def test_every_birdnet_v3_label_in_the_model_file_resolves_to_its_intended_category() -> None:
    provider = RuntimeTaxonomyProvider()
    labels = load_birdnet_v3_labels(_birdnet_v3_labels_csv_path())
    misfiled_animals: list[tuple[str, str]] = []
    non_animal_rows: list[tuple[str, str]] = []
    for label in labels:
        category = _resolve_label_category(
            _v3_result(label.common_name_en, label.scientific_name, label.taxon_group), provider
        )
        if label.taxon_group not in BIRDNET_V3_ANIMAL_TAXON_GROUPS:
            non_animal_rows.append((label.common_name_en, label.taxon_group))
            assert category == BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES[label.pipeline_label]
        elif label.scientific_name == "Homo sapiens":
            assert category == "human"
        elif category != "wildlife":
            misfiled_animals.append((label.pipeline_label, category))
    assert misfiled_animals == []
    assert sorted(non_animal_rows) == sorted(BIRDNET_V3_NON_ANIMAL_ROWS)


def test_yamnet_labels_keep_the_taxonomy_answer() -> None:
    provider = RuntimeTaxonomyProvider()
    class_map_text = (yamnet_model_asset_directory() / "assets" / "yamnet_class_map.csv").read_text(encoding="utf-8")
    yamnet_labels = _parse_class_map_csv(class_map_text)
    assert len(yamnet_labels) == 521
    for label in yamnet_labels:
        result = ClassificationResult(label=label, confidence=0.9, features={"model": "yamnet"})
        assert _resolve_label_category(result, provider) == provider.category_for_label(label)


def test_learned_label_rows_do_not_override_species_provenance_but_configuration_does() -> None:
    cardinal = ClassificationResult(
        label="northern cardinal",
        confidence=0.9,
        features={"model": "birdnet_v2m4", "raw_species": "Cardinalis cardinalis_Northern Cardinal"},
    )
    learned_provider = RuntimeTaxonomyProvider()
    # A persisted pre-fix row, reloaded on start, plus a runtime registration.
    learned_provider.merge_labels([{"name": "northern cardinal", "category": "vehicle"}])
    learned_provider.register_label("scarlet tanager", "vehicle")
    assert _resolve_label_category(cardinal, learned_provider) == "wildlife"
    configured_provider = RuntimeTaxonomyProvider(label_to_category={"northern cardinal": "security"})
    assert _resolve_label_category(cardinal, configured_provider) == "security"


@pytest.mark.asyncio
async def test_startup_correction_refiles_only_known_species_rows(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "labels.db")
    await storage.initialize()
    try:
        seeded_rows = {
            "northern cardinal": ("vehicle", "birdnet"),
            "scarlet tanager": ("vehicle", "review"),
            "common firecrest": ("security", "birdnet"),
            "canada warbler": ("vehicle", "birdnet"),  # operator-configured below
            "american robin": ("wildlife", "birdnet"),
            "Engine": ("vehicle", "yamnet"),
            "gun": ("security", "birdnet"),
            "car alarm": ("vehicle", "yamnet"),
        }
        for name, (category, source) in seeded_rows.items():
            await storage.upsert_label(name=name, category=category, source=source, created_ns=1)
        provider = RuntimeTaxonomyProvider(label_to_category={"canada warbler": "vehicle"})

        corrected = await correct_persisted_birdnet_species_label_categories(storage, provider)
        assert sorted(corrected) == ["common firecrest", "northern cardinal", "scarlet tanager"]
        stored = {row["name"]: (row["category"], row["source"]) for row in await storage.list_labels()}
        assert stored["northern cardinal"] == ("wildlife", "birdnet")
        assert stored["scarlet tanager"] == ("wildlife", "review")
        assert stored["canada warbler"] == ("vehicle", "birdnet")
        for untouched in ("american robin", "Engine", "gun", "car alarm"):
            assert stored[untouched] == seeded_rows[untouched]

        assert await correct_persisted_birdnet_species_label_categories(storage, provider) == []
    finally:
        await storage.close()


def test_known_species_labels_exclude_every_non_wildlife_class_name() -> None:
    species_labels = known_birdnet_animal_species_labels()
    assert "northern cardinal" in species_labels
    assert "gryllus assimilis" in species_labels
    for non_wildlife_name, category in {
        **BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES,
        **BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES,
    }.items():
        if category != "wildlife":
            assert non_wildlife_name not in species_labels
    assert "human" not in species_labels
