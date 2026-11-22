"""Taxonomy categories implied by BirdNET's own vocabulary (v2.4 and V3).

BirdNET labels are species names, and the substring name heuristics in
``core/taxonomy.py`` misread them ("scarlet tanager" contains "car" -> vehicle,
"common firecrest" contains "fire" -> security). Model provenance is the
reliable signal: every BirdNET species label is wildlife. Each model also has a
handful of non-species sound classes, which get an explicit category here
rather than whatever a substring happens to match.

Interface contract: ``birdnet_vocabulary_category`` returns a category for
every real BirdNET label and ``None`` for anything that is not a BirdNET result
(the caller then keeps the taxonomy's own answer). Abstentions are filtered by
the caller.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from minimappr.classifiers.base import ClassificationResult
from minimappr.classifiers.birdnet_v3_labels import (
    BIRDNET_V3_ANIMAL_TAXON_GROUPS,
    DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
)

BIRDNET_V24_MODEL_NAME = "birdnet_v2m4"
# Mirrors birdnet_v3.BIRDNET_V3_MODEL_NAME without importing the ONNX classifier
# module into the hot path.
BIRDNET_V3_MODEL_NAME = "birdnet_v3"

WILDLIFE_CATEGORY = "wildlife"

# v2.4's non-species classes, keyed on the pipeline label (lower-cased common
# name). In the label file they repeat the class name ("Gun_Gun"); so do two
# cricket species ("Gryllus assimilis_Gryllus assimilis"), which is why the
# lookup is an explicit table rather than a "scientific == common" rule.
BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES: dict[str, str] = {
    "dog": WILDLIFE_CATEGORY,
    "engine": "vehicle",
    "environmental": "unknown",
    "fireworks": "security",
    "gun": "security",
    "human non-vocal": "human",
    "human vocal": "human",
    "human whistle": "human",
    "noise": "unknown",
    "power tools": "human",
    "siren": "security",
}

# V3's sound-source classes (taxon groups Anthropogenic / Geophony / Music).
BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES: dict[str, str] = {
    "aircraft": "vehicle",
    "boat": "vehicle",
    "chainsaw": "human",
    "engine": "vehicle",
    "explosion": "security",
    "fire": "security",
    "fireworks": "security",
    "gun": "security",
    "hammer": "human",
    "motor vehicle": "vehicle",
    "music": "human",
    "power tool": "human",
    "sawing": "human",
    "siren": "security",
    "train": "vehicle",
    "vehicle horn": "vehicle",
}

# V3 files Homo sapiens under Mammalia; a person is not wildlife (IFF friendly).
BIRDNET_V3_HUMAN_SCIENTIFIC_NAME = "homo sapiens"
HUMAN_CATEGORY = "human"


def birdnet_vocabulary_category(classification: ClassificationResult) -> str | None:
    """Category implied by the BirdNET vocabulary, or None for non-BirdNET results."""
    model_name = classification.features.get("model")
    if not isinstance(model_name, str):
        return None
    normalized_model_name = model_name.strip().lower()
    label_key = classification.label.strip().lower()
    if normalized_model_name == BIRDNET_V3_MODEL_NAME:
        return _birdnet_v3_category(classification, label_key)
    if normalized_model_name == BIRDNET_V24_MODEL_NAME:
        return BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES.get(label_key, WILDLIFE_CATEGORY)
    return None


def _birdnet_v3_category(classification: ClassificationResult, label_key: str) -> str | None:
    taxon_group = classification.features.get("taxon_group")
    if taxon_group in BIRDNET_V3_ANIMAL_TAXON_GROUPS:
        scientific_name = str(classification.features.get("scientific_name") or "").strip().lower()
        if scientific_name == BIRDNET_V3_HUMAN_SCIENTIFIC_NAME:
            return HUMAN_CATEGORY
        return WILDLIFE_CATEGORY
    if taxon_group is None:
        # No taxon group means a result shape this table cannot vouch for.
        return None
    return BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES.get(label_key)


@lru_cache(maxsize=1)
def known_birdnet_animal_species_labels(
    label_map_path: Path = DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
) -> frozenset[str]:
    """Pipeline labels of every v2.4 animal species and its mapped V3 name.

    Read from the in-repo v2.4 -> V3 map (all 6522 v2.4 outputs). V3-only species
    are absent: the V3 labels file is a downloaded model asset, not in the repo.
    Used only to correct ``labels`` rows persisted with a heuristic category.
    """
    species_labels: set[str] = set()
    with Path(label_map_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            v24_label = row["v24_common_name"].strip().lower()
            if v24_label in BIRDNET_V24_NON_SPECIES_CLASS_CATEGORIES:
                continue
            species_labels.add(v24_label)
            v3_label = row["v3_common_name_en"].strip().lower()
            if v3_label:
                species_labels.add(v3_label)
    # Shared names with a non-wildlife meaning in either vocabulary stay out.
    species_labels.difference_update(BIRDNET_V3_SOUND_SOURCE_CLASS_CATEGORIES)
    species_labels.discard(HUMAN_CATEGORY)
    return frozenset(species_labels)


async def correct_persisted_birdnet_species_label_categories(storage, taxonomy_provider) -> list[str]:
    """Re-file ``labels`` rows for known BirdNET animal species as wildlife.

    Before the S-gate fix the name heuristics won for BirdNET species, so rows
    like ("northern cardinal", "vehicle") were persisted and reloaded into the
    taxonomy on every start. Only rows whose name is a known species label and
    whose category is not already wildlife are touched; a label the operator
    configured explicitly keeps its configured category. Idempotent. Historical
    ``detections.label_category`` values are left as recorded.
    """
    configured_category_for_label = getattr(taxonomy_provider, "configured_category_for_label", None)
    species_labels = known_birdnet_animal_species_labels()
    corrected_label_names: list[str] = []
    for row in await storage.list_labels():
        name = str(row.get("name") or "")
        if name.strip().lower() not in species_labels:
            continue
        if str(row.get("category") or "").strip().lower() == WILDLIFE_CATEGORY:
            continue
        if callable(configured_category_for_label) and configured_category_for_label(name) is not None:
            continue
        await storage.upsert_label(
            name=name,
            category=WILDLIFE_CATEGORY,
            source=str(row.get("source") or "runtime"),
            created_ns=int(row.get("created_ns") or 0),
        )
        corrected_label_names.append(name)
    return corrected_label_names
